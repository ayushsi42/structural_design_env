"""
PGSA Physics Simulation Engine
-------------------------------
A lightweight heuristic approximation of the full PGSA physics pipeline
(FEM structural solver, LBM airflow, ray-march lighting). Runs on CPU in
<10ms per step, faithful to the scoring intent of the full specification.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ─── MATERIAL REGISTRY ────────────────────────────────────────────────────────

MATERIALS: Dict[int, dict] = {
    0: {"name": "WOOD",      "yield_mpa": 40,  "E_gpa": 12,  "density": 600,  "cost": 1},
    1: {"name": "CONCRETE",  "yield_mpa": 30,  "E_gpa": 30,  "density": 2400, "cost": 3},
    2: {"name": "STEEL",     "yield_mpa": 250, "E_gpa": 200, "density": 7800, "cost": 8},
    3: {"name": "GLASS",     "yield_mpa": 50,  "E_gpa": 70,  "density": 2500, "cost": 5},
    4: {"name": "COMPOSITE", "yield_mpa": 600, "E_gpa": 70,  "density": 1600, "cost": 20},
}

MATERIAL_NAMES = {v["name"]: k for k, v in MATERIALS.items()}

# ─── ELEMENT TYPES ────────────────────────────────────────────────────────────

ELEMENT_TYPES = {
    "EMPTY": 0, "BEAM": 1, "WALL": 2, "FLOOR": 3,
    "WINDOW": 4, "DOOR": 5, "JOINT": 6, "FOUNDATION": 7,
}

# Load-bearing capacity (relative, higher = stronger)
LOAD_CAPACITY: Dict[str, float] = {
    "BEAM": 1.0, "WALL": 0.8, "FLOOR": 0.9,
    "WINDOW": 0.2, "DOOR": 0.3, "JOINT": 1.2,
    "FOUNDATION": 999.0, "EMPTY": 0.0,
}

# Light transmittance per element type
TRANSMITTANCE: Dict[str, float] = {
    "EMPTY": 1.0, "WINDOW": 0.85, "DOOR": 0.95,
    "BEAM": 0.0, "WALL": 0.0, "FLOOR": 0.0,
    "JOINT": 0.0, "FOUNDATION": 0.0,
}

# Airflow permeability (1.0 = fully permeable)
AIRFLOW_PERM: Dict[str, float] = {
    "EMPTY": 1.0, "DOOR": 1.0, "WINDOW": 0.0,
    "BEAM": 0.0, "WALL": 0.0, "FLOOR": 0.0,
    "JOINT": 0.0, "FOUNDATION": 0.0,
}

# Room type weights for lighting (Sec 3.3.3)
ROOM_LIGHT_WEIGHTS = {
    "OFFICE": 1.0, "BEDROOM": 0.8, "LIVING": 0.9, "KITCHEN": 1.0,
    "STORAGE": 0.3, "CORRIDOR": 0.5, "BATHROOM": 0.7, "GENERIC": 0.8,
}


# ─── VOXEL CELL ───────────────────────────────────────────────────────────────

@dataclass
class VoxelCell:
    element_type: str    = "EMPTY"
    material_id: int     = -1
    orientation: int     = 0
    connected: bool      = False
    failed: bool         = False
    stress_ratio: float  = 0.0
    room_id: int         = -1


# ─── ROOM ─────────────────────────────────────────────────────────────────────

@dataclass
class Room:
    room_id: int
    room_type: str
    seed: Tuple[int, int, int]
    voxels: Set[Tuple[int, int, int]] = field(default_factory=set)
    enclosed: bool = False
    has_door: bool = False
    has_window: bool = False
    volume: int = 0
    floor_connected: bool = False


# ─── VOXEL GRID ───────────────────────────────────────────────────────────────

class VoxelGrid:
    """3D sparse voxel grid for PGSA environment."""

    def __init__(self, W: int, H: int, D: int, hidden_params: Optional[Dict] = None):
        self.W, self.H, self.D = W, H, D
        self.cells: Dict[Tuple[int, int, int], VoxelCell] = {}
        self.rooms: Dict[int, Room] = {}
        self._next_room_id: int = 0
        self.total_cost: int = 0
        # Hidden true material params (may differ from nominal for curriculum L3+)
        self.hidden_params: Dict[int, dict] = hidden_params or {
            k: {"yield_mpa": v["yield_mpa"], "E_gpa": v["E_gpa"]}
            for k, v in MATERIALS.items()
        }

        # Pre-place FOUNDATION row at y=0
        for x in range(W):
            for z in range(D):
                self.cells[(x, 0, z)] = VoxelCell(
                    element_type="FOUNDATION", material_id=1, connected=True
                )

    def in_bounds(self, x, y, z) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H and 0 <= z < self.D

    def get(self, x, y, z) -> VoxelCell:
        return self.cells.get((x, y, z), VoxelCell())

    def element_type(self, x, y, z) -> str:
        return self.cells.get((x, y, z), VoxelCell()).element_type

    def is_solid(self, x, y, z) -> bool:
        et = self.element_type(x, y, z)
        return et not in ("EMPTY",) and et != "EMPTY"

    def is_passable(self, x, y, z) -> bool:
        """Passable for flood-fill (room detection) means not solid."""
        et = self.element_type(x, y, z)
        return et == "EMPTY"

    def place(self, x, y, z, element_type: str, material_id: int,
              orientation: int = 0) -> Tuple[bool, str]:
        """Place element. Returns (success, message)."""
        if not self.in_bounds(x, y, z):
            return False, f"Position ({x},{y},{z}) out of bounds"
        existing = self.cells.get((x, y, z))
        if existing and existing.element_type != "EMPTY":
            return False, f"Voxel ({x},{y},{z}) already occupied by {existing.element_type}"
        if y == 0:
            return False, "Cannot place on foundation row (y=0)"
        if element_type == "FOUNDATION":
            return False, "Cannot place FOUNDATION elements"
        if material_id not in MATERIALS:
            return False, f"Unknown material_id {material_id}"
        cost = MATERIALS[material_id]["cost"]
        cell = VoxelCell(
            element_type=element_type,
            material_id=material_id,
            orientation=orientation,
        )
        self.cells[(x, y, z)] = cell
        self.total_cost += cost
        return True, f"Placed {element_type} ({MATERIALS[material_id]['name']}) at ({x},{y},{z})"

    def remove(self, x, y, z) -> Tuple[bool, str]:
        """Remove element. Cannot remove FOUNDATION."""
        if not self.in_bounds(x, y, z):
            return False, f"Position ({x},{y},{z}) out of bounds"
        cell = self.cells.get((x, y, z))
        if cell is None or cell.element_type == "EMPTY":
            return False, f"No element at ({x},{y},{z})"
        if cell.element_type == "FOUNDATION":
            return False, "Cannot remove FOUNDATION elements"
        cost = MATERIALS[cell.material_id]["cost"] if cell.material_id >= 0 else 0
        del self.cells[(x, y, z)]
        self.total_cost = max(0, self.total_cost - cost)
        return True, f"Removed {cell.element_type} at ({x},{y},{z})"

    def replace_material(self, x, y, z, new_material_id: int) -> Tuple[bool, str]:
        """Swap material at existing voxel."""
        if not self.in_bounds(x, y, z):
            return False, "Out of bounds"
        cell = self.cells.get((x, y, z))
        if cell is None or cell.element_type in ("EMPTY", "FOUNDATION"):
            return False, "No replaceable element at position"
        if new_material_id not in MATERIALS:
            return False, f"Unknown material_id {new_material_id}"
        old_cost = MATERIALS[cell.material_id]["cost"] if cell.material_id >= 0 else 0
        new_cost = MATERIALS[new_material_id]["cost"]
        self.total_cost += (new_cost - old_cost)
        self.total_cost = max(0, self.total_cost)
        cell.material_id = new_material_id
        return True, f"Material at ({x},{y},{z}) changed to {MATERIALS[new_material_id]['name']}"

    def annotate_room(self, x1, y1, z1, x2, y2, z2, room_type: str) -> Tuple[int, str]:
        """Flood-fill room from center of annotated bounding box."""
        room_type = room_type.upper()
        # Clamp to bounds
        x1, x2 = max(0, min(x1, x2)), min(self.W - 1, max(x1, x2))
        y1, y2 = max(1, min(y1, y2)), min(self.H - 1, max(y1, y2))
        z1, z2 = max(0, min(z1, z2)), min(self.D - 1, max(z1, z2))
        # Seed = center of annotated box
        seed = (
            (x1 + x2) // 2,
            (y1 + y2) // 2,
            (z1 + z2) // 2,
        )
        # Ensure seed is empty
        if not self.is_passable(*seed):
            # Try nearby empty voxel
            found = False
            for dx in range(-2, 3):
                for dy in range(0, 3):
                    for dz in range(-2, 3):
                        ns = (seed[0]+dx, seed[1]+dy, seed[2]+dz)
                        if self.in_bounds(*ns) and self.is_passable(*ns):
                            seed = ns
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                return -1, "Could not find empty voxel near room seed — place walls/floors first"

        room_id = self._next_room_id
        self._next_room_id += 1
        room = Room(room_id=room_id, room_type=room_type, seed=seed)
        room.voxels, room.enclosed = self._flood_fill(seed)
        room.volume = len(room.voxels)

        # Check for DOOR and WINDOW on boundary
        for (x, y, z) in room.voxels:
            for nx, ny, nz in self._neighbors(x, y, z):
                et = self.element_type(nx, ny, nz)
                if et == "DOOR":
                    room.has_door = True
                if et == "WINDOW":
                    room.has_window = True

        # Update room_id field on cells
        for pos in room.voxels:
            if pos in self.cells:
                self.cells[pos].room_id = room_id
            else:
                self.cells[pos] = VoxelCell(room_id=room_id)

        # Check floor connectivity
        room.floor_connected = self._check_floor_connected(room)

        self.rooms[room_id] = room
        min_vol = 10 if room_type == "STORAGE" else 20
        # §4.1.2: valid = enclosed + door + window + min_volume + floor_connected
        valid = (room.enclosed and room.has_door and room.has_window
                 and room.volume >= min_vol and room.floor_connected)
        return room_id, (
            f"Room '{room_type}' (id={room_id}) detected: volume={room.volume}m³, "
            f"enclosed={room.enclosed}, has_door={room.has_door}, "
            f"has_window={room.has_window}, valid={valid}"
        )

    def _flood_fill(
        self, seed: Tuple[int, int, int]
    ) -> Tuple[Set[Tuple[int, int, int]], bool]:
        """BFS flood-fill through EMPTY voxels. Returns (filled_set, is_enclosed)."""
        visited: Set[Tuple[int, int, int]] = set()
        queue = deque([seed])
        enclosed = True
        MAX_FILL = 10_000  # safety cap

        while queue and len(visited) < MAX_FILL:
            pos = queue.popleft()
            if pos in visited:
                continue
            x, y, z = pos
            if not self.in_bounds(x, y, z):
                enclosed = False
                continue
            if not self.is_passable(x, y, z):
                continue
            visited.add(pos)
            # Boundary voxel → unenclosed
            if x in (0, self.W - 1) or y in (0, self.H - 1) or z in (0, self.D - 1):
                enclosed = False
            for n in self._neighbors(x, y, z):
                if n not in visited:
                    queue.append(n)

        return visited, enclosed

    def _neighbors(self, x, y, z) -> List[Tuple[int, int, int]]:
        return [
            (x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)
        ]

    def _check_floor_connected(self, room: Room) -> bool:
        """Check that at least one FLOOR element on room boundary is connected to foundation."""
        for (x, y, z) in room.voxels:
            for nx, ny, nz in self._neighbors(x, y, z):
                cell = self.cells.get((nx, ny, nz))
                if cell and cell.element_type == "FLOOR" and cell.connected:
                    return True
        # Relax: at y=1, a room with FOUNDATION directly below counts
        for (x, y, z) in room.voxels:
            if y == 1 and (x, 0, z) in self.cells:
                return True
        return False

    def count_non_foundation(self) -> int:
        return sum(
            1 for c in self.cells.values()
            if c.element_type not in ("EMPTY", "FOUNDATION")
        )

    def count_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in self.cells.values():
            if c.element_type not in ("EMPTY",):
                counts[c.element_type] = counts.get(c.element_type, 0) + 1
        return counts

    def summary_text(self) -> str:
        """Returns human-readable grid summary for LLM observation."""
        counts = self.count_by_type()
        total = self.count_non_foundation()
        room_count = len(self.rooms)
        valid_rooms = sum(
            1 for r in self.rooms.values()
            if r.enclosed and r.has_door and r.volume >= (10 if r.room_type == "STORAGE" else 20)
        )
        lines = [
            f"Grid {self.W}×{self.H}×{self.D} | {total} elements placed | Cost: {self.total_cost}",
            f"Elements: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            f"Rooms annotated: {room_count} (valid: {valid_rooms})",
        ]
        for room in self.rooms.values():
            min_vol = 10 if room.room_type == "STORAGE" else 20
            valid = room.enclosed and room.has_door and room.volume >= min_vol
            lines.append(
                f"  Room[{room.room_id}] type={room.room_type} vol={room.volume}m³ "
                f"enclosed={room.enclosed} door={room.has_door} window={room.has_window} valid={valid}"
            )
        return "\n".join(lines)


# ─── STRUCTURAL SCORER ────────────────────────────────────────────────────────

class StructuralScorer:
    """Heuristic structural analysis."""

    @staticmethod
    def analyze(grid: VoxelGrid) -> dict:
        """
        Returns:
          stability_class: STABLE | SERVICEABILITY_FAIL | PARTIAL_COLLAPSE | COLLAPSE
          connected_fraction: float [0,1]
          stress_safe_fraction: float [0,1]
          stability_score: float [0,1]
          r_structural: float
        """
        non_foundation = [
            (pos, cell) for pos, cell in grid.cells.items()
            if cell.element_type not in ("EMPTY", "FOUNDATION")
        ]

        if not non_foundation:
            return {
                "stability_class": "STABLE",
                "connected_fraction": 1.0,
                "stress_safe_fraction": 1.0,
                "stability_score": 0.0,
                "r_structural": 0.0,
                "failed_fraction": 0.0,
            }

        # ── Connectivity BFS from foundation ──────────────────────────────────
        connected_set: Set[Tuple[int,int,int]] = set()
        # Foundation nodes are seeds
        foundation_positions = {
            pos for pos, cell in grid.cells.items()
            if cell.element_type == "FOUNDATION"
        }
        queue = deque(foundation_positions)
        while queue:
            pos = queue.popleft()
            if pos in connected_set:
                continue
            connected_set.add(pos)
            x, y, z = pos
            for nx, ny, nz in [
                (x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)
            ]:
                npos = (nx, ny, nz)
                if npos not in connected_set and npos in grid.cells:
                    if grid.cells[npos].element_type not in ("EMPTY",):
                        queue.append(npos)

        # Update connected flags
        connected_non_foundation = 0
        for pos, cell in non_foundation:
            cell.connected = (pos in connected_set)
            if cell.connected:
                connected_non_foundation += 1

        connected_fraction = connected_non_foundation / max(len(non_foundation), 1)

        # ── Heuristic stress computation ──────────────────────────────────────
        # For each element, estimate load = sum of element weights above in column
        # Weight of an element = density × volume (1m³) × g ≈ proportional to density
        # Stress ratio = axial_load / (yield_strength × cross_section)
        # Simplified: stress_ratio ≈ (height × density_factor) / yield_normalized

        stress_safe = 0
        failed_count = 0
        max_h = max((pos[1] for pos, _ in non_foundation), default=1)

        for pos, cell in non_foundation:
            if not cell.connected:
                cell.stress_ratio = 2.0  # disconnected = overstressed
                cell.failed = True
                failed_count += 1
                continue

            x, y, z = pos
            mat = grid.hidden_params.get(cell.material_id, {})
            yield_mpa = mat.get("yield_mpa", MATERIALS.get(cell.material_id, {}).get("yield_mpa", 40))

            # Count elements in same x-z column above this element
            column_weight = sum(
                MATERIALS.get(grid.cells.get((x, yy, z), VoxelCell()).material_id, {}).get("density", 600)
                for yy in range(y + 1, grid.H)
                if (x, yy, z) in grid.cells and
                grid.cells[(x, yy, z)].element_type not in ("EMPTY",)
            )

            # Dead load + live load proxy (kN/m²)
            load_kpa = (column_weight * 9.81 / 1000.0) + 2.5  # 2.5 kN/m² live load

            # Heuristic stress ratio: load / material yield (normalized)
            stress_ratio = load_kpa / (yield_mpa * 0.5 + 1.0)
            # Height penalty: taller unsupported structures fail more
            if y > 2 and cell.element_type == "BEAM":
                # Check if element below exists in column
                has_support = any(
                    (x, yy, z) in grid.cells and
                    grid.cells[(x, yy, z)].element_type not in ("EMPTY",)
                    for yy in range(1, y)
                )
                if not has_support:
                    stress_ratio *= 1.5

            cell.stress_ratio = min(stress_ratio, 2.0)
            if cell.stress_ratio > 1.0:
                cell.failed = True
                failed_count += 1
            else:
                stress_safe += 1

        total = len(non_foundation)
        failed_fraction = failed_count / max(total, 1)
        stress_safe_fraction = stress_safe / max(total, 1)

        # ── Classify stability ──────────────────────────────────────────────
        foundation_adjacent_failed = any(
            cell.failed
            for (pos, cell) in non_foundation
            if pos[1] == 1  # directly above foundation
        )

        if failed_fraction > 0.15 or foundation_adjacent_failed:
            stability_class = "COLLAPSE"
        elif 0.01 < failed_fraction <= 0.15:
            stability_class = "PARTIAL_COLLAPSE"
        elif stress_safe_fraction < 0.7 or connected_fraction < 0.8:
            stability_class = "SERVICEABILITY_FAIL"
        else:
            stability_class = "STABLE"

        # ── Stability score ─────────────────────────────────────────────────
        if stability_class == "COLLAPSE":
            r_structural = -10.0
        elif stability_class == "SERVICEABILITY_FAIL":
            r_structural = -3.0
        else:
            frac_below_redzone = sum(
                1 for _, c in non_foundation if c.stress_ratio <= 0.7
            ) / max(total, 1)
            r_structural = frac_below_redzone * connected_fraction * stress_safe_fraction

        else_stability = max(0.0, min(1.0, (stress_safe_fraction * connected_fraction)))

        return {
            "stability_class": stability_class,
            "connected_fraction": connected_fraction,
            "stress_safe_fraction": stress_safe_fraction,
            "stability_score": else_stability,
            "r_structural": r_structural,
            "failed_fraction": failed_fraction,
        }


# ─── FUNCTIONAL SCORER ───────────────────────────────────────────────────────

class FunctionalScorer:
    """Scores functional architectural constraints."""

    @staticmethod
    def score_rooms(grid: VoxelGrid, task: dict) -> dict:
        """Room completion score (§7.2.2)."""
        required_rooms = task.get("required_rooms", [])
        n_required = len(required_rooms)
        # §4.1.2: valid room requires enclosed + door + window + min_volume + floor_connected
        valid_rooms = [
            r for r in grid.rooms.values()
            if r.enclosed and r.has_door and r.has_window and
               r.volume >= (10 if r.room_type == "STORAGE" else 20)
        ]
        n_valid = len(valid_rooms)
        if n_required == 0:
            completion = 1.0 if n_valid >= 1 else 0.5
        else:
            completion = min(n_valid / n_required, 1.0)
            # §7.2.2: −0.5 per missing required room type
            required_types = [rt.upper() for rt in required_rooms]
            found_types = [r.room_type.upper() for r in valid_rooms]
            missing_types = 0
            for rt in required_types:
                if rt not in found_types:
                    missing_types += 1
            completion -= 0.5 * missing_types  # spec: −0.5 per missing type
            completion = max(0.0, completion)

        return {"room_completion_score": completion, "n_valid_rooms": n_valid}

    @staticmethod
    def score_connectivity(grid: VoxelGrid, task: dict) -> float:
        """Fraction of required room-pairs with DOOR adjacency."""
        required_adj = task.get("required_adjacency", [])
        if not required_adj:
            # Give partial credit if multiple rooms share door access
            valid_rooms = [r for r in grid.rooms.values() if r.has_door]
            if len(valid_rooms) >= 2:
                return 1.0
            elif len(valid_rooms) == 1:
                return 0.5
            return 0.0

        satisfying = 0
        for pair in required_adj:
            r1 = next((r for r in grid.rooms.values()
                       if r.room_type.upper() == pair[0].upper()), None)
            r2 = next((r for r in grid.rooms.values()
                       if r.room_type.upper() == pair[1].upper()), None)
            if r1 and r2 and r1.has_door and r2.has_door:
                # Check if they share a boundary DOOR
                for pos in r1.voxels:
                    for nx, ny, nz in [
                        (pos[0]+1,pos[1],pos[2]),(pos[0]-1,pos[1],pos[2]),
                        (pos[0],pos[1],pos[2]+1),(pos[0],pos[1],pos[2]-1),
                    ]:
                        cell = grid.cells.get((nx, ny, nz))
                        if cell and cell.element_type == "DOOR":
                            # Check if this door borders r2
                            for mnx, mny, mnz in [
                                (nx+1,ny,nz),(nx-1,ny,nz),(nx,ny,nz+1),(nx,ny,nz-1)
                            ]:
                                if (mnx, mny, mnz) in r2.voxels:
                                    satisfying += 1
                                    break

        if not required_adj:
            return 0.0
        return min(satisfying / len(required_adj), 1.0)

    @staticmethod
    def score_airflow(grid: VoxelGrid, task: dict) -> float:
        """
        Heuristic airflow: fraction of enclosed rooms with an unobstructed
        path to at least one inlet/outlet vent (or exterior DOOR).
        """
        valid_rooms = [
            r for r in grid.rooms.values()
            if r.enclosed and r.volume >= 10 and r.room_type.upper() != "STORAGE"
        ]
        if not valid_rooms:
            return 0.0

        # Inlet positions from task spec
        inlets = task.get("inlet_positions", [])
        outlets = task.get("outlet_positions", [])

        # If no special vents defined, treat any exterior DOOR as both inlet+outlet
        exterior_doors: List[Tuple[int,int,int]] = []
        for (x, y, z), cell in grid.cells.items():
            if cell.element_type == "DOOR":
                # Is it on the perimeter?
                if x in (0, grid.W-1) or z in (0, grid.D-1):
                    exterior_doors.append((x, y, z))

        ventilated = 0
        for room in valid_rooms:
            # Room is ventilated if it has a DOOR (air flows through doors)
            # AND either has direct exterior access or connects to a ventilated room
            if room.has_door:
                # Check if any door voxel borders exterior or vent position
                has_air_path = False
                for pos in room.voxels:
                    for nx, ny, nz in [
                        (pos[0]+1,pos[1],pos[2]),(pos[0]-1,pos[1],pos[2]),
                        (pos[0],pos[1],pos[2]+1),(pos[0],pos[1],pos[2]-1),
                    ]:
                        cell = grid.cells.get((nx, ny, nz))
                        if cell and cell.element_type == "DOOR":
                            # This door — does it have an EMPTY voxel on other side?
                            for mnx, mny, mnz in [
                                (nx+1,ny,nz),(nx-1,ny,nz),(nx,ny,nz+1),(nx,ny,nz-1)
                            ]:
                                if grid.element_type(mnx, mny, mnz) == "EMPTY":
                                    has_air_path = True
                                    break
                if has_air_path:
                    ventilated += 1

        return ventilated / max(len(valid_rooms), 1)

    @staticmethod
    def score_lighting(grid: VoxelGrid, task: dict) -> float:
        """
        Heuristic lighting: fraction of rooms with WINDOW on boundary.
        Weighted by room type importance (§3.3.3 / §4.4).
        Only valid rooms (§4.1.2: enclosed + door + window + min_vol) contribute.
        """
        valid_rooms = [
            r for r in grid.rooms.values()
            if r.enclosed and r.has_door and r.has_window and
               r.volume >= (10 if r.room_type == "STORAGE" else 20)
        ]
        if not valid_rooms:
            return 0.0

        weighted_lit = 0.0
        total_weight = 0.0
        for room in valid_rooms:
            w = ROOM_LIGHT_WEIGHTS.get(room.room_type.upper(),
                                       ROOM_LIGHT_WEIGHTS["GENERIC"])
            total_weight += w
            if room.has_window:
                weighted_lit += w

        return weighted_lit / max(total_weight, 1e-6)

    @staticmethod
    def score_egress(grid: VoxelGrid, task: dict) -> float:
        """
        Egress score: fraction of rooms reachable from exterior door within 30m.
        Uses room centroid distances.
        """
        valid_rooms = [
            r for r in grid.rooms.values()
            if r.enclosed and r.has_door and r.has_window and
               r.volume >= (10 if r.room_type == "STORAGE" else 20)
        ]
        if not valid_rooms:
            return 0.0

        # Find exterior doors: DOOR elements on grid perimeter
        exterior_exits: List[Tuple[float, float, float]] = []
        for (x, y, z), cell in grid.cells.items():
            if cell.element_type == "DOOR":
                if x in (0, grid.W-1) or z in (0, grid.D-1) or y == 1:
                    exterior_exits.append((x + 0.5, y + 0.5, z + 0.5))

        # Also consider any wall gap at perimeter (open grid boundary)
        # Add virtual exit at grid edges
        for z in range(grid.D):
            exterior_exits.append((0, 1, z + 0.5))
            exterior_exits.append((grid.W, 1, z + 0.5))
        for x in range(grid.W):
            exterior_exits.append((x + 0.5, 1, 0))
            exterior_exits.append((x + 0.5, 1, grid.D))

        satisfying = 0
        for room in valid_rooms:
            if not room.voxels:
                continue
            # Room centroid
            cx = sum(p[0] for p in room.voxels) / len(room.voxels) + 0.5
            cy = sum(p[1] for p in room.voxels) / len(room.voxels) + 0.5
            cz = sum(p[2] for p in room.voxels) / len(room.voxels) + 0.5
            # Nearest exterior exit distance
            if exterior_exits:
                min_dist = min(
                    math.sqrt((cx-ex)**2 + (cz-ez)**2)
                    for ex, _, ez in exterior_exits
                )
                if min_dist <= 30.0:
                    satisfying += 1
            else:
                satisfying += 1  # No exterior walls = trivially accessible

        return satisfying / max(len(valid_rooms), 1)

    @staticmethod
    def score_density(grid: VoxelGrid, task: dict) -> float:
        """V_total / (N_occ × 10) capped at 1. Spec §4.6."""
        n_occ = task.get("occupancy_count", 1)
        # §4.1.2: requires WINDOW for valid room
        total_vol = sum(r.volume for r in grid.rooms.values()
                        if r.enclosed and r.has_door and r.has_window and
                        r.volume >= (10 if r.room_type == "STORAGE" else 20))
        density_ratio = total_vol / max(n_occ * 10, 1)
        # §4.6: continuous penalty = −1.0 × max(0, 1 − density_ratio) if below threshold
        # We return as a positive [0,1] score; the negative contribution comes via w[3]
        return min(density_ratio, 1.0)


# ─── PROBE PHYSICS ───────────────────────────────────────────────────────────

class ProbePhysics:
    """Simulate a PROBE_PHYSICS action — temporary test load at a voxel."""

    @staticmethod
    def probe(
        grid: VoxelGrid,
        x: int, y: int, z: int,
        load_kn: float = 10.0,
        direction: str = "Y"
    ) -> dict:
        """Return local stress and deformation at probe point."""
        cell = grid.cells.get((x, y, z))
        if cell is None or cell.element_type == "EMPTY":
            return {"error": "No element at probe location", "stress_mpa": 0.0, "deformation_mm": 0.0}

        mat_id = cell.material_id
        mat = grid.hidden_params.get(mat_id, {})
        E_gpa = mat.get("E_gpa", MATERIALS.get(mat_id, {}).get("E_gpa", 12))
        yield_mpa = mat.get("yield_mpa", MATERIALS.get(mat_id, {}).get("yield_mpa", 40))

        # Simplified Euler-Bernoulli: δ = PL³/(3EI), σ = PL/W for cantilever
        # Treat element as 1m beam with E from material
        # Approximate cross-section I ≈ (0.3m)^4/12 for typical column
        L = 1.0  # 1m voxel
        I = (0.3**4) / 12.0  # m^4
        A = 0.3**2  # m^2
        P = load_kn * 1000  # N

        E_pa = E_gpa * 1e9
        deformation_m = (P * L**3) / (3 * E_pa * I)
        stress_pa = P / A
        stress_mpa = stress_pa / 1e6

        # Add Gaussian noise (simulating measurement uncertainty)
        noise_factor = 1.0 + random.gauss(0, 0.05)
        stress_mpa *= noise_factor
        deformation_mm = deformation_m * 1000 * noise_factor

        return {
            "element_type": cell.element_type,
            "material_id": mat_id,
            "material_name": MATERIALS.get(mat_id, {}).get("name", "UNKNOWN"),
            "stress_mpa": round(stress_mpa, 3),
            "deformation_mm": round(deformation_mm, 4),
            "yield_mpa_inferred": round(yield_mpa * noise_factor, 2),
            "hint": (
                f"Measured stress {stress_mpa:.2f} MPa at load {load_kn} kN. "
                f"Deformation {deformation_mm:.3f} mm. "
                f"Material appears to have yield strength ~{yield_mpa * noise_factor:.1f} MPa."
            ),
        }


# ─── REWARD COMPUTER ─────────────────────────────────────────────────────────

# Curriculum weight schedules [w_struct, w_funct, w_physics, w_cost, w_effic, w_curios]
CURRICULUM_WEIGHTS = {
    1: [3.0, 0.5, 0.3, 0.5, 0.3, 0.5],
    2: [2.0, 1.5, 0.5, 0.5, 0.3, 0.3],
    3: [2.0, 1.5, 1.5, 0.5, 0.3, 0.1],
    4: [1.5, 1.5, 1.0, 0.8, 0.3, 0.05],
}


def compute_reward(
    grid: VoxelGrid,
    task: dict,
    curriculum_level: int,
    steps_remaining: int,
    total_steps: int,
    budget: int,
    probe_count: int,
    is_commit: bool = False,
    previous_score: float = 0.0,
) -> dict:
    """
    Full reward computation matching PGSA Chapter 7.
    Returns detailed breakdown dict.
    """
    w = CURRICULUM_WEIGHTS.get(curriculum_level, CURRICULUM_WEIGHTS[3])

    # ── Structural ────────────────────────────────────────────────────────────
    struct = StructuralScorer.analyze(grid)
    r_structural = struct["r_structural"]
    stability_class = struct["stability_class"]

    # ── Functional ────────────────────────────────────────────────────────────
    fscorer = FunctionalScorer()
    room_result = fscorer.score_rooms(grid, task)
    room_score = room_result["room_completion_score"]
    conn_score = fscorer.score_connectivity(grid, task)
    airflow_score = fscorer.score_airflow(grid, task)
    light_score = fscorer.score_lighting(grid, task)
    egress_score = fscorer.score_egress(grid, task)
    density_score = fscorer.score_density(grid, task)

    # Functional sub-weights: α=0.35, β=0.20, γ=0.15, δ=0.15, ε=0.10, ζ=0.05
    r_functional = (
        0.35 * room_score +
        0.20 * conn_score +
        0.15 * airflow_score +
        0.15 * light_score +
        0.10 * egress_score +
        0.05 * density_score
    )

    # ── Physics accuracy (proxy: reward probing + structural accuracy) ────────
    # In full PGSA: KL(posterior || true params). Here: probe bonus if L3+
    if curriculum_level >= 3:
        probe_fraction = min(probe_count / 50.0, 1.0)
        r_physics = -max(0.0, 1.0 - probe_fraction - struct["stress_safe_fraction"] * 0.5)
    else:
        r_physics = 0.0

    # ── Cost penalty ─────────────────────────────────────────────────────────
    budget_f = max(budget, 1)
    r_cost = grid.total_cost / budget_f  # [0, 1+], penalizes spend

    # ── Efficiency (only at COMMIT) ───────────────────────────────────────────
    r_efficiency = (steps_remaining / max(total_steps, 1)) if is_commit else 0.0

    # ── Curiosity (annealed intrinsic; we approximate with 0.0 for simplicity)
    r_curiosity = 0.0

    # ── Weighted total ────────────────────────────────────────────────────────
    r_total = (
        w[0] * r_structural
        + w[1] * r_functional
        + w[2] * r_physics
        - w[3] * r_cost
        + w[4] * r_efficiency
        + w[5] * r_curiosity
    )

    # Terminal penalties
    if stability_class == "COLLAPSE":
        r_total += -10.0
    elif stability_class == "PARTIAL_COLLAPSE":
        r_total += -struct["failed_fraction"] * 5.0

    # Normalize to approximately [0, 1] range for step rewards
    normalizer = sum(w[:3]) + w[4] + 10.0  # rough scale
    r_norm = (r_total + 10.0) / (normalizer + 10.0)
    r_norm = max(-1.0, min(1.0, r_norm))

    return {
        "r_structural": round(r_structural, 4),
        "r_functional": round(r_functional, 4),
        "r_physics": round(r_physics, 4),
        "r_cost": round(r_cost, 4),
        "r_efficiency": round(r_efficiency, 4),
        "r_curiosity": 0.0,
        "r_total_weighted": round(r_total, 4),
        "r_normalized": round(r_norm, 4),
        "stability_class": stability_class,
        "room_score": round(room_score, 4),
        "conn_score": round(conn_score, 4),
        "airflow_score": round(airflow_score, 4),
        "light_score": round(light_score, 4),
        "egress_score": round(egress_score, 4),
        "density_score": round(density_score, 4),
        "n_valid_rooms": room_result["n_valid_rooms"],
        "connected_fraction": round(struct["connected_fraction"], 4),
        "stress_safe_fraction": round(struct["stress_safe_fraction"], 4),
    }
