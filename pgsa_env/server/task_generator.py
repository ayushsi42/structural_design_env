"""
PGSA Task Generator
--------------------
Samples episode task specifications for the three difficulty tiers:
  - Easy  (curriculum level 1): stable_shelter
  - Medium (curriculum level 2): functional_office
  - Hard   (curriculum level 3): material_detective
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── TASK SPEC DATACLASS ──────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    task_id: str
    difficulty: str            # "easy" | "medium" | "hard"
    curriculum_level: int      # 1 | 2 | 3
    grid_dims: Tuple[int,int,int]  # (W, H, D)
    budget: int
    action_budget: int
    probe_budget: int
    required_rooms: List[str]  # room type strings
    required_adjacency: List[Tuple[str,str]]
    occupancy_count: int
    inlet_positions: List[Tuple[int,int,int]]
    outlet_positions: List[Tuple[int,int,int]]
    sun_azimuth: float
    sun_elevation: float
    hidden_material_props: bool   # True if yield/E are hidden
    advancement_threshold: float
    description: str           # Human-readable task description for LLM
    hint: str                  # Useful beginner hint

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "curriculum_level": self.curriculum_level,
            "grid_dims": list(self.grid_dims),
            "budget": self.budget,
            "action_budget": self.action_budget,
            "probe_budget": self.probe_budget,
            "required_rooms": self.required_rooms,
            "required_adjacency": [list(p) for p in self.required_adjacency],
            "occupancy_count": self.occupancy_count,
            "inlet_positions": [list(p) for p in self.inlet_positions],
            "outlet_positions": [list(p) for p in self.outlet_positions],
            "sun_azimuth": self.sun_azimuth,
            "sun_elevation": self.sun_elevation,
            "hidden_material_props": self.hidden_material_props,
            "advancement_threshold": self.advancement_threshold,
        }


# ─── HIDDEN PARAMETER SAMPLER ─────────────────────────────────────────────────

from pgsa_env.server.physics_sim import MATERIALS  # noqa: E402


def _sample_hidden_params(
    curriculum_level: int,
    rng: random.Random
) -> Dict[int, dict]:
    """
    Sample true material properties for an episode.
    L1/L2: nominal values (no hidden-ness).
    L3+:   log-normal draw with σ_scale = 0.30.
    """
    sigma_scale_map = {1: 0.0, 2: 0.0, 3: 0.30, 4: 0.40}
    sigma = sigma_scale_map.get(curriculum_level, 0.0)

    params: Dict[int, dict] = {}
    for mat_id, mat in MATERIALS.items():
        if sigma == 0.0:
            params[mat_id] = {
                "yield_mpa": mat["yield_mpa"],
                "E_gpa": mat["E_gpa"],
            }
        else:
            params[mat_id] = {
                "yield_mpa": rng.lognormvariate(
                    math.log(mat["yield_mpa"]), sigma
                ),
                "E_gpa": rng.lognormvariate(
                    math.log(mat["E_gpa"]), sigma
                ),
            }
    return params


import math  # noqa: E402 (placed after imports to mirror original ordering)


# ─── TASK GENERATORS ─────────────────────────────────────────────────────────

def _vent_positions(
    W: int, D: int,
    rng: random.Random,
    n_inlets: int = 1,
    n_outlets: int = 1
) -> Tuple[List[Tuple[int,int,int]], List[Tuple[int,int,int]]]:
    """Pick random vent positions on the perimeter walls."""
    inlets, outlets = [], []
    for _ in range(n_inlets):
        side = rng.choice(["west", "north"])
        y = rng.randint(1, 3)
        if side == "west":
            inlets.append((0, y, rng.randint(1, D - 2)))
        else:
            inlets.append((rng.randint(1, W - 2), y, 0))
    for _ in range(n_outlets):
        side = rng.choice(["east", "south"])
        y = rng.randint(1, 3)
        if side == "east":
            outlets.append((W - 1, y, rng.randint(1, D - 2)))
        else:
            outlets.append((rng.randint(1, W - 2), y, D - 1))
    return inlets, outlets


def generate_easy_task(seed: Optional[int] = None) -> TaskSpec:
    """
    Level 1 — Stable Shelter
    8×8×8 grid. Build a stable freestanding structure with at least one enclosed room.
    All material properties fully visible. Simple load cases only.
    """
    rng = random.Random(seed)
    W, H, D = 8, 8, 8
    inlets, outlets = _vent_positions(W, D, rng, 1, 1)
    sun_az = rng.uniform(0, 2 * math.pi)
    sun_el = rng.uniform(math.pi / 6, math.pi / 2)

    description = (
        "=== TASK: Stable Shelter (EASY) ===\n\n"
        "You are an architect designing a small structural shelter.\n"
        f"Grid: {W}×{H}×{D} voxels (each voxel = 1m³). Foundation pre-placed at y=0.\n\n"
        "REQUIREMENTS:\n"
        "  1. Build a structurally stable freestanding structure (no collapse).\n"
        "  2. Enclose at least one room using walls, floors, and a door.\n"
        "  3. Annotate the room with ANNOTATE_ROOM.\n"
        "  4. Stay within the material cost budget of 200 units.\n\n"
        "AVAILABLE ELEMENTS: BEAM, WALL, FLOOR, DOOR, WINDOW, JOINT\n"
        "MATERIALS: 0=WOOD(cost=1), 1=CONCRETE(cost=3), 2=STEEL(cost=8), "
        "3=GLASS(cost=5), 4=COMPOSITE(cost=20)\n\n"
        "SCORING: 50% structural stability + 50% room completion\n\n"
        "GRADING SCALE: 0.0 (collapse, no rooms) → 1.0 (stable, enclosed room with door)\n\n"
        "Take ONE action per step as a JSON object in your response. Examples:\n"
        '  {"action_type": "PLACE_ELEMENT", "x": 2, "y": 1, "z": 2, '
        '"element_type": "WALL", "material_id": 0, "orientation": 0}\n'
        '  {"action_type": "PLACE_ELEMENT", "x": 2, "y": 1, "z": 6, '
        '"element_type": "WALL", "material_id": 0, "orientation": 0}\n'
        '  {"action_type": "ANNOTATE_ROOM", "x1": 2, "y1": 1, "z1": 2, '
        '"x2": 5, "y2": 3, "z2": 6, "room_type": "STORAGE"}\n'
        '  {"action_type": "COMMIT_DESIGN"}\n\n'
        "TIP: Build walls around a rectangular area, add a FLOOR on top, "
        "include at least one DOOR and one WINDOW, then annotate the room."
    )

    return TaskSpec(
        task_id="stable_shelter",
        difficulty="easy",
        curriculum_level=1,
        grid_dims=(W, H, D),
        budget=200,
        action_budget=500,
        probe_budget=0,  # No probing at level 1
        required_rooms=[],  # Just need ≥1 valid room
        required_adjacency=[],
        occupancy_count=5,
        inlet_positions=inlets,
        outlet_positions=outlets,
        sun_azimuth=sun_az,
        sun_elevation=sun_el,
        hidden_material_props=False,
        advancement_threshold=0.60,
        description=description,
        hint="Start by building 4 walls at y=1 in a square, add a FLOOR on top, "
             "include a DOOR on one wall and a WINDOW on another, then ANNOTATE_ROOM.",
    )


def generate_medium_task(seed: Optional[int] = None) -> TaskSpec:
    """
    Level 2 — Functional Office
    16×12×16 grid. Build 2–3 rooms with door connectivity, airflow, and lighting.
    """
    rng = random.Random(seed)
    W, H, D = 16, 12, 16

    room_configs = [
        (["OFFICE", "CORRIDOR"], [("OFFICE", "CORRIDOR")]),
        (["BEDROOM", "LIVING"], [("BEDROOM", "LIVING")]),
        (["OFFICE", "KITCHEN", "CORRIDOR"], [("OFFICE", "CORRIDOR"), ("KITCHEN", "CORRIDOR")]),
    ]
    rooms, adj = rng.choice(room_configs)
    inlets, outlets = _vent_positions(W, D, rng, 2, 1)
    sun_az = rng.uniform(0, 2 * math.pi)
    sun_el = rng.uniform(math.pi / 6, math.pi / 2)
    n_occ = rng.randint(8, 20)

    room_str = ", ".join(rooms)
    adj_str = " and ".join(f"{a}↔{b}" for a, b in adj)

    description = (
        "=== TASK: Functional Office (MEDIUM) ===\n\n"
        "You are designing a multi-room office building.\n"
        f"Grid: {W}×{H}×{D} voxels (each voxel = 1m³). Foundation pre-placed at y=0.\n\n"
        "REQUIREMENTS:\n"
        f"  1. Build {len(rooms)} enclosed rooms: {room_str}\n"
        f"  2. Required door adjacency: {adj_str}\n"
        "  3. Each room needs: walls enclosing it, at least 1 DOOR, at least 1 WINDOW.\n"
        f"  4. Airflow: rooms should have DOOR connections to exterior or other ventilated rooms.\n"
        "  5. Lighting: WINDOW elements on room walls.\n"
        f"  6. Budget: 1000 cost units. Occupancy: {n_occ} people.\n\n"
        "MATERIALS: 0=WOOD(cost=1), 1=CONCRETE(cost=3), 2=STEEL(cost=8), "
        "3=GLASS(cost=5), 4=COMPOSITE(cost=20)\n"
        "ELEMENTS: BEAM, WALL, FLOOR, DOOR, WINDOW, JOINT\n\n"
        "SCORING: Room completion(35%) + Connectivity(20%) + Airflow(15%) + "
        "Lighting(15%) + Egress(10%) + Density(5%)\n\n"
        "GRADING SCALE: 0.0 → 1.0 based on all constraints satisfied\n\n"
        "Take ONE action per step as a JSON object. Finish with COMMIT_DESIGN.\n\n"
        "STRATEGY: Build each room as a box (4 walls + floor + ceiling), add DOORs "
        "between rooms and to exterior, add WINDOW on outer walls. "
        "Use ANNOTATE_ROOM after each room is enclosed.\n\n"
        f"Inlet vents at: {inlets}\n"
        f"Outlet vents at: {outlets}"
    )

    return TaskSpec(
        task_id="functional_office",
        difficulty="medium",
        curriculum_level=2,
        grid_dims=(W, H, D),
        budget=1000,
        action_budget=500,
        probe_budget=0,
        required_rooms=rooms,
        required_adjacency=adj,
        occupancy_count=n_occ,
        inlet_positions=inlets,
        outlet_positions=outlets,
        sun_azimuth=sun_az,
        sun_elevation=sun_el,
        hidden_material_props=False,
        advancement_threshold=0.65,
        description=description,
        hint="Build each room one at a time. Use WALL for exterior, WALL for interior dividers. "
             "DOOR between rooms, WINDOW on outer walls. ANNOTATE_ROOM for each enclosed space.",
    )


def generate_hard_task(seed: Optional[int] = None) -> TaskSpec:
    """
    Level 3 — Material Detective
    16×16×16 grid. 4–5 rooms with hidden material properties.
    Must use PROBE_PHYSICS to infer material strengths.
    """
    rng = random.Random(seed)
    W, H, D = 16, 16, 16

    room_configs = [
        (["OFFICE", "BEDROOM", "KITCHEN", "CORRIDOR"],
         [("OFFICE", "CORRIDOR"), ("BEDROOM", "CORRIDOR"), ("KITCHEN", "CORRIDOR")]),
        (["LIVING", "BEDROOM", "KITCHEN", "BATHROOM", "CORRIDOR"],
         [("LIVING", "CORRIDOR"), ("BEDROOM", "CORRIDOR"), ("KITCHEN", "LIVING")]),
    ]
    rooms, adj = rng.choice(room_configs)
    inlets, outlets = _vent_positions(W, D, rng, 2, 2)
    sun_az = rng.uniform(0, 2 * math.pi)
    sun_el = rng.uniform(math.pi / 6, math.pi / 2)
    n_occ = rng.randint(15, 30)

    room_str = ", ".join(rooms)
    adj_str = "; ".join(f"{a}↔{b}" for a, b in adj)

    description = (
        "=== TASK: Material Detective (HARD) ===\n\n"
        "You are designing a multi-room complex under physical uncertainty.\n"
        f"Grid: {W}×{H}×{D} voxels. Foundation pre-placed at y=0.\n\n"
        "⚠ WARNING: Material yield strengths and elastic moduli are HIDDEN this episode.\n"
        "True values are drawn from log-normal distributions (±30% variation).\n"
        "Use PROBE_PHYSICS to test materials before committing to heavy structures!\n\n"
        "REQUIREMENTS:\n"
        f"  1. Build {len(rooms)} enclosed rooms: {room_str}\n"
        f"  2. Door adjacency: {adj_str}\n"
        "  3. Full lighting (WINDOW), airflow (DOOR to exterior), egress.\n"
        f"  4. Budget: 2000 cost units. Occupancy: {n_occ} people.\n"
        "  5. Probe budget: 50 PROBE_PHYSICS actions.\n\n"
        "PROBE_PHYSICS usage (probing reduces physics uncertainty → better reward):\n"
        '  {"action_type": "PROBE_PHYSICS", "x": 5, "y": 1, "z": 5, '
        '"load_kn": 10.0, "direction": "Y"}\n\n'
        "SCORING: Full reward including Physics Accuracy term (penalty if wrong material chosen).\n"
        "Physics accuracy rewards using probes AND choosing materials that survive hidden loads.\n\n"
        "GRADING SCALE: 0.0 → 1.0 (full functional + physics accuracy)\n\n"
        "Take ONE action per step as JSON. Finish with COMMIT_DESIGN.\n\n"
        "STRATEGY:\n"
        "  1. Probe foundation-level elements to gauge material stiffness.\n"
        "  2. Choose high-strength materials (STEEL/COMPOSITE) for taller columns.\n"
        "  3. Build rooms methodically — one room at a time.\n"
        "  4. Use ANNOTATE_ROOM after each room wall is complete.\n\n"
        f"Inlet vents: {inlets} | Outlet vents: {outlets}"
    )

    return TaskSpec(
        task_id="material_detective",
        difficulty="hard",
        curriculum_level=3,
        grid_dims=(W, H, D),
        budget=2000,
        action_budget=500,
        probe_budget=50,
        required_rooms=rooms,
        required_adjacency=adj,
        occupancy_count=n_occ,
        inlet_positions=inlets,
        outlet_positions=outlets,
        sun_azimuth=sun_az,
        sun_elevation=sun_el,
        hidden_material_props=True,
        advancement_threshold=0.60,
        description=description,
        hint="Use PROBE_PHYSICS early on freshly placed elements. Compare deformation_mm "
             "across materials to identify which has higher E (lower deflection = higher E).",
    )


# ─── PUBLIC INTERFACE ─────────────────────────────────────────────────────────

TASK_GENERATORS = {
    "easy":   generate_easy_task,
    "medium": generate_medium_task,
    "hard":   generate_hard_task,
    "stable_shelter":      generate_easy_task,
    "functional_office":   generate_medium_task,
    "material_detective":  generate_hard_task,
}


def generate_task(difficulty: str = "easy", seed: Optional[int] = None) -> TaskSpec:
    """Generate a task specification for the given difficulty."""
    generator = TASK_GENERATORS.get(difficulty, generate_easy_task)
    return generator(seed=seed)


def sample_hidden_params(task: TaskSpec, seed: Optional[int] = None) -> Dict[int, dict]:
    """Sample hidden material parameters for an episode."""
    rng = random.Random(seed)
    return _sample_hidden_params(task.curriculum_level, rng)
