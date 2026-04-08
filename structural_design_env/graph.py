"""
StructuralGraph: maintains nodes and elements for the building frame.

Nodes: n_{grid_x}_{grid_y}_{floor}
Elements:
  - col_{x}_{y}_{floor}         for columns (floor → floor+1)
  - beam_{x1}_{y1}_{x2}_{y2}_{floor} for beams
  - wall_{x1}_{y1}_{x2}_{y2}_{floor} for shear walls
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class NodeData:
    node_id: str
    x_m: float   # East (grid_x in metres)
    y_m: float   # North (grid_y in metres)
    z_m: float   # Up (floor * floor_height_m)
    floor: int
    is_fixed_base: bool = False


@dataclass
class ElementData:
    element_id: str
    element_type: str       # "column" | "beam" | "wall"
    section: str
    node_i: str             # bottom/left node
    node_j: str             # top/right node
    length_m: float
    orientation: Optional[str] = None   # "x" | "y" for beams/walls
    thickness_m: Optional[float] = None  # walls only


class StructuralGraph:
    """Maintains the structural model as a node-element graph."""

    def __init__(self, floor_height_m: float = 3.5):
        self.floor_height_m = floor_height_m
        self.nodes: Dict[str, NodeData] = {}
        self.elements: Dict[str, ElementData] = {}

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    @staticmethod
    def node_id(grid_x: int, grid_y: int, floor: int) -> str:
        return f"n_{grid_x}_{grid_y}_{floor}"

    def add_node(self, grid_x: int, grid_y: int, floor: int) -> str:
        nid = self.node_id(grid_x, grid_y, floor)
        if nid not in self.nodes:
            # Physical coordinates: 1 grid cell = 1 metre
            self.nodes[nid] = NodeData(
                node_id=nid,
                x_m=float(grid_x),
                y_m=float(grid_y),
                z_m=float(floor) * self.floor_height_m,
                floor=floor,
                is_fixed_base=(floor == 0),
            )
        return nid

    def get_node(self, grid_x: int, grid_y: int, floor: int) -> Optional[NodeData]:
        return self.nodes.get(self.node_id(grid_x, grid_y, floor))

    # ------------------------------------------------------------------
    # Column
    # ------------------------------------------------------------------

    @staticmethod
    def column_id(grid_x: int, grid_y: int, floor: int) -> str:
        return f"col_{grid_x}_{grid_y}_{floor}"

    def place_column(self, grid_x: int, grid_y: int, floor: int, section: str) -> str:
        """Place a column spanning from floor to floor+1 at (grid_x, grid_y)."""
        # Ensure both end nodes exist
        nid_bottom = self.add_node(grid_x, grid_y, floor)
        nid_top = self.add_node(grid_x, grid_y, floor + 1)

        eid = self.column_id(grid_x, grid_y, floor)
        self.elements[eid] = ElementData(
            element_id=eid,
            element_type="column",
            section=section,
            node_i=nid_bottom,
            node_j=nid_top,
            length_m=self.floor_height_m,
        )
        return eid

    def has_column(self, grid_x: int, grid_y: int, floor: int) -> bool:
        return self.column_id(grid_x, grid_y, floor) in self.elements

    # ------------------------------------------------------------------
    # Beam
    # ------------------------------------------------------------------

    @staticmethod
    def beam_id(x1: int, y1: int, x2: int, y2: int, floor: int) -> str:
        # Canonical ordering: smaller coords first
        if (x1, y1) > (x2, y2):
            x1, y1, x2, y2 = x2, y2, x1, y1
        return f"beam_{x1}_{y1}_{x2}_{y2}_{floor}"

    def place_beam(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        floor: int,
        section: str,
        orientation: str,
    ) -> str:
        """Place a beam connecting two column nodes at the same floor."""
        nid_i = self.node_id(from_x, from_y, floor + 1)
        nid_j = self.node_id(to_x, to_y, floor + 1)

        # Beam nodes sit at floor+1 level (top of the supporting columns)
        # Ensure nodes exist (they should if columns were placed, but be safe)
        if nid_i not in self.nodes:
            self.add_node(from_x, from_y, floor + 1)
        if nid_j not in self.nodes:
            self.add_node(to_x, to_y, floor + 1)

        dx = abs(to_x - from_x)
        dy = abs(to_y - from_y)
        length_m = float(dx if dx > 0 else dy)

        eid = self.beam_id(from_x, from_y, to_x, to_y, floor)
        self.elements[eid] = ElementData(
            element_id=eid,
            element_type="beam",
            section=section,
            node_i=nid_i,
            node_j=nid_j,
            length_m=length_m,
            orientation=orientation,
        )
        return eid

    def has_beam(self, from_x: int, from_y: int, to_x: int, to_y: int, floor: int) -> bool:
        return self.beam_id(from_x, from_y, to_x, to_y, floor) in self.elements

    # ------------------------------------------------------------------
    # Wall
    # ------------------------------------------------------------------

    @staticmethod
    def wall_id(x1: int, y1: int, x2: int, y2: int, floor: int) -> str:
        if (x1, y1) > (x2, y2):
            x1, y1, x2, y2 = x2, y2, x1, y1
        return f"wall_{x1}_{y1}_{x2}_{y2}_{floor}"

    def add_wall(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        floor: int,
        thickness_m: float,
        orientation: str,
    ) -> str:
        """Add a shear wall between two column nodes."""
        nid_i = self.node_id(from_x, from_y, floor + 1)
        nid_j = self.node_id(to_x, to_y, floor + 1)
        if nid_i not in self.nodes:
            self.add_node(from_x, from_y, floor + 1)
        if nid_j not in self.nodes:
            self.add_node(to_x, to_y, floor + 1)

        dx = abs(to_x - from_x)
        dy = abs(to_y - from_y)
        length_m = float(dx if dx > 0 else dy)

        eid = self.wall_id(from_x, from_y, to_x, to_y, floor)
        # Treat wall as a very stiff beam-like element
        self.elements[eid] = ElementData(
            element_id=eid,
            element_type="wall",
            section="wall",
            node_i=nid_i,
            node_j=nid_j,
            length_m=length_m,
            orientation=orientation,
            thickness_m=thickness_m,
        )
        return eid

    def has_wall(self, from_x: int, from_y: int, to_x: int, to_y: int, floor: int) -> bool:
        return self.wall_id(from_x, from_y, to_x, to_y, floor) in self.elements

    # ------------------------------------------------------------------
    # Remove element
    # ------------------------------------------------------------------

    def remove_element(self, element_id: str) -> bool:
        """Remove an element. Returns True if it existed."""
        if element_id in self.elements:
            del self.elements[element_id]
            # Prune orphan non-base nodes (no elements reference them)
            self._prune_orphan_nodes()
            return True
        return False

    def _prune_orphan_nodes(self):
        """Remove nodes not referenced by any element and not a fixed base."""
        referenced = set()
        for elem in self.elements.values():
            referenced.add(elem.node_i)
            referenced.add(elem.node_j)
        to_delete = [
            nid
            for nid, nd in self.nodes.items()
            if nid not in referenced and not nd.is_fixed_base
        ]
        for nid in to_delete:
            del self.nodes[nid]

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_columns_at_floor(self, floor: int) -> list:
        return [
            e
            for e in self.elements.values()
            if e.element_type == "column" and e.node_i.endswith(f"_{floor}")
        ]

    def get_beams_at_floor(self, floor: int) -> list:
        return [
            e
            for e in self.elements.values()
            if e.element_type == "beam" and e.node_i.endswith(f"_{floor + 1}")
        ]

    def node_index_map(self) -> dict:
        """Return mapping node_id -> integer index (for stiffness assembly)."""
        return {nid: i for i, nid in enumerate(sorted(self.nodes.keys()))}

    def total_steel_mass_kg(self) -> float:
        """Compute total steel mass for all elements."""
        from structural_design_env.solver.sections import (
            COLUMN_SECTIONS,
            BEAM_SECTIONS,
        )

        mass = 0.0
        for elem in self.elements.values():
            if elem.element_type == "wall":
                # Concrete wall: ~2400 kg/m³ (approximate)
                t = elem.thickness_m or 0.2
                mass += 2400.0 * t * 3.0 * elem.length_m  # height~3m approx
                continue
            props = (
                COLUMN_SECTIONS.get(elem.section)
                or BEAM_SECTIONS.get(elem.section)
            )
            if props:
                mass += props["mass_kg_m"] * elem.length_m
        return mass

    def copy(self) -> "StructuralGraph":
        """Return a deep copy for redundancy checks."""
        import copy

        return copy.deepcopy(self)
