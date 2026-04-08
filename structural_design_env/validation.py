"""
Action geometry validator for StructuralDesignEnv.

Validates agent actions before they are applied to the graph/grid.
"""

from __future__ import annotations

from typing import Tuple

from structural_design_env.models import StructuralAction, TaskConfig
from structural_design_env.solver.sections import (
    COLUMN_SECTIONS,
    BEAM_SECTIONS,
    COLUMN_SECTION_ORDER,
    BEAM_SECTION_ORDER,
)

GRID_SIZE = 20


def validate_action(
    action: StructuralAction,
    graph,
    grid,
    task_config: TaskConfig,
) -> Tuple[bool, str]:
    """
    Validate a structural action.

    Returns
    -------
    (True, "") if valid
    (False, "ERROR_MESSAGE") if invalid
    """
    atype = action.action_type

    # ------------------------------------------------------------------
    # done
    # ------------------------------------------------------------------
    if atype == "done":
        return True, ""

    # ------------------------------------------------------------------
    # place_column
    # ------------------------------------------------------------------
    if atype == "place_column":
        gx = action.grid_x
        gy = action.grid_y
        floor = action.floor
        section = action.section

        if gx is None or gy is None:
            return False, "place_column requires grid_x and grid_y"
        if floor is None:
            return False, "place_column requires floor"
        if not (0 <= gx < GRID_SIZE):
            return False, f"grid_x={gx} out of range [0, {GRID_SIZE - 1}]"
        if not (0 <= gy < GRID_SIZE):
            return False, f"grid_y={gy} out of range [0, {GRID_SIZE - 1}]"
        if not (0 <= floor < task_config.n_floors):
            return False, f"floor={floor} out of range [0, {task_config.n_floors - 1}]"
        if section is None:
            return False, "place_column requires section"
        if section not in COLUMN_SECTIONS:
            return False, f"Invalid column section '{section}'. Choose from {COLUMN_SECTION_ORDER}"
        # Check site bounds (1 grid cell = 1m)
        if gx >= task_config.site_width_m:
            return False, f"grid_x={gx} exceeds site_width={task_config.site_width_m}m"
        if gy >= task_config.site_depth_m:
            return False, f"grid_y={gy} exceeds site_depth={task_config.site_depth_m}m"
        if graph.has_column(gx, gy, floor):
            return False, f"Column already exists at ({gx},{gy}) floor {floor}"
        return True, ""

    # ------------------------------------------------------------------
    # place_beam
    # ------------------------------------------------------------------
    if atype == "place_beam":
        fx = action.from_node_x
        fy = action.from_node_y
        tx = action.to_node_x
        ty = action.to_node_y
        floor = action.floor
        section = action.section
        orientation = action.orientation

        if any(v is None for v in [fx, fy, tx, ty]):
            return False, "place_beam requires from_node_x/y and to_node_x/y"
        if floor is None:
            return False, "place_beam requires floor"
        if section is None:
            return False, "place_beam requires section"
        if section not in BEAM_SECTIONS:
            return False, f"Invalid beam section '{section}'. Choose from {BEAM_SECTION_ORDER}"
        if orientation not in ("x", "y"):
            return False, "place_beam requires orientation 'x' or 'y'"

        # Axis-aligned check
        if fx != tx and fy != ty:
            return False, f"Beam must be axis-aligned: from ({fx},{fy}) to ({tx},{ty}) is diagonal"
        if fx == tx and fy == ty:
            return False, "Beam start and end are the same node"

        if not (0 <= floor < task_config.n_floors):
            return False, f"floor={floor} out of range [0, {task_config.n_floors - 1}]"

        # Check columns exist at both ends for this floor
        if not graph.has_column(fx, fy, floor):
            return False, f"No column at ({fx},{fy}) floor {floor} — beam endpoint requires a column"
        if not graph.has_column(tx, ty, floor):
            return False, f"No column at ({tx},{ty}) floor {floor} — beam endpoint requires a column"

        # Check beam doesn't already exist
        if graph.has_beam(fx, fy, tx, ty, floor):
            return False, f"Beam already exists between ({fx},{fy}) and ({tx},{ty}) at floor {floor}"

        # Bounds check
        for x, y in [(fx, fy), (tx, ty)]:
            if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
                return False, f"Node ({x},{y}) out of grid range"

        return True, ""

    # ------------------------------------------------------------------
    # upgrade_section
    # ------------------------------------------------------------------
    if atype == "upgrade_section":
        eid = action.element_id
        if eid is None:
            return False, "upgrade_section requires element_id"
        elem = graph.elements.get(eid)
        if elem is None:
            return False, f"Element '{eid}' does not exist"
        if elem.element_type == "wall":
            return False, "Cannot upgrade section of a wall"

        current = elem.section
        if current in COLUMN_SECTION_ORDER:
            if current == COLUMN_SECTION_ORDER[-1]:
                return False, f"Section '{current}' is already at maximum"
        elif current in BEAM_SECTION_ORDER:
            if current == BEAM_SECTION_ORDER[-1]:
                return False, f"Section '{current}' is already at maximum"
        else:
            return False, f"Unknown section '{current}'"
        return True, ""

    # ------------------------------------------------------------------
    # downgrade_section
    # ------------------------------------------------------------------
    if atype == "downgrade_section":
        eid = action.element_id
        if eid is None:
            return False, "downgrade_section requires element_id"
        elem = graph.elements.get(eid)
        if elem is None:
            return False, f"Element '{eid}' does not exist"
        if elem.element_type == "wall":
            return False, "Cannot downgrade section of a wall"

        current = elem.section
        if current in COLUMN_SECTION_ORDER:
            if current == COLUMN_SECTION_ORDER[0]:
                return False, f"Section '{current}' is already at minimum"
        elif current in BEAM_SECTION_ORDER:
            if current == BEAM_SECTION_ORDER[0]:
                return False, f"Section '{current}' is already at minimum"
        else:
            return False, f"Unknown section '{current}'"
        return True, ""

    # ------------------------------------------------------------------
    # remove_element
    # ------------------------------------------------------------------
    if atype == "remove_element":
        eid = action.element_id
        if eid is None:
            return False, "remove_element requires element_id"
        if eid not in graph.elements:
            return False, f"Element '{eid}' does not exist"
        return True, ""

    # ------------------------------------------------------------------
    # add_wall
    # ------------------------------------------------------------------
    if atype == "add_wall":
        fx = action.from_node_x
        fy = action.from_node_y
        tx = action.to_node_x
        ty = action.to_node_y
        floor = action.floor
        thickness = action.thickness_m

        if any(v is None for v in [fx, fy, tx, ty]):
            return False, "add_wall requires from_node_x/y and to_node_x/y"
        if floor is None:
            return False, "add_wall requires floor"

        # Axis-aligned
        if fx != tx and fy != ty:
            return False, f"Wall must be axis-aligned: from ({fx},{fy}) to ({tx},{ty}) is diagonal"
        if fx == tx and fy == ty:
            return False, "Wall start and end are the same node"

        if not (0 <= floor < task_config.n_floors):
            return False, f"floor={floor} out of range"

        # Columns at both ends
        if not graph.has_column(fx, fy, floor):
            return False, f"No column at ({fx},{fy}) floor {floor}"
        if not graph.has_column(tx, ty, floor):
            return False, f"No column at ({tx},{ty}) floor {floor}"

        # Thickness
        if thickness is not None and thickness not in (0.2, 0.3):
            return False, f"thickness_m must be 0.2 or 0.3, got {thickness}"

        # Already exists?
        if graph.has_wall(fx, fy, tx, ty, floor):
            return False, f"Wall already exists between ({fx},{fy}) and ({tx},{ty}) at floor {floor}"

        return True, ""

    return False, f"Unknown action_type: '{atype}'"
