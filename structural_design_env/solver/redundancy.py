"""
Progressive collapse / redundancy check per EN 1991-1-7 (GSA guidelines).

For each column, temporarily remove it from the graph and re-solve with
accidental load combination (1.0*DL + 0.5*LL). The structure is considered
redundant if it survives all single column removals with max UR <= 2.0.
"""

from __future__ import annotations

from structural_design_env.solver.stiffness_matrix import StructuralSolver
from structural_design_env.solver.eurocode3 import check_member
from structural_design_env.solver.sections import (
    COLUMN_SECTIONS,
    BEAM_SECTIONS,
    get_section_props,
)


def check_column_removal_redundancy(graph, task_config) -> bool:
    """
    Check if the structure is redundant under single column removal.

    Parameters
    ----------
    graph : StructuralGraph (original, unmodified)
    task_config : TaskConfig

    Returns
    -------
    True if structure survives all individual column removals.
    """
    solver = StructuralSolver()

    # Identify all columns
    column_ids = [
        eid
        for eid, elem in graph.elements.items()
        if elem.element_type == "column"
    ]

    if len(column_ids) <= 1:
        # Single column — can't be redundant
        return False

    # Accidental load combination: 1.0*DL + 0.5*LL
    from structural_design_env.solver.load_generator import generate_loads
    from structural_design_env.models import TaskConfig

    # Build a reduced task config with accidental combination
    acc_config = task_config.model_copy(
        update={
            "dead_load_kPa": task_config.dead_load_kPa,
            "live_load_kPa": task_config.live_load_kPa * 0.5,
            "wind_load_kN_per_m": 0.0,   # no wind in accidental
            "seismic_ag_g": 0.0,          # no seismic in accidental
        }
    )

    for col_id in column_ids:
        # Copy the graph and remove the column
        reduced_graph = graph.copy()
        reduced_graph.remove_element(col_id)

        if len(reduced_graph.elements) == 0:
            return False  # removing any column leaves nothing

        # Generate loads on the reduced structure
        loads = generate_loads(reduced_graph, acc_config)

        # Solve
        result = solver.solve(reduced_graph, loads)

        if not result.converged:
            return False

        # Check all members for UR <= 2.0 (robustness factor)
        for eid, forces in result.member_forces.items():
            elem = reduced_graph.elements.get(eid)
            if elem is None:
                continue
            if elem.element_type == "wall":
                continue

            props = COLUMN_SECTIONS.get(elem.section) or BEAM_SECTIONS.get(elem.section)
            if props is None:
                continue

            checks = check_member(
                element_type=elem.element_type,
                section_props=props,
                forces=forces,
                L_m=elem.length_m,
            )

            if checks.max_UR > 2.0:
                return False

    return True
