"""
Task 3: Seismic Hospital — Hard difficulty.

3-story hospital in seismic Zone 3.
Scored by seismic drift + budget + redundancy + utilisation.
"""

from structural_design_env.models import TaskConfig, StructuralObservation

TASK3_CONFIG = TaskConfig(
    task_id="task3_hospital",
    name="Seismic Hospital",
    difficulty="hard",
    site_width_m=20.0,
    site_depth_m=20.0,
    n_floors=3,
    floor_height_m=4.0,
    dead_load_kPa=6.0,
    live_load_kPa=4.0,
    wind_load_kN_per_m=2.0,
    seismic_ag_g=0.25,
    seismic_gamma_I=1.5,
    max_steps=85,
    reference_mass_kg=14000.0,
    deflection_limit_mm=13.3,  # 4000/300 ≈ 13.3 mm
)


def grade_task3(final_obs: StructuralObservation, graph=None) -> float:
    """
    Grade the hospital design.

    Parameters
    ----------
    final_obs : StructuralObservation
    graph : StructuralGraph (needed for redundancy check)

    Returns a score in [0, 1].
    """
    if not final_obs.is_structurally_valid:
        return round(max(0.0, 0.05 * (1.0 - final_obs.n_code_violations / 40)), 4)

    # Seismic performance (drift)
    seismic_ok = final_obs.max_lateral_drift_ratio <= 1.0
    seismic_score = (
        1.0
        if seismic_ok
        else max(0.0, 1.0 - (final_obs.max_lateral_drift_ratio - 1.0) * 2.0)
    )

    # Budget (mass vs reference)
    budget_score = max(
        0.0,
        min(1.0, 1.0 - (final_obs.total_steel_mass_kg - 15000.0) / 10000.0),
    )

    # Redundancy (progressive collapse)
    redundancy_score = 0.0
    if graph is not None:
        try:
            from structural_design_env.solver.redundancy import (
                check_column_removal_redundancy,
            )

            redundancy_ok = check_column_removal_redundancy(graph, TASK3_CONFIG)
            redundancy_score = 1.0 if redundancy_ok else 0.0
        except Exception:
            redundancy_score = 0.0

    # Utilisation ratio quality
    if final_obs.critical_members:
        avg_UR = sum(m.max_UR for m in final_obs.critical_members) / len(
            final_obs.critical_members
        )
    else:
        avg_UR = 0.0

    util_score = (
        1.0 if 0.70 <= avg_UR <= 0.85 else max(0.0, 1.0 - abs(avg_UR - 0.775) / 0.2)
    )

    score = (
        0.30 * seismic_score
        + 0.30 * budget_score
        + 0.25 * redundancy_score
        + 0.15 * util_score
    )
    return round(max(0.0, min(1.0, score)), 4)
