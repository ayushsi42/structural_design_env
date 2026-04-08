"""
Task 1: Column Grid Warehouse — Easy difficulty.

Single-story 20×10m warehouse. No lateral loads.
Scored by validity + material efficiency.
"""

from structural_design_env.models import TaskConfig, StructuralObservation

TASK1_CONFIG = TaskConfig(
    task_id="task1_warehouse",
    name="Column Grid Warehouse",
    difficulty="easy",
    site_width_m=20.0,
    site_depth_m=10.0,
    n_floors=1,
    floor_height_m=4.0,
    dead_load_kPa=3.5,
    live_load_kPa=2.0,
    wind_load_kN_per_m=0.0,
    seismic_ag_g=0.0,
    seismic_gamma_I=1.0,
    max_steps=25,
    reference_mass_kg=520.0,
    deflection_limit_mm=16.7,  # 5000/300 ≈ 16.7 mm
)


def grade_task1(final_obs: StructuralObservation) -> float:
    """
    Grade the warehouse design.

    Returns a score in [0, 1].
    """
    if not final_obs.is_structurally_valid:
        partial = max(0.0, 1.0 - final_obs.n_code_violations / 10) * 0.2
        return round(partial, 4)

    OPTIMAL_MASS = 520.0
    efficiency = max(
        0.0,
        1.0 - (final_obs.total_steel_mass_kg - OPTIMAL_MASS) / 1000.0,
    )

    if final_obs.critical_members:
        avg_UR = sum(m.max_UR for m in final_obs.critical_members) / len(
            final_obs.critical_members
        )
    else:
        avg_UR = 0.0

    # Reward designs where UR is in the "sweet spot" 0.70–0.85
    if 0.70 <= avg_UR <= 0.85:
        margin_score = 1.0
    elif avg_UR < 0.70:
        margin_score = avg_UR / 0.70
    else:
        margin_score = max(0.0, 1.0 - (avg_UR - 0.85) / 0.15)

    score = 0.7 * efficiency + 0.3 * margin_score
    return round(max(0.0, min(1.0, score)), 4)
