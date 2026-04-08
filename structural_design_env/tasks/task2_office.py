"""
Task 2: Multi-Story Office — Medium difficulty.

3-story 20×20m office with wind and light seismic.
Scored by drift control + efficiency + torsional balance.
"""

from structural_design_env.models import TaskConfig, StructuralObservation

TASK2_CONFIG = TaskConfig(
    task_id="task2_office",
    name="Multi-Story Office",
    difficulty="medium",
    site_width_m=20.0,
    site_depth_m=20.0,
    n_floors=3,
    floor_height_m=3.5,
    dead_load_kPa=4.5,
    live_load_kPa=3.0,
    wind_load_kN_per_m=1.5,
    seismic_ag_g=0.04,
    seismic_gamma_I=1.0,
    max_steps=55,
    reference_mass_kg=3200.0,
    deflection_limit_mm=11.7,  # 3500/300 ≈ 11.7 mm
)


def _check_open_plan(placed_elements: list, site_width_m: float, site_depth_m: float, radius_m: float = 4.0) -> float:
    """
    Return 1.0 if no columns are within radius_m of the building centre.
    Degrades linearly to 0 as columns approach the centre.
    Node ID format: n_{x}_{y}_{floor}
    """
    center_x = site_width_m / 2.0   # 10.0 for 20m
    center_y = site_depth_m / 2.0   # 10.0 for 20m

    cols = [e for e in placed_elements if e.get("type") == "column"]
    if not cols:
        return 0.0  # No columns at all → penalise

    min_dist = float("inf")
    for col in cols:
        node_i = col.get("node_i", "")
        parts = node_i.split("_")
        if len(parts) >= 4:
            try:
                cx = int(parts[1])
                cy = int(parts[2])
                dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
            except ValueError:
                pass

    if min_dist == float("inf"):
        return 0.0
    if min_dist >= radius_m:
        return 1.0
    return max(0.0, min_dist / radius_m)


def grade_task2(final_obs: StructuralObservation) -> float:
    """
    Grade the office building design.

    Returns a score in [0, 1].
    """
    if not final_obs.is_structurally_valid:
        return round(max(0.0, 1.0 - final_obs.n_code_violations / 20) * 0.15, 4)

    # Drift performance
    drift_ok = final_obs.max_lateral_drift_ratio <= 1.0
    drift_score = (
        1.0
        if drift_ok
        else max(0.0, 1.0 - (final_obs.max_lateral_drift_ratio - 1.0))
    )

    # Material efficiency
    OPTIMAL_MASS = 3200.0
    mass_score = max(
        0.0,
        1.0 - (final_obs.total_steel_mass_kg - OPTIMAL_MASS) / 5000.0,
    )

    # Torsional balance: check for walls in both directions
    walls = [e for e in final_obs.placed_elements if e.get("type") == "wall"]
    walls_x = [w for w in walls if w.get("orientation") == "x"]
    walls_y = [w for w in walls if w.get("orientation") == "y"]
    torsion_score = 1.0 if (walls_x and walls_y) else 0.3

    # Open plan: no columns within 4m of building centre
    open_plan_score = _check_open_plan(
        final_obs.placed_elements, final_obs.site_width_m, final_obs.site_depth_m, radius_m=4.0
    )

    score = 0.35 * drift_score + 0.35 * mass_score + 0.20 * torsion_score + 0.10 * open_plan_score
    return round(max(0.0, min(1.0, score)), 4)
