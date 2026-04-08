"""
Three-stage physics reward for StructuralDesignEnv.

Stage 1: Member-level physics improvement (bending, buckling, violations)
Stage 2: Global structure behaviour (deflection, lateral drift)
Stage 3: Material efficiency
Plus terminal reward and behavioural shaping.
"""

from __future__ import annotations

import numpy as np

from structural_design_env.models import StructuralAction, StructuralObservation, TaskConfig


def compute_reward(
    prev_obs: StructuralObservation,
    curr_obs: StructuralObservation,
    action: StructuralAction,
    task_config: TaskConfig,
) -> float:
    r = 0.0

    # ------------------------------------------------------------------
    # Stage 1: Member-level physics improvement
    # ------------------------------------------------------------------

    # Bending utilisation reduction
    delta_max_UR = prev_obs.max_UR_bending - curr_obs.max_UR_bending
    r += 0.25 * float(np.tanh(2.0 * delta_max_UR))

    # Buckling utilisation reduction
    delta_buckle = prev_obs.max_UR_buckling - curr_obs.max_UR_buckling
    r += 0.15 * float(np.tanh(2.0 * delta_buckle))

    # Code violations fixed
    violations_fixed = prev_obs.n_code_violations - curr_obs.n_code_violations
    r += 0.10 * violations_fixed

    # ------------------------------------------------------------------
    # Stage 2: Global structure performance
    # ------------------------------------------------------------------

    # Deflection
    defl_limit = task_config.deflection_limit_mm
    defl_ratio = curr_obs.max_deflection_mm / defl_limit if defl_limit > 0 else 0.0
    r += 0.10 * (1.0 - min(defl_ratio, 2.0))

    # Lateral drift
    drift_ratio = curr_obs.max_lateral_drift_ratio
    r += 0.10 * (1.0 - min(drift_ratio, 2.0))

    # ------------------------------------------------------------------
    # Stage 3: Material efficiency penalty
    # ------------------------------------------------------------------
    mass_ratio = curr_obs.total_steel_mass_kg / max(task_config.reference_mass_kg, 1.0)
    r -= 0.05 * max(0.0, mass_ratio - 1.0)

    # ------------------------------------------------------------------
    # Behavioural shaping
    # ------------------------------------------------------------------
    if action.action_type == "remove_element":
        r -= 0.03

    if curr_obs.last_action_result == "INVALID":
        r -= 0.10

    # Wasteful upgrade: penalise upgrading a member that is already under-utilised
    if action.action_type == "upgrade_section" and action.element_id:
        for cm in prev_obs.critical_members:
            if cm.id == action.element_id and cm.max_UR < 0.60:
                r -= 0.05
                break

    # ------------------------------------------------------------------
    # Terminal reward
    # ------------------------------------------------------------------
    if action.action_type == "done":
        if curr_obs.is_structurally_valid:
            efficiency = max(0.0, 1.0 - curr_obs.total_steel_mass_kg / task_config.reference_mass_kg)
            r += 0.5 * efficiency

            if curr_obs.critical_members:
                avg_UR = np.mean([m.max_UR for m in curr_obs.critical_members])
            else:
                avg_UR = 0.0
            safety_bonus = max(0.0, 0.85 - avg_UR) * 0.3
            r += safety_bonus
        else:
            n_elem = max(1, len(curr_obs.placed_elements))
            fail_severity = curr_obs.n_code_violations / n_elem
            r -= 0.3 * (1.0 + fail_severity)

    return float(np.clip(r, -1.0, 1.0))
