"""
PGSA Task Graders
------------------
Three deterministic graders producing scores in [0.0, 1.0]:
  - grade_easy    → stable_shelter
  - grade_medium  → functional_office
  - grade_hard    → material_detective
"""

from __future__ import annotations


def grade_easy(episode_state: dict) -> float:
    """
    Grade: Stable Shelter
    Score = 0.5 × stability_score + 0.5 × room_score
    - stability_score: 1.0 if STABLE, 0.5 if SERVICEABILITY_FAIL, 0.0 if collapsed
    - room_score: 1.0 if ≥1 valid enclosed room w/ door, 0.5 if just rooms exist, 0.0 otherwise
    """
    reward = episode_state.get("reward_breakdown", {})
    stability_class = reward.get("stability_class", "COLLAPSE")
    n_valid_rooms = reward.get("n_valid_rooms", 0)
    room_score = reward.get("room_score", 0.0)
    connected_fraction = reward.get("connected_fraction", 0.0)
    stress_safe = reward.get("stress_safe_fraction", 0.0)

    # Stability sub-score
    if stability_class == "STABLE":
        stability_score = min(connected_fraction * stress_safe * 1.2, 1.0)
    elif stability_class == "SERVICEABILITY_FAIL":
        stability_score = 0.4
    elif stability_class == "PARTIAL_COLLAPSE":
        stability_score = 0.1
    else:  # COLLAPSE or no structure
        stability_score = 0.0

    # Room sub-score
    if n_valid_rooms >= 1:
        room_sub = min(room_score * 1.1, 1.0)
    elif n_valid_rooms == 0 and episode_state.get("grid_summary", "").count("WALL") > 2:
        room_sub = 0.15  # Has walls but no enclosed room
    else:
        room_sub = 0.0

    score = 0.5 * stability_score + 0.5 * room_sub
    return round(max(0.0, min(1.0, score)), 4)


def grade_medium(episode_state: dict) -> float:
    """
    Grade: Functional Office
    Weighted average of functional sub-scores.
    """
    reward = episode_state.get("reward_breakdown", {})
    stability_class = reward.get("stability_class", "COLLAPSE")

    # If structure collapsed, severe penalty
    if stability_class == "COLLAPSE":
        return 0.0

    room_score   = reward.get("room_score", 0.0)      # 35%
    conn_score   = reward.get("conn_score", 0.0)       # 20%
    airflow      = reward.get("airflow_score", 0.0)    # 15%
    lighting     = reward.get("light_score", 0.0)      # 15%
    egress       = reward.get("egress_score", 0.0)     # 10%
    density      = reward.get("density_score", 0.0)    # 5%

    # Structural bonus/penalty
    if stability_class == "STABLE":
        struct_bonus = 0.10
    elif stability_class == "SERVICEABILITY_FAIL":
        struct_bonus = 0.0
    else:
        struct_bonus = -0.15

    score = (
        0.35 * room_score +
        0.20 * conn_score +
        0.15 * airflow +
        0.15 * lighting +
        0.10 * egress +
        0.05 * density +
        struct_bonus
    )
    return round(max(0.0, min(1.0, score)), 4)


def grade_hard(episode_state: dict) -> float:
    """
    Grade: Material Detective
    Full functional reward + physics accuracy bonus from PROBE usage.
    """
    reward = episode_state.get("reward_breakdown", {})
    stability_class = reward.get("stability_class", "COLLAPSE")

    if stability_class == "COLLAPSE":
        return 0.0

    # Same functional components as medium
    room_score = reward.get("room_score", 0.0)
    conn_score = reward.get("conn_score", 0.0)
    airflow    = reward.get("airflow_score", 0.0)
    lighting   = reward.get("light_score", 0.0)
    egress     = reward.get("egress_score", 0.0)
    density    = reward.get("density_score", 0.0)

    functional_score = (
        0.35 * room_score +
        0.20 * conn_score +
        0.15 * airflow +
        0.15 * lighting +
        0.10 * egress +
        0.05 * density
    )

    # Physics accuracy component: bonus for probing
    probe_count = episode_state.get("probe_count", 0)
    probe_bonus = min(probe_count / 50.0, 1.0) * 0.20  # up to 20% bonus for probing

    # Structural quality (harder since material props are hidden)
    if stability_class == "STABLE":
        struct_bonus = 0.10
    elif stability_class == "SERVICEABILITY_FAIL":
        struct_bonus = -0.05
    else:
        struct_bonus = -0.20

    # Physics reward proxy: penalize if material uncertainty was high
    r_physics = reward.get("r_physics", -0.5)
    physics_score = max(0.0, (r_physics + 1.0) / 1.0) * 0.15  # normalize to [0, 0.15]

    score = (
        0.55 * functional_score +  # reduced weight vs medium
        physics_score +
        probe_bonus +
        struct_bonus
    )
    return round(max(0.0, min(1.0, score)), 4)


# ─── GRADER DISPATCH ─────────────────────────────────────────────────────────

GRADERS = {
    "stable_shelter":    grade_easy,
    "functional_office": grade_medium,
    "material_detective": grade_hard,
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


def grade(task_id: str, episode_state: dict) -> float:
    """Dispatch to the appropriate grader. Returns score ∈ [0.0, 1.0]."""
    grader = GRADERS.get(task_id, grade_easy)
    return grader(episode_state)
