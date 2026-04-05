"""
PGSA Pydantic Models
---------------------
Typed Action, Observation, and State models for the PGSA OpenEnv environment.
These are the data contracts between the agent and the environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── ACTION ───────────────────────────────────────────────────────────────────

class PGSAAction(BaseModel):
    """
    Action sent by the LLM agent.

    The `message` field must contain a JSON-encoded action object with at minimum
    an `action_type` field. Example valid messages:

    Place a wall:
      '{"action_type": "PLACE_ELEMENT", "x": 3, "y": 1, "z": 3,
        "element_type": "WALL", "material_id": 0, "orientation": 0}'

    Probe physics:
      '{"action_type": "PROBE_PHYSICS", "x": 3, "y": 1, "z": 3,
        "load_kn": 10.0, "direction": "Y"}'

    Annotate a room:
      '{"action_type": "ANNOTATE_ROOM", "x1": 1, "y1": 1, "z1": 1,
        "x2": 5, "y2": 4, "z2": 5, "room_type": "OFFICE"}'

    Commit design:
      '{"action_type": "COMMIT_DESIGN"}'

    Supported action_type values:
      PLACE_ELEMENT, REMOVE_ELEMENT, REPLACE_MATERIAL,
      PLACE_BATCH, PROBE_PHYSICS, ANNOTATE_ROOM, QUERY_BELIEF, COMMIT_DESIGN
    """
    message: str = Field(
        ...,
        description="JSON-encoded action. Must contain 'action_type' and relevant parameters.",
        examples=[
            '{"action_type": "PLACE_ELEMENT", "x": 3, "y": 1, "z": 3, "element_type": "WALL", "material_id": 0}',
            '{"action_type": "COMMIT_DESIGN"}',
        ]
    )


# ─── OBSERVATION ──────────────────────────────────────────────────────────────

class PGSAObservation(BaseModel):
    """
    Observation returned to the agent after each step or reset.

    The primary field for LLM agents is `message` — a natural language summary
    of the current observation. All other fields are structured data for
    programmatic access.
    """
    message: str = Field(
        ...,
        description=(
            "Natural-language observation summary for the LLM agent. "
            "Includes step number, action result, grid state, constraint scores, "
            "and remaining budgets."
        )
    )
    task_description: str = Field(
        ...,
        description="Full task specification. Constant within an episode. "
                    "Describes requirements, action format, and scoring."
    )
    grid_summary: str = Field(
        "", description="Text summary of the current voxel grid state."
    )
    constraint_status: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Current scores for each functional constraint: "
            "stability_class, n_valid_rooms, room_score, airflow_score, "
            "light_score, egress_score, connected_fraction."
        )
    )
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward values for the current step."
    )
    budget_remaining: int = Field(
        0, description="Material cost budget remaining for this episode."
    )
    probe_budget_remaining: int = Field(
        0, description="Number of PROBE_PHYSICS actions remaining."
    )
    step: int = Field(0, description="Current step count within the episode.")
    done: bool = Field(False, description="True if the episode has ended.")
    graded_score: Optional[float] = Field(
        None, description="Final graded score [0.0, 1.0]. Only set after COMMIT_DESIGN."
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary info: action_history, episode_id, probe_count."
    )


# ─── STATE ────────────────────────────────────────────────────────────────────

class PGSAState(BaseModel):
    """
    Episode-level state metadata.
    Returned by /state endpoint and env.state() call.
    """
    episode_id: str = Field("", description="Unique identifier for this episode.")
    step_count: int = Field(0, description="Number of steps taken in the current episode.")
    curriculum_level: int = Field(1, description="Current curriculum level (1=easy, 2=medium, 3=hard).")
    task_id: str = Field("", description="Task identifier (stable_shelter, functional_office, material_detective).")
    task_difficulty: str = Field("easy", description="Difficulty tier: easy | medium | hard.")
    total_reward: float = Field(0.0, description="Cumulative reward accumulated this episode.")
    is_done: bool = Field(False, description="Whether the episode has ended.")
    budget_remaining: int = Field(0, description="Remaining material cost budget.")
    graded_score: Optional[float] = Field(
        None, description="Final graded score. Only available after COMMIT_DESIGN."
    )
