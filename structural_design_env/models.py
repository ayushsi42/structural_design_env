"""
Pydantic models for StructuralDesignEnv:
- StructuralAction
- CriticalMember
- StructuralObservation
- TaskConfig
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class StructuralAction(BaseModel):
    action_type: Literal[
        "place_column",
        "place_beam",
        "upgrade_section",
        "downgrade_section",
        "remove_element",
        "add_wall",
        "done",
    ]
    grid_x: Optional[int] = None
    grid_y: Optional[int] = None
    floor: Optional[int] = None
    section: Optional[str] = None
    from_node_x: Optional[int] = None
    from_node_y: Optional[int] = None
    to_node_x: Optional[int] = None
    to_node_y: Optional[int] = None
    orientation: Optional[str] = None  # "x" or "y"
    element_id: Optional[str] = None
    thickness_m: Optional[float] = None
    reasoning: Optional[str] = None


class CriticalMember(BaseModel):
    id: str
    type: str
    section: str
    length_m: float
    UR_bending: float
    UR_shear: float
    UR_buckling: float
    UR_deflection: float
    max_UR: float
    N_Ed_kN: float
    M_Ed_kNm: float
    V_Ed_kN: float


class StructuralObservation(BaseModel):
    # Site context
    site_width_m: float
    site_depth_m: float
    n_floors: int
    floor_height_m: float
    dead_load_kPa: float
    live_load_kPa: float
    wind_load_kN_per_m: float
    seismic_ag_g: float
    task_id: str

    # Layout
    grid_plan: List[List[List[str]]]  # list of 20×20 ASCII grids (one per floor)
    placed_elements: List[dict]
    n_elements_placed: int

    # Physics
    critical_members: List[CriticalMember]
    max_UR_bending: float
    max_UR_buckling: float
    max_UR_shear: float
    max_deflection_mm: float
    max_lateral_drift_ratio: float
    n_code_violations: int
    is_structurally_valid: bool

    # Design metrics
    total_steel_mass_kg: float
    material_efficiency_score: float

    # Episode state
    step_count: int
    max_steps: int
    last_action_error: Optional[str] = None
    last_action_result: str  # "PLACED" | "INVALID" | "UPDATED" | "REMOVED" | "RESET"
    episode_id: str

    # Human-readable summary
    message: str = ""


class TaskConfig(BaseModel):
    task_id: str
    name: str
    difficulty: str
    site_width_m: float
    site_depth_m: float
    n_floors: int
    floor_height_m: float
    dead_load_kPa: float
    live_load_kPa: float
    wind_load_kN_per_m: float
    seismic_ag_g: float
    seismic_gamma_I: float
    max_steps: int
    reference_mass_kg: float
    deflection_limit_mm: float  # typically L/300
