"""
StructuralDesignEnv — OpenEnv environment for steel building frame design.

Agents place columns, beams, and shear walls on a structural grid and receive
physics analysis feedback via the direct stiffness method and Eurocode 3 checks.
"""

from structural_design_env.env import StructuralDesignEnv
from structural_design_env.models import (
    StructuralAction,
    StructuralObservation,
    TaskConfig,
    CriticalMember,
)

__version__ = "1.0.0"

__all__ = [
    "StructuralDesignEnv",
    "StructuralAction",
    "StructuralObservation",
    "TaskConfig",
    "CriticalMember",
]
