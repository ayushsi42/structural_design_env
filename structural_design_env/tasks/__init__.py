"""Task definitions for StructuralDesignEnv."""

from .task1_warehouse import TASK1_CONFIG, grade_task1
from .task2_office import TASK2_CONFIG, grade_task2
from .task3_hospital import TASK3_CONFIG, grade_task3

TASK_REGISTRY = {
    "task1_warehouse": (TASK1_CONFIG, grade_task1),
    "task2_office": (TASK2_CONFIG, grade_task2),
    "task3_hospital": (TASK3_CONFIG, grade_task3),
}

__all__ = [
    "TASK1_CONFIG",
    "TASK2_CONFIG",
    "TASK3_CONFIG",
    "grade_task1",
    "grade_task2",
    "grade_task3",
    "TASK_REGISTRY",
]
