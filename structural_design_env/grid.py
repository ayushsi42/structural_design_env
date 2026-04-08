"""
PlanGrid: 2D plan grid state (top-view layout per floor).

Grid is 20×20 integer array per floor:
  0 = empty
  1 = column
  2 = beam_x  (horizontal, along x-axis)
  3 = beam_y  (vertical, along y-axis)
  4 = wall

ASCII representation:
  '.' = empty
  'C' = column
  'B' = beam (any direction)
  'W' = wall
"""

from __future__ import annotations

from typing import List

import numpy as np

GRID_SIZE = 20

EMPTY = 0
COLUMN = 1
BEAM_X = 2
BEAM_Y = 3
WALL = 4

_ASCII_MAP = {
    EMPTY: ".",
    COLUMN: "C",
    BEAM_X: "B",
    BEAM_Y: "B",
    WALL: "W",
}


class PlanGrid:
    """Maintains a 20×20 integer grid per floor for plan layout."""

    def __init__(self, n_floors: int = 1):
        self.n_floors = n_floors
        # grids[floor][row][col] — row=y axis, col=x axis
        self._grids: List[np.ndarray] = [
            np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
            for _ in range(n_floors)
        ]

    def _check_floor(self, floor: int):
        if not (0 <= floor < self.n_floors):
            raise ValueError(f"Floor {floor} out of range [0, {self.n_floors - 1}]")

    def _check_coord(self, x: int, y: int):
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            raise ValueError(f"Grid position ({x},{y}) out of range [0,19]")

    # ------------------------------------------------------------------
    # Set / clear
    # ------------------------------------------------------------------

    def set(self, x: int, y: int, floor: int, value: int):
        self._check_floor(floor)
        self._check_coord(x, y)
        self._grids[floor][y, x] = value

    def get(self, x: int, y: int, floor: int) -> int:
        self._check_floor(floor)
        self._check_coord(x, y)
        return int(self._grids[floor][y, x])

    def clear(self, x: int, y: int, floor: int):
        self.set(x, y, floor, EMPTY)

    # ------------------------------------------------------------------
    # Element placement helpers
    # ------------------------------------------------------------------

    def place_column(self, x: int, y: int, floor: int):
        self.set(x, y, floor, COLUMN)

    def place_beam(self, x1: int, y1: int, x2: int, y2: int, floor: int, orientation: str):
        value = BEAM_X if orientation == "x" else BEAM_Y
        # Mark every cell along the path
        if x1 == x2:
            # vertical beam (y-direction)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for y in range(y_min, y_max + 1):
                if self.get(x1, y, floor) == EMPTY:
                    self.set(x1, y, floor, value)
        else:
            # horizontal beam (x-direction)
            x_min, x_max = min(x1, x2), max(x1, x2)
            for x in range(x_min, x_max + 1):
                if self.get(x, y1, floor) == EMPTY:
                    self.set(x, y1, floor, value)

    def place_wall(self, x1: int, y1: int, x2: int, y2: int, floor: int):
        if x1 == x2:
            y_min, y_max = min(y1, y2), max(y1, y2)
            for y in range(y_min, y_max + 1):
                self.set(x1, y, floor, WALL)
        else:
            x_min, x_max = min(x1, x2), max(x1, x2)
            for x in range(x_min, x_max + 1):
                self.set(x, y1, floor, WALL)

    # ------------------------------------------------------------------
    # ASCII visualization
    # ------------------------------------------------------------------

    def to_ascii_grid(self, floor: int) -> List[List[str]]:
        """Return 20×20 list of strings representing the floor plan."""
        self._check_floor(floor)
        grid = self._grids[floor]
        result = []
        for row in range(GRID_SIZE - 1, -1, -1):  # top row = high y
            result.append([_ASCII_MAP.get(int(grid[row, col]), "?") for col in range(GRID_SIZE)])
        return result

    def to_dict(self) -> dict:
        return {
            "n_floors": self.n_floors,
            "grids": [g.tolist() for g in self._grids],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlanGrid":
        obj = cls(n_floors=data["n_floors"])
        for i, g in enumerate(data["grids"]):
            obj._grids[i] = np.array(g, dtype=np.int32)
        return obj
