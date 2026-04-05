"""
PGSA Environment — Core Implementation
----------------------------------------
PGSAEnvironment implements the OpenEnv Environment base class.
Handles all 8 PGSA action types, runs physics scoring, and
assembles natural-language observations for LLM agents.
"""

from __future__ import annotations

import json
import random
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pgsa_env.server.physics_sim import VoxelGrid, ProbePhysics, compute_reward, MATERIALS
from pgsa_env.server.task_generator import TaskSpec, generate_task, sample_hidden_params
from pgsa_env.server.graders import grade


# ─── MOTIF LIBRARY (20 predefined motifs) ─────────────────────────────────────

def _get_motif(motif_id: int) -> List[dict]:
    """
    Return relative element placements for the given motif_id.
    Each placement: {dx, dy, dz, element_type, material_id}
    """
    motifs = {
        0:  [{"dx":0,"dy":0,"dz":0,"element_type":"BEAM","material_id":0}],  # simple column
        1:  [{"dx":0,"dy":0,"dz":0,"element_type":"BEAM","material_id":0},
             {"dx":1,"dy":0,"dz":0,"element_type":"BEAM","material_id":0}],  # L-beam
        2:  [{"dx":0,"dy":0,"dz":0,"element_type":"WALL","material_id":0},
             {"dx":0,"dy":1,"dz":0,"element_type":"WALL","material_id":0},
             {"dx":0,"dy":2,"dz":0,"element_type":"WALL","material_id":0}],  # wall column
        3:  [  # portal frame
             {"dx":0,"dy":0,"dz":0,"element_type":"BEAM","material_id":1},
             {"dx":0,"dy":1,"dz":0,"element_type":"BEAM","material_id":1},
             {"dx":0,"dy":2,"dz":0,"element_type":"BEAM","material_id":1},
             {"dx":2,"dy":0,"dz":0,"element_type":"BEAM","material_id":1},
             {"dx":2,"dy":1,"dz":0,"element_type":"BEAM","material_id":1},
             {"dx":2,"dy":2,"dz":0,"element_type":"BEAM","material_id":1},
             {"dx":0,"dy":3,"dz":0,"element_type":"FLOOR","material_id":1},
             {"dx":1,"dy":3,"dz":0,"element_type":"FLOOR","material_id":1},
             {"dx":2,"dy":3,"dz":0,"element_type":"FLOOR","material_id":1},
        ],
        4:  [  # simple wall panel 2x3
             {"dx":0,"dy":0,"dz":0,"element_type":"WALL","material_id":0},
             {"dx":0,"dy":1,"dz":0,"element_type":"WALL","material_id":0},
             {"dx":0,"dy":2,"dz":0,"element_type":"WALL","material_id":0},
             {"dx":1,"dy":0,"dz":0,"element_type":"WALL","material_id":0},
             {"dx":1,"dy":1,"dz":0,"element_type":"WALL","material_id":0},
             {"dx":1,"dy":2,"dz":0,"element_type":"WALL","material_id":0},
        ],
        5:  [  # windowed wall panel
             {"dx":0,"dy":0,"dz":0,"element_type":"WALL","material_id":0},
             {"dx":0,"dy":1,"dz":0,"element_type":"WINDOW","material_id":3},
             {"dx":0,"dy":2,"dz":0,"element_type":"WALL","material_id":0},
        ],
        6:  [  # floor bay 3x3
             {"dx":i,"dy":0,"dz":k,"element_type":"FLOOR","material_id":1}
             for i in range(3) for k in range(3)
        ],
        7:  [  # corner frame
             {"dx":0,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":0,"dy":1,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":0,"dy":2,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":0,"dy":0,"dz":2,"element_type":"BEAM","material_id":2},
             {"dx":0,"dy":1,"dz":2,"element_type":"BEAM","material_id":2},
             {"dx":0,"dy":2,"dz":2,"element_type":"BEAM","material_id":2},
        ],
        8:  [  # simple room box (4 walls + floor + door + window)
             *[{"dx":0,"dy":i,"dz":j,"element_type":"WALL","material_id":0}
               for i in range(3) for j in range(4)],
             *[{"dx":j,"dy":i,"dz":0,"element_type":"WALL","material_id":0}
               for i in range(3) for j in range(1,3)],
             *[{"dx":j,"dy":i,"dz":3,"element_type":"WALL","material_id":0}
               for i in range(3) for j in range(1,3)],
             *[{"dx":j,"dy":3,"dz":k,"element_type":"FLOOR","material_id":1}
               for j in range(4) for k in range(4)],
             {"dx":0,"dy":1,"dz":1,"element_type":"DOOR","material_id":0},
             {"dx":0,"dy":1,"dz":2,"element_type":"WINDOW","material_id":3},
        ],
        9:  [  # T-junction (3 beams)
             {"dx":0,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":1,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":0,"dy":0,"dz":1,"element_type":"BEAM","material_id":2},
        ],
        10: [  # X-cross (4 beams)
             {"dx":0,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":1,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":0,"dy":0,"dz":1,"element_type":"BEAM","material_id":2},
             {"dx":-1,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
        ],
        11: [  # triangular truss (6 elements)
             {"dx":0,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":1,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":2,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":0,"dy":1,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":1,"dy":1,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":2,"dy":1,"dz":0,"element_type":"BEAM","material_id":2},
        ],
        12: [  # rectangular truss (8 elements)
             *[{"dx":i,"dy":0,"dz":0,"element_type":"BEAM","material_id":2} for i in range(4)],
             *[{"dx":i,"dy":1,"dz":0,"element_type":"BEAM","material_id":2} for i in range(4)],
        ],
        13: [  # staircase bay (12 elements — ascending steps in Y)
             *[{"dx":0,"dy":i,"dz":i,"element_type":"FLOOR","material_id":1} for i in range(4)],
             *[{"dx":0,"dy":i,"dz":i,"element_type":"WALL","material_id":0} for i in range(1,5)],
             *[{"dx":1,"dy":i,"dz":i,"element_type":"FLOOR","material_id":1} for i in range(4)],
        ],
        14: [  # arch (5-element curved approximation)
             {"dx":0,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":0,"dy":1,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":1,"dy":2,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":2,"dy":1,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":2,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
        ],
        15: [  # roof ridge (3 elements)
             {"dx":0,"dy":0,"dz":0,"element_type":"FLOOR","material_id":1},
             {"dx":1,"dy":0,"dz":0,"element_type":"FLOOR","material_id":1},
             {"dx":2,"dy":0,"dz":0,"element_type":"FLOOR","material_id":1},
        ],
        16: [  # vaulted room (22 elements — box + arch ceiling)
             *[{"dx":0,"dy":i,"dz":j,"element_type":"WALL","material_id":1} for i in range(3) for j in range(5)],
             *[{"dx":4,"dy":i,"dz":j,"element_type":"WALL","material_id":1} for i in range(3) for j in range(5)],
             {"dx":1,"dy":3,"dz":2,"element_type":"BEAM","material_id":2},
             {"dx":2,"dy":4,"dz":2,"element_type":"BEAM","material_id":2},
             {"dx":3,"dy":3,"dz":2,"element_type":"BEAM","material_id":2},
             {"dx":0,"dy":1,"dz":0,"element_type":"DOOR","material_id":0},
             {"dx":4,"dy":1,"dz":2,"element_type":"WINDOW","material_id":3},
        ],
        17: [  # cantilever bracket (4 elements)
             {"dx":0,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":1,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":2,"dy":0,"dz":0,"element_type":"BEAM","material_id":2},
             {"dx":2,"dy":1,"dz":0,"element_type":"JOINT","material_id":2},
        ],
        18: [  # shear wall pair (6 elements)
             *[{"dx":0,"dy":i,"dz":0,"element_type":"WALL","material_id":1} for i in range(3)],
             *[{"dx":3,"dy":i,"dz":0,"element_type":"WALL","material_id":1} for i in range(3)],
        ],
        19: [  # moment frame (10 elements)
             *[{"dx":0,"dy":i,"dz":0,"element_type":"BEAM","material_id":2} for i in range(4)],
             *[{"dx":4,"dy":i,"dz":0,"element_type":"BEAM","material_id":2} for i in range(4)],
             {"dx":1,"dy":4,"dz":0,"element_type":"FLOOR","material_id":1},
             {"dx":2,"dy":4,"dz":0,"element_type":"FLOOR","material_id":1},
        ],
    }
    return motifs.get(motif_id % 20, motifs[0])


# ─── ENVIRONMENT ──────────────────────────────────────────────────────────────

class PGSAEnvironment:
    """
    PGSA OpenEnv environment.
    Manages episode state, action dispatch, physics scoring, and reward.
    """

    MAX_PROBE_USES = 50

    def __init__(self):
        self._grid: Optional[VoxelGrid] = None
        self._task: Optional[TaskSpec] = None
        self._episode_id: str = ""
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._done: bool = True
        self._action_history: List[dict] = []
        self._probe_count: int = 0
        self._reward_breakdown: dict = {}
        self._graded_score: Optional[float] = None
        self._last_probe_result: Optional[dict] = None
        self._hidden_params: dict = {}

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def reset(self, difficulty: str = "easy", seed: Optional[int] = None) -> dict:
        """Initialize a new episode. Returns initial observation dict."""
        self._task = generate_task(difficulty, seed=seed)
        self._hidden_params = sample_hidden_params(self._task, seed=seed)
        W, H, D = self._task.grid_dims
        self._grid = VoxelGrid(W, H, D, hidden_params=self._hidden_params)
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._action_history = []
        self._probe_count = 0
        self._reward_breakdown = {}
        self._graded_score = None
        self._last_probe_result = None
        return self._build_observation(
            message=f"Episode started. {self._task.hint}",
            just_reset=True
        )

    def step(self, action_message: str) -> Tuple[dict, float, bool, dict]:
        """
        Process one action.
        Returns: (observation, reward, done, info)
        """
        if self._done:
            return self._build_observation("Episode is already done. Call reset()."), 0.0, True, {}

        # Parse action JSON
        action, parse_error = self._parse_action(action_message)
        if parse_error:
            obs = self._build_observation(
                f"⚠ Action parse error: {parse_error}. "
                "Send a valid JSON action. Example: "
                '{"action_type": "PLACE_ELEMENT", "x": 1, "y": 1, "z": 1, '
                '"element_type": "WALL", "material_id": 0, "orientation": 0}'
            )
            return obs, -0.05, False, {"error": parse_error}

        self._step_count += 1
        action_type = action.get("action_type", "").upper()

        # Dispatch action
        result_msg, reward_delta, terminated = self._dispatch_action(action, action_type)

        # Record history (trimmed)
        self._action_history.append({
            "step": self._step_count,
            "action_type": action_type,
            "result": result_msg[:120],
        })
        if len(self._action_history) > 20:
            self._action_history = self._action_history[-20:]

        # Compute reward if not a terminal action
        if action_type != "COMMIT_DESIGN" and self._grid:
            self._reward_breakdown = compute_reward(
                grid=self._grid,
                task=self._task.to_dict(),
                curriculum_level=self._task.curriculum_level,
                steps_remaining=self._task.action_budget - self._step_count,
                total_steps=self._task.action_budget,
                budget=self._task.budget,
                probe_count=self._probe_count,
                is_commit=False,
            )
            # Only award incremental delta
            new_norm = self._reward_breakdown["r_normalized"]
            reward_delta = max(-0.5, new_norm - (self._total_reward if self._step_count > 1 else 0.0))
            reward_delta = min(reward_delta, 0.2)  # cap per-step gain

        # Check truncation — action budget
        if self._step_count >= self._task.action_budget:
            terminated = True
            result_msg += " | Action budget exhausted."

        # §6.5: Cost budget exceeded → truncation with −2.0 penalty
        if self._grid and self._grid.total_cost > self._task.budget:
            terminated = True
            reward_delta -= 2.0
            result_msg += " | Material budget exceeded (−2.0 penalty)."

        if self._grid and self._reward_breakdown.get("stability_class") == "COLLAPSE":
            terminated = True
            reward_delta = -0.5
            result_msg += " | COLLAPSE! Structure failed catastrophically."

        self._total_reward += reward_delta
        self._done = terminated

        obs = self._build_observation(result_msg)
        # §9.5: Full info dictionary
        action_mask = self._compute_action_mask()
        reward_bkdn = self._reward_breakdown
        constraint_flags = {
            "room_complete": reward_bkdn.get("n_valid_rooms", 0) >= max(len(self._task.required_rooms), 1),
            "connected":     reward_bkdn.get("connected_fraction", 0.0) >= 0.8,
            "airflow_ok":    reward_bkdn.get("airflow_score", 0.0) >= 0.5,
            "light_ok":      reward_bkdn.get("light_score", 0.0) >= 0.5,
            "egress_ok":     reward_bkdn.get("egress_score", 0.0) >= 1.0,
            "density_ok":    reward_bkdn.get("density_score", 0.0) >= 0.5,
        } if self._task else {}
        # §9.5: physics_belief_error (mean relative error of nominal vs hidden)
        physics_belief_error = 0.0
        if self._task and self._task.hidden_material_props and self._hidden_params:
            errors = []
            for mat_id, mat in MATERIALS.items():
                hidden = self._hidden_params.get(mat_id, {})
                for prop in ("yield_mpa", "E_gpa"):
                    true_val = hidden.get(prop, mat[prop])
                    nom_val = mat[prop]
                    errors.append(abs(nom_val - true_val) / max(true_val, 1e-6))
            physics_belief_error = sum(errors) / max(len(errors), 1)
        info = {
            "step": self._step_count,
            "reward_breakdown": reward_bkdn,
            "constraint_flags": constraint_flags,
            "probe_budget_remaining": max(0, self._task.probe_budget - self._probe_count),
            "graded_score": self._graded_score,
            "probe_result": self._last_probe_result,
            "action_mask": action_mask,
            "physics_belief_error": round(physics_belief_error, 4),
            "n_valid_rooms": reward_bkdn.get("n_valid_rooms", 0),
            "curriculum_level": self._task.curriculum_level if self._task else 1,
            "budget_remaining": (self._task.budget - self._grid.total_cost) if self._grid and self._task else 0,
            "stability_class": reward_bkdn.get("stability_class", "UNKNOWN"),
        }
        self._last_probe_result = None
        return obs, round(reward_delta, 4), terminated, info

    def state(self) -> dict:
        """Return current episode state metadata."""
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "curriculum_level": self._task.curriculum_level if self._task else 1,
            "task_id": self._task.task_id if self._task else "",
            "task_difficulty": self._task.difficulty if self._task else "easy",
            "total_reward": round(self._total_reward, 4),
            "is_done": self._done,
            "budget_remaining": (
                self._task.budget - self._grid.total_cost
                if self._grid and self._task else 0
            ),
            "graded_score": self._graded_score,
        }

    # ── Action Dispatch ────────────────────────────────────────────────────────

    def _dispatch_action(
        self, action: dict, action_type: str
    ) -> Tuple[str, float, bool]:
        """
        Route to the appropriate action handler.
        Returns: (message, reward_delta, terminated)
        """
        try:
            if action_type == "PLACE_ELEMENT":
                return self._action_place(action)
            elif action_type == "REMOVE_ELEMENT":
                return self._action_remove(action)
            elif action_type == "REPLACE_MATERIAL":
                return self._action_replace_material(action)
            elif action_type == "PLACE_BATCH":
                return self._action_place_batch(action)
            elif action_type == "PROBE_PHYSICS":
                return self._action_probe(action)
            elif action_type == "ANNOTATE_ROOM":
                return self._action_annotate(action)
            elif action_type == "QUERY_BELIEF":
                return self._action_query_belief(action)
            elif action_type == "COMMIT_DESIGN":
                return self._action_commit(action)
            else:
                return (
                    f"Unknown action_type '{action_type}'. "
                    "Valid: PLACE_ELEMENT, REMOVE_ELEMENT, REPLACE_MATERIAL, "
                    "PLACE_BATCH, PROBE_PHYSICS, ANNOTATE_ROOM, QUERY_BELIEF, COMMIT_DESIGN",
                    -0.05, False
                )
        except Exception as e:
            return f"Action error: {e}", -0.05, False

    def _action_place(self, a: dict) -> Tuple[str, float, bool]:
        x, y, z = int(a.get("x", 0)), int(a.get("y", 0)), int(a.get("z", 0))
        et = str(a.get("element_type", "BEAM")).upper()
        mat = int(a.get("material_id", 0))
        ori = int(a.get("orientation", 0))

        # Budget check
        cost = MATERIALS.get(mat, {}).get("cost", 1)
        remaining = self._task.budget - self._grid.total_cost
        if cost > remaining:
            return f"Insufficient budget ({remaining} remaining, {cost} needed)", -0.02, False

        ok, msg = self._grid.place(x, y, z, et, mat, ori)
        return msg, 0.0, False

    def _action_remove(self, a: dict) -> Tuple[str, float, bool]:
        x, y, z = int(a.get("x", 0)), int(a.get("y", 0)), int(a.get("z", 0))
        ok, msg = self._grid.remove(x, y, z)
        return msg, 0.0, False

    def _action_replace_material(self, a: dict) -> Tuple[str, float, bool]:
        x, y, z = int(a.get("x", 0)), int(a.get("y", 0)), int(a.get("z", 0))
        new_mat = int(a.get("new_material_id", a.get("material_id", 0)))
        ok, msg = self._grid.replace_material(x, y, z, new_mat)
        return msg, 0.0, False

    def _action_place_batch(self, a: dict) -> Tuple[str, float, bool]:
        cx, cy, cz = int(a.get("x", 0)), int(a.get("y", 0)), int(a.get("z", 0))
        motif_id = int(a.get("motif_id", 0))
        scale = max(0.5, min(4.0, float(a.get("scale", 1.0))))
        override_mat = a.get("material_id", None)
        placements = _get_motif(motif_id)
        placed, skipped = 0, 0
        total_cost = 0
        msgs = []
        for p in placements:
            dx = round(p["dx"] * scale)
            dy = round(p["dy"] * scale)
            dz = round(p["dz"] * scale)
            mat = int(override_mat) if override_mat is not None else int(p["material_id"])
            et = p["element_type"]
            cost = MATERIALS.get(mat, {}).get("cost", 1)
            remaining = self._task.budget - self._grid.total_cost
            if cost > remaining:
                skipped += 1
                continue
            ok, msg = self._grid.place(cx + dx, cy + dy, cz + dz, et, mat, 0)
            if ok:
                placed += 1
                total_cost += cost
            else:
                skipped += 1
        return (
            f"PLACE_BATCH motif={motif_id} scale={scale}: placed {placed} elements "
            f"(skipped {skipped}) | cost={total_cost}",
            0.0, False
        )

    def _action_probe(self, a: dict) -> Tuple[str, float, bool]:
        if self._task.probe_budget == 0:
            return "PROBE_PHYSICS not available at this curriculum level.", -0.05, False
        if self._probe_count >= self._task.probe_budget:
            return f"Probe budget exhausted ({self._task.probe_budget} used).", -0.05, False

        x, y, z = int(a.get("x", 0)), int(a.get("y", 0)), int(a.get("z", 0))
        load_kn = float(a.get("load_kn", 10.0))
        direction = str(a.get("direction", "Y")).upper()

        result = ProbePhysics.probe(self._grid, x, y, z, load_kn, direction)
        self._probe_count += 1
        self._last_probe_result = result

        if "error" in result:
            return f"Probe failed: {result['error']}", -0.02, False

        return (
            f"PROBE RESULT at ({x},{y},{z}): {result['hint']} "
            f"[Probes used: {self._probe_count}/{self._task.probe_budget}]",
            0.0, False
        )

    def _action_annotate(self, a: dict) -> Tuple[str, float, bool]:
        x1, y1, z1 = int(a.get("x1", 0)), int(a.get("y1", 1)), int(a.get("z1", 0))
        x2, y2, z2 = int(a.get("x2", 1)), int(a.get("y2", 2)), int(a.get("z2", 1))
        rt = str(a.get("room_type", "GENERIC")).upper()
        room_id, msg = self._grid.annotate_room(x1, y1, z1, x2, y2, z2, rt)
        reward_delta = 0.05 if room_id >= 0 else 0.0  # small reward for successfully identifying a room
        return msg, reward_delta, False

    def _action_query_belief(self, a: dict) -> Tuple[str, float, bool]:
        """Return current physics belief state (probe summary)."""
        lines = ["PHYSICS BELIEF UPDATE:"]
        for mat_id, mat_data in MATERIALS.items():
            hidden = self._hidden_params.get(mat_id, {})
            true_yield = hidden.get("yield_mpa", mat_data["yield_mpa"])
            true_E = hidden.get("E_gpa", mat_data["E_gpa"])
            if self._task.hidden_material_props:
                lines.append(
                    f"  Material {mat_id} ({mat_data['name']}): "
                    f"yield_nominal={mat_data['yield_mpa']} MPa (true≈±30%), "
                    f"E_nominal={mat_data['E_gpa']} GPa (true≈±30%)"
                )
            else:
                lines.append(
                    f"  Material {mat_id} ({mat_data['name']}): "
                    f"yield={mat_data['yield_mpa']} MPa, E={mat_data['E_gpa']} GPa"
                )
        lines.append(
            f"Probes used: {self._probe_count}/{self._task.probe_budget}. "
            "Use PROBE_PHYSICS to narrow uncertainty on placed elements."
        )
        return "\n".join(lines), 0.0, False

    def _action_commit(self, a: dict) -> Tuple[str, float, bool]:
        """Finalize design. Run full grader and terminate episode."""
        if self._step_count < 1:
            return "Must take at least 1 action before committing.", -0.05, False

        # Final full reward
        self._reward_breakdown = compute_reward(
            grid=self._grid,
            task=self._task.to_dict(),
            curriculum_level=self._task.curriculum_level,
            steps_remaining=self._task.action_budget - self._step_count,
            total_steps=self._task.action_budget,
            budget=self._task.budget,
            probe_count=self._probe_count,
            is_commit=True,
        )

        # Run grader
        episode_state = {
            "reward_breakdown": self._reward_breakdown,
            "grid_summary": self._grid.summary_text() if self._grid else "",
            "probe_count": self._probe_count,
        }
        self._graded_score = grade(self._task.task_id, episode_state)

        # Terminal reward = graded_score (as a delta from current total)
        terminal_reward = self._graded_score - max(0.0, self._total_reward)

        msg = (
            f"DESIGN COMMITTED ✓\n"
            f"Graded Score: {self._graded_score:.4f} / 1.0\n"
            f"Stability: {self._reward_breakdown['stability_class']}\n"
            f"Rooms: {self._reward_breakdown['n_valid_rooms']} valid\n"
            f"Airflow: {self._reward_breakdown['airflow_score']:.2f} | "
            f"Light: {self._reward_breakdown['light_score']:.2f} | "
            f"Egress: {self._reward_breakdown['egress_score']:.2f}\n"
            f"Steps used: {self._step_count} / {self._task.action_budget}\n"
            f"Cost used: {self._grid.total_cost} / {self._task.budget}"
        )
        return msg, terminal_reward, True

    # ── Observation Builder ────────────────────────────────────────────────────

    def _build_observation(self, message: str, just_reset: bool = False) -> dict:
        """Assemble the full observation dict returned to the agent."""
        grid_summary = self._grid.summary_text() if self._grid else "No grid initialized."
        stability = self._reward_breakdown.get("stability_class", "UNKNOWN")
        n_valid = self._reward_breakdown.get("n_valid_rooms", 0)
        budget_rem = (self._task.budget - self._grid.total_cost) if self._grid and self._task else 0
        probe_rem = max(0, self._task.probe_budget - self._probe_count) if self._task else 0

        constraint_status = {
            "stability_class": stability,
            "n_valid_rooms": n_valid,
            "room_score": round(self._reward_breakdown.get("room_score", 0.0), 3),
            "airflow_score": round(self._reward_breakdown.get("airflow_score", 0.0), 3),
            "light_score": round(self._reward_breakdown.get("light_score", 0.0), 3),
            "egress_score": round(self._reward_breakdown.get("egress_score", 0.0), 3),
            "connected_fraction": round(self._reward_breakdown.get("connected_fraction", 0.0), 3),
        }

        full_message = (
            f"[Step {self._step_count}] {message}\n"
            f"---\n{grid_summary}\n"
            f"---\n"
            f"Stability: {stability} | Rooms: {n_valid} valid\n"
            f"Budget: {budget_rem} remaining | Probe budget: {probe_rem} remaining\n"
            f"Cumulative reward: {self._total_reward:.4f}"
        )

        return {
            "message": full_message,
            "task_description": self._task.description if self._task else "",
            "grid_summary": grid_summary,
            "constraint_status": constraint_status,
            "reward_breakdown": self._reward_breakdown,
            "budget_remaining": budget_rem,
            "probe_budget_remaining": probe_rem,
            "step": self._step_count,
            "done": self._done,
            "graded_score": self._graded_score,
            "info": {
                "action_history": self._action_history[-5:],
                "episode_id": self._episode_id,
                "probe_count": self._probe_count,
            },
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_action(message: str) -> Tuple[Optional[dict], Optional[str]]:
        """
        Extract JSON action from agent message.
        Tries to find a JSON object in the message string.
        """
        message = message.strip()
        # Try direct parse first
        try:
            action = json.loads(message)
            if isinstance(action, dict):
                return action, None
        except json.JSONDecodeError:
            pass

        # Try to extract JSON block from within text
        import re
        matches = re.findall(r'\{[^{}]*\}', message, re.DOTALL)
        for m in matches:
            try:
                action = json.loads(m)
                if isinstance(action, dict) and "action_type" in action:
                    return action, None
            except json.JSONDecodeError:
                continue

        # Check for common text commands
        upper = message.upper()
        if "COMMIT" in upper:
            return {"action_type": "COMMIT_DESIGN"}, None

        return None, (
            f"Could not parse JSON action from message: '{message[:100]}'. "
            "Send a JSON object with 'action_type' field."
        )

    def _compute_action_mask(self) -> dict:
        """
        Compute action validity mask per §6.4.
        Returns dict of {action_type_name: bool} for all 8 action types.
        """
        if not self._grid or not self._task:
            return {a: False for a in [
                "PLACE_ELEMENT", "REMOVE_ELEMENT", "REPLACE_MATERIAL",
                "PLACE_BATCH", "PROBE_PHYSICS", "ANNOTATE_ROOM",
                "QUERY_BELIEF", "COMMIT_DESIGN",
            ]}

        budget_remaining = self._task.budget - self._grid.total_cost
        cheapest_cost = 1  # WOOD
        n_non_foundation = self._grid.count_non_foundation()

        return {
            # Invalid if budget < cheapest element or no empty voxels
            "PLACE_ELEMENT":    budget_remaining >= cheapest_cost,
            # Invalid only if nothing to remove (all FOUNDATION)
            "REMOVE_ELEMENT":   n_non_foundation > 0,
            # Invalid if no non-FOUNDATION elements exist
            "REPLACE_MATERIAL": n_non_foundation > 0,
            # Invalid if insufficient budget for motif 0 at scale 0.5
            "PLACE_BATCH":      budget_remaining >= cheapest_cost,
            # Invalid if probe budget exhausted
            "PROBE_PHYSICS":    self._probe_count < self._task.probe_budget,
            # Never invalid
            "ANNOTATE_ROOM":    True,
            "QUERY_BELIEF":     True,
            # Invalid only before first action (step_count < 1)
            "COMMIT_DESIGN":    self._step_count >= 1,
        }
