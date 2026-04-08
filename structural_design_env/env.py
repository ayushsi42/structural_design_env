"""
StructuralDesignEnv: Main environment class (OpenEnv specification).

The agent places columns, beams, and shear walls on a structural grid,
then receives physics analysis results from the direct stiffness solver
and Eurocode 3 compliance checks.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np

from structural_design_env.graph import StructuralGraph
from structural_design_env.grid import PlanGrid
from structural_design_env.models import (
    CriticalMember,
    StructuralAction,
    StructuralObservation,
    TaskConfig,
)
from structural_design_env.reward import compute_reward
from structural_design_env.solver import (
    StructuralSolver,
    check_member,
    generate_loads,
)
from structural_design_env.solver.sections import (
    COLUMN_SECTIONS,
    BEAM_SECTIONS,
    upgrade_section,
    downgrade_section,
)
from structural_design_env.tasks import TASK_REGISTRY
from structural_design_env.validation import validate_action


class StructuralDesignEnv:
    """
    Steel frame structural engineering environment.

    Follows the OpenEnv step/reset interface.
    """

    def __init__(self):
        self.graph: Optional[StructuralGraph] = None
        self.grid: Optional[PlanGrid] = None
        self.task_config: Optional[TaskConfig] = None
        self.episode_id: str = ""
        self.step_count: int = 0
        self.done: bool = False
        self._solver = StructuralSolver()
        self._prev_obs: Optional[StructuralObservation] = None
        self._current_obs: Optional[StructuralObservation] = None
        self._last_action_result: str = "RESET"
        self._last_action_error: Optional[str] = None
        self._consecutive_invalid: int = 0
        self._last_solver_result = None  # stored for /query_forces endpoint

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str = "task1_warehouse",
        seed: Optional[int] = None,
    ) -> dict:
        if seed is not None:
            np.random.seed(seed)

        if task_id not in TASK_REGISTRY:
            task_id = "task1_warehouse"

        self.task_config, _ = TASK_REGISTRY[task_id]
        tc = self.task_config

        self.graph = StructuralGraph(floor_height_m=tc.floor_height_m)
        self.grid = PlanGrid(n_floors=tc.n_floors)
        self.episode_id = str(uuid.uuid4())[:8]
        self.step_count = 0
        self.done = False
        self._last_action_result = "RESET"
        self._last_action_error = None
        self._consecutive_invalid = 0

        obs = self._build_observation()
        self._prev_obs = obs
        self._current_obs = obs
        return obs.model_dump()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, message: str) -> Tuple[dict, float, bool, dict]:
        """
        Process one agent action.

        Parameters
        ----------
        message : JSON string containing a StructuralAction

        Returns
        -------
        (observation_dict, reward, done, info)
        """
        if self.done:
            obs = self._current_obs or self._build_observation()
            return obs.model_dump(), 0.0, True, {"error": "Episode already done"}

        # Parse action
        try:
            raw = json.loads(message) if isinstance(message, str) else message
            action = StructuralAction(**raw)
        except Exception as e:
            self._last_action_result = "INVALID"
            self._last_action_error = f"Failed to parse action: {e}"
            self._consecutive_invalid += 1
            self.step_count += 1
            obs = self._build_observation()
            self._prev_obs = self._current_obs
            self._current_obs = obs
            done = (self.step_count >= self.task_config.max_steps
                    or self._consecutive_invalid >= 5)
            if done:
                self.done = True
            return obs.model_dump(), -0.10, done, {"error": str(e)}

        # done action
        if action.action_type == "done":
            self.done = True
            self._last_action_result = "PLACED"
            self._last_action_error = None
            self.step_count += 1
            obs = self._build_observation()
            prev = self._current_obs or obs
            self._prev_obs = prev
            self._current_obs = obs
            reward = compute_reward(prev, obs, action, self.task_config)
            # Grade the episode
            graded_score = self._grade()
            info = {
                "graded_score": graded_score,
                "step_count": self.step_count,
                "is_structurally_valid": obs.is_structurally_valid,
            }
            return obs.model_dump(), float(reward), True, info

        # Validate
        valid, err_msg = validate_action(action, self.graph, self.grid, self.task_config)
        if not valid:
            self._last_action_result = "INVALID"
            self._last_action_error = err_msg
            self._consecutive_invalid += 1
            self.step_count += 1
            obs = self._build_observation()
            prev = self._current_obs or obs
            self._prev_obs = prev
            self._current_obs = obs
            reward = compute_reward(prev, obs, action, self.task_config)
            done = (self.step_count >= self.task_config.max_steps
                    or self._consecutive_invalid >= 5)
            if done:
                self.done = True
            return obs.model_dump(), float(reward), done, {"validation_error": err_msg}

        # Apply action
        result_str = self._apply_action(action)
        self._last_action_result = result_str
        self._last_action_error = None
        self._consecutive_invalid = 0  # reset streak on valid action

        self.step_count += 1
        obs = self._build_observation()
        prev = self._current_obs or obs
        self._prev_obs = prev
        self._current_obs = obs
        reward = compute_reward(prev, obs, action, self.task_config)

        done = self.step_count >= self.task_config.max_steps
        if done:
            self.done = True

        info: Dict[str, Any] = {"step_count": self.step_count}
        if done:
            info["graded_score"] = self._grade()

        return obs.model_dump(), float(reward), done, info

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> dict:
        """Return episode state metadata."""
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_config.task_id if self.task_config else None,
            "step_count": self.step_count,
            "max_steps": self.task_config.max_steps if self.task_config else 0,
            "done": self.done,
            "n_elements": len(self.graph.elements) if self.graph else 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action: StructuralAction) -> str:
        """Apply a validated action to the graph and grid. Returns result string."""
        tc = self.task_config
        atype = action.action_type

        if atype == "place_column":
            gx, gy, floor = action.grid_x, action.grid_y, action.floor
            self.graph.place_column(gx, gy, floor, action.section)
            self.grid.place_column(gx, gy, floor)
            return "PLACED"

        if atype == "place_beam":
            fx, fy = action.from_node_x, action.from_node_y
            tx, ty = action.to_node_x, action.to_node_y
            floor = action.floor
            orient = action.orientation or ("x" if fy == ty else "y")
            self.graph.place_beam(fx, fy, tx, ty, floor, action.section, orient)
            self.grid.place_beam(fx, fy, tx, ty, floor, orient)
            return "PLACED"

        if atype == "upgrade_section":
            eid = action.element_id
            elem = self.graph.elements[eid]
            new_section = upgrade_section(elem.section)
            if new_section:
                elem.section = new_section
            return "UPDATED"

        if atype == "downgrade_section":
            eid = action.element_id
            elem = self.graph.elements[eid]
            new_section = downgrade_section(elem.section)
            if new_section:
                elem.section = new_section
            return "UPDATED"

        if atype == "remove_element":
            eid = action.element_id
            self.graph.remove_element(eid)
            # Clear grid cells where possible (best effort)
            return "REMOVED"

        if atype == "add_wall":
            fx, fy = action.from_node_x, action.from_node_y
            tx, ty = action.to_node_x, action.to_node_y
            floor = action.floor
            thickness = action.thickness_m or 0.2
            orient = action.orientation or ("x" if fy == ty else "y")
            self.graph.add_wall(fx, fy, tx, ty, floor, thickness, orient)
            self.grid.place_wall(fx, fy, tx, ty, floor)
            return "PLACED"

        return "UNKNOWN"

    def _build_observation(self) -> StructuralObservation:
        """Run solver, code checks, and build the full observation."""
        tc = self.task_config
        graph = self.graph

        # ---- Physics solve ----
        solver_result = None
        member_checks: Dict[str, Any] = {}
        lateral_drift_ratio = 0.0

        if len(graph.elements) >= 1 and len(graph.nodes) >= 2:
            loads = generate_loads(graph, tc)
            solver_result = self._solver.solve(graph, loads)
            self._last_solver_result = solver_result  # store for /query_forces

            if solver_result.converged:
                # Determine bracing: walls → braced frame → lower effective length
                is_braced = any(
                    e.element_type == "wall" for e in graph.elements.values()
                )
                # EC3 §6.3.1: k=0.7 braced (non-sway), k=1.5 unbraced (sway)
                L_eff_factor = 0.7 if is_braced else 1.5

                # Run code checks on all elements (steel + walls)
                for eid, elem in graph.elements.items():
                    forces = solver_result.member_forces.get(eid, {
                        "N": 0.0, "V": 0.0, "M_max": 0.0, "delta_max_mm": 0.0
                    })

                    if elem.element_type == "wall":
                        # RC shear wall check (EC2 simplified)
                        checks = check_member(
                            element_type="wall",
                            section_props={},
                            forces=forces,
                            L_m=elem.length_m,
                            floor_height_m=tc.floor_height_m,
                            thickness_m=elem.thickness_m or 0.2,
                        )
                    else:
                        props = COLUMN_SECTIONS.get(elem.section) or BEAM_SECTIONS.get(elem.section)
                        if props is None:
                            continue
                        checks = check_member(
                            element_type=elem.element_type,
                            section_props=props,
                            forces=forces,
                            L_m=elem.length_m,
                            L_eff_factor=L_eff_factor if elem.element_type == "column" else 1.0,
                        )
                    member_checks[eid] = checks

                # Lateral drift ratio: max story drift / story height
                lateral_drift_ratio = self._compute_lateral_drift(solver_result, tc)

        # ---- Build CriticalMember list ----
        critical_members = []
        for eid, checks in member_checks.items():
            elem = graph.elements.get(eid)
            if elem is None:
                continue
            forces = (
                solver_result.member_forces.get(eid, {})
                if solver_result and solver_result.converged
                else {}
            )
            cm = CriticalMember(
                id=eid,
                type=elem.element_type,
                section=elem.section,
                length_m=round(elem.length_m, 3),
                UR_bending=checks.UR_bending,
                UR_shear=checks.UR_shear,
                UR_buckling=checks.UR_buckling,
                UR_deflection=checks.UR_deflection,
                max_UR=checks.max_UR,
                N_Ed_kN=round(forces.get("N", 0.0) / 1000.0, 3),
                M_Ed_kNm=round(forces.get("M_max", 0.0) / 1000.0, 3),
                V_Ed_kN=round(forces.get("V", 0.0) / 1000.0, 3),
            )
            critical_members.append(cm)

        # Sort by max_UR descending for easy reading
        critical_members.sort(key=lambda m: -m.max_UR)

        # ---- Aggregate metrics ----
        max_UR_bending = max((c.UR_bending for c in critical_members), default=0.0)
        max_UR_buckling = max((c.UR_buckling for c in critical_members), default=0.0)
        max_UR_shear = max((c.UR_shear for c in critical_members), default=0.0)
        max_deflection_mm = max(
            (
                solver_result.member_forces.get(eid, {}).get("delta_max_mm", 0.0)
                for eid in member_checks
            ),
            default=0.0,
        ) if (solver_result and solver_result.converged) else 0.0

        n_code_violations = sum(
            1 for c in critical_members if not (c.max_UR <= 1.0)
        )
        is_structurally_valid = (
            n_code_violations == 0
            and len(critical_members) > 0
            and (solver_result.converged if solver_result else False)
        )

        # ---- Mass + carbon footprint ----
        total_mass_kg = graph.total_steel_mass_kg()
        ref_mass = max(tc.reference_mass_kg, 1.0)
        material_efficiency = max(0.0, 1.0 - abs(total_mass_kg - ref_mass) / ref_mass)
        # Steel basic oxygen furnace: ~1.85 kg CO2 per kg steel
        carbon_kg = round(total_mass_kg * 1.85, 1)

        # ---- Displacement hotspots (top-5 most displaced nodes) ----
        import math as _math
        displacement_hotspots = []
        if solver_result and solver_result.converged:
            hotspots_raw = []
            for nid, d in solver_result.node_displacements.items():
                mag_mm = _math.sqrt(d["ux"] ** 2 + d["uy"] ** 2 + d["uz"] ** 2) * 1000.0
                hotspots_raw.append({"node_id": nid, "displacement_mm": round(mag_mm, 3)})
            hotspots_raw.sort(key=lambda x: -x["displacement_mm"])
            displacement_hotspots = hotspots_raw[:5]

        # ---- Braced frame / effective length factor ----
        is_braced = any(e.element_type == "wall" for e in graph.elements.values())
        eff_length_factor = 0.7 if is_braced else (1.5 if graph.elements else 1.0)

        # ---- Grid plan (all floors) ----
        grid_plan = []
        for floor in range(tc.n_floors):
            grid_plan.append(self.grid.to_ascii_grid(floor))

        # ---- Placed elements summary ----
        placed_elements = []
        for eid, elem in graph.elements.items():
            entry = {
                "id": eid,
                "type": elem.element_type,
                "section": elem.section,
                "length_m": round(elem.length_m, 3),
                "node_i": elem.node_i,
                "node_j": elem.node_j,
            }
            if elem.orientation:
                entry["orientation"] = elem.orientation
            placed_elements.append(entry)

        # ---- Message ----
        msg = self._build_message(
            tc=tc,
            placed_elements=placed_elements,
            critical_members=critical_members,
            max_UR_bending=max_UR_bending,
            max_UR_buckling=max_UR_buckling,
            max_deflection_mm=max_deflection_mm,
            lateral_drift_ratio=lateral_drift_ratio,
            n_code_violations=n_code_violations,
            is_valid=is_structurally_valid,
            total_mass_kg=total_mass_kg,
            carbon_kg=carbon_kg,
            is_braced=is_braced,
            eff_length_factor=eff_length_factor,
            solver_converged=(solver_result.converged if solver_result else False),
            solver_error=(solver_result.error if solver_result else None),
        )

        obs = StructuralObservation(
            site_width_m=tc.site_width_m,
            site_depth_m=tc.site_depth_m,
            n_floors=tc.n_floors,
            floor_height_m=tc.floor_height_m,
            dead_load_kPa=tc.dead_load_kPa,
            live_load_kPa=tc.live_load_kPa,
            wind_load_kN_per_m=tc.wind_load_kN_per_m,
            seismic_ag_g=tc.seismic_ag_g,
            task_id=tc.task_id,
            grid_plan=grid_plan,
            placed_elements=placed_elements,
            n_elements_placed=len(placed_elements),
            critical_members=critical_members,
            max_UR_bending=round(max_UR_bending, 4),
            max_UR_buckling=round(max_UR_buckling, 4),
            max_UR_shear=round(max_UR_shear, 4),
            max_deflection_mm=round(max_deflection_mm, 4),
            max_lateral_drift_ratio=round(lateral_drift_ratio, 4),
            n_code_violations=n_code_violations,
            is_structurally_valid=is_structurally_valid,
            total_steel_mass_kg=round(total_mass_kg, 2),
            material_efficiency_score=round(material_efficiency, 4),
            carbon_kg=carbon_kg,
            is_braced_frame=is_braced,
            effective_length_factor=eff_length_factor,
            displacement_hotspots=displacement_hotspots,
            step_count=self.step_count,
            max_steps=tc.max_steps,
            last_action_error=self._last_action_error,
            last_action_result=self._last_action_result,
            episode_id=self.episode_id,
            message=msg,
        )
        return obs

    def _compute_lateral_drift(self, solver_result, tc: TaskConfig) -> float:
        """
        Compute maximum lateral drift ratio = max_story_drift / (h/500).

        Iterates over every column and computes the vector horizontal drift
        between its base and top nodes, capturing torsional effects that a
        floor-average approach would miss.
        """
        if not solver_result or not solver_result.converged:
            return 0.0

        import math as _math
        drift_limit = tc.floor_height_m / 500.0
        if drift_limit <= 0:
            return 0.0

        max_drift_ratio = 0.0
        nd = solver_result.node_displacements

        for eid, elem in self.graph.elements.items():
            if elem.element_type != "column":
                continue
            d_i = nd.get(elem.node_i, {})
            d_j = nd.get(elem.node_j, {})
            dx = d_j.get("ux", 0.0) - d_i.get("ux", 0.0)
            dy = d_j.get("uy", 0.0) - d_i.get("uy", 0.0)
            story_drift = _math.sqrt(dx * dx + dy * dy)
            ratio = story_drift / drift_limit
            if ratio > max_drift_ratio:
                max_drift_ratio = ratio

        return max_drift_ratio

    def _grade(self) -> float:
        """Compute final grade using task-specific grader."""
        if self._current_obs is None:
            return 0.0

        task_id = self.task_config.task_id
        if task_id == "task1_warehouse":
            from structural_design_env.tasks.task1_warehouse import grade_task1
            return grade_task1(self._current_obs)
        elif task_id == "task2_office":
            from structural_design_env.tasks.task2_office import grade_task2
            return grade_task2(self._current_obs)
        elif task_id == "task3_hospital":
            from structural_design_env.tasks.task3_hospital import grade_task3
            return grade_task3(self._current_obs, self.graph)
        return 0.0

    def _build_message(
        self,
        tc: TaskConfig,
        placed_elements: list,
        critical_members: list,
        max_UR_bending: float,
        max_UR_buckling: float,
        max_deflection_mm: float,
        lateral_drift_ratio: float,
        n_code_violations: int,
        is_valid: bool,
        total_mass_kg: float,
        carbon_kg: float,
        is_braced: bool,
        eff_length_factor: float,
        solver_converged: bool,
        solver_error: Optional[str],
    ) -> str:
        """Build a human/LLM-readable text summary of the current state."""
        lines = [
            f"=== {tc.name} (step {self.step_count}/{tc.max_steps}) ===",
            f"Site: {tc.site_width_m}m x {tc.site_depth_m}m, {tc.n_floors} floor(s), h={tc.floor_height_m}m",
            f"Loads: DL={tc.dead_load_kPa}kPa, LL={tc.live_load_kPa}kPa, "
            f"Wind={tc.wind_load_kN_per_m}kN/m, ag={tc.seismic_ag_g}g",
            "",
            f"Elements placed: {len(placed_elements)}",
        ]

        if not solver_converged:
            lines.append(
                f"SOLVER: Not converged ({solver_error or 'no elements or disconnected structure'})"
            )
        else:
            status = "VALID" if is_valid else f"INVALID ({n_code_violations} violations)"
            frame_type = f"BRACED (k={eff_length_factor})" if is_braced else f"UNBRACED/SWAY (k={eff_length_factor})"
            lines += [
                f"Structural status: {status}",
                f"Frame type: {frame_type}",
                f"Max UR bending: {max_UR_bending:.3f}  |  Max UR buckling: {max_UR_buckling:.3f}",
                f"Max deflection: {max_deflection_mm:.2f}mm (limit {tc.deflection_limit_mm:.1f}mm)",
                f"Max lateral drift ratio: {lateral_drift_ratio:.3f} (limit 1.0)",
                f"Total steel mass: {total_mass_kg:.0f} kg (reference {tc.reference_mass_kg:.0f} kg)",
                f"Carbon footprint: {carbon_kg:.0f} kg CO2",
            ]

        if critical_members:
            lines.append("")
            lines.append("Critical members (top by UR):")
            for cm in critical_members[:5]:
                lines.append(
                    f"  [{cm.id}] {cm.section} L={cm.length_m}m "
                    f"UR_bend={cm.UR_bending:.3f} UR_buck={cm.UR_buckling:.3f} "
                    f"UR_defl={cm.UR_deflection:.3f} max_UR={cm.max_UR:.3f}"
                )

        lines += [
            "",
            f"Last action: {self._last_action_result}"
            + (f" — {self._last_action_error}" if self._last_action_error else ""),
            "",
            "Available actions: place_column, place_beam, upgrade_section, "
            "downgrade_section, remove_element, add_wall, done",
            'Example: {"action_type": "place_column", "grid_x": 5, "grid_y": 0, "floor": 0, "section": "HEB200"}',
        ]

        return "\n".join(lines)
