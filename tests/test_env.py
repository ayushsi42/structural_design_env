"""
Integration tests for StructuralDesignEnv.

Covers:
- reset returns valid observation
- step with valid place_column action
- step with invalid action returns INVALID
- done action terminates episode
- full warehouse episode (place columns + beams + done)
- upgrade/downgrade section
- remove_element
- add_wall
- step count increments
- reward clipped to [-1, 1]
"""

import json

import pytest

from structural_design_env.env import StructuralDesignEnv
from structural_design_env.models import StructuralObservation


class TestReset:
    def test_reset_task1(self):
        env = StructuralDesignEnv()
        obs_dict = env.reset(task_id="task1_warehouse")
        assert "task_id" in obs_dict
        assert obs_dict["task_id"] == "task1_warehouse"
        assert obs_dict["step_count"] == 0
        assert obs_dict["n_elements_placed"] == 0
        assert obs_dict["last_action_result"] == "RESET"

    def test_reset_task2(self):
        env = StructuralDesignEnv()
        obs = env.reset(task_id="task2_office")
        assert obs["n_floors"] == 3
        assert obs["site_width_m"] == 20.0

    def test_reset_task3(self):
        env = StructuralDesignEnv()
        obs = env.reset(task_id="task3_hospital")
        assert obs["seismic_ag_g"] == 0.25
        # seismic_gamma_I is an internal config field, not in observation
        assert env.task_config.seismic_gamma_I == pytest.approx(1.5)

    def test_reset_invalid_task_falls_back(self):
        env = StructuralDesignEnv()
        obs = env.reset(task_id="nonexistent_task")
        assert obs["task_id"] == "task1_warehouse"

    def test_reset_clears_elements(self):
        env = StructuralDesignEnv()
        env.reset()
        action = json.dumps({"action_type": "place_column", "grid_x": 0, "grid_y": 0, "floor": 0, "section": "HEB200"})
        env.step(action)
        env.reset()
        assert env.step_count == 0
        assert len(env.graph.elements) == 0

    def test_observation_has_grid_plan(self):
        env = StructuralDesignEnv()
        obs = env.reset()
        gp = obs["grid_plan"]
        assert isinstance(gp, list)
        assert len(gp) == 1  # task1 has 1 floor
        assert len(gp[0]) == 20
        assert len(gp[0][0]) == 20


class TestPlaceColumn:
    def setup_method(self):
        self.env = StructuralDesignEnv()
        self.env.reset(task_id="task1_warehouse")

    def test_valid_place_column(self):
        action = json.dumps({
            "action_type": "place_column",
            "grid_x": 0, "grid_y": 0, "floor": 0,
            "section": "HEB200",
        })
        obs, reward, done, info = self.env.step(action)
        assert obs["last_action_result"] == "PLACED"
        assert obs["n_elements_placed"] == 1
        assert obs["step_count"] == 1

    def test_duplicate_column_invalid(self):
        action = json.dumps({"action_type": "place_column", "grid_x": 5, "grid_y": 5, "floor": 0, "section": "HEB200"})
        self.env.step(action)
        obs, _, _, _ = self.env.step(action)
        assert obs["last_action_result"] == "INVALID"

    def test_out_of_bounds_column_invalid(self):
        action = json.dumps({"action_type": "place_column", "grid_x": 25, "grid_y": 0, "floor": 0, "section": "HEB200"})
        obs, _, _, _ = self.env.step(action)
        assert obs["last_action_result"] == "INVALID"

    def test_wrong_floor_invalid(self):
        action = json.dumps({"action_type": "place_column", "grid_x": 0, "grid_y": 0, "floor": 5, "section": "HEB200"})
        obs, _, _, _ = self.env.step(action)
        assert obs["last_action_result"] == "INVALID"

    def test_invalid_section_invalid(self):
        action = json.dumps({"action_type": "place_column", "grid_x": 0, "grid_y": 0, "floor": 0, "section": "BADSTEEL"})
        obs, _, _, _ = self.env.step(action)
        assert obs["last_action_result"] == "INVALID"

    def test_reward_clipped(self):
        action = json.dumps({"action_type": "place_column", "grid_x": 0, "grid_y": 0, "floor": 0, "section": "HEB200"})
        _, reward, _, _ = self.env.step(action)
        assert -1.0 <= reward <= 1.0


class TestPlaceBeam:
    def setup_method(self):
        self.env = StructuralDesignEnv()
        self.env.reset(task_id="task1_warehouse")
        # Place two columns
        self.env.step(json.dumps({"action_type": "place_column", "grid_x": 0, "grid_y": 0, "floor": 0, "section": "HEB200"}))
        self.env.step(json.dumps({"action_type": "place_column", "grid_x": 5, "grid_y": 0, "floor": 0, "section": "HEB200"}))

    def test_valid_beam(self):
        action = json.dumps({
            "action_type": "place_beam",
            "from_node_x": 0, "from_node_y": 0,
            "to_node_x": 5, "to_node_y": 0,
            "floor": 0, "section": "IPE300", "orientation": "x",
        })
        obs, _, _, _ = self.env.step(action)
        assert obs["last_action_result"] == "PLACED"
        assert obs["n_elements_placed"] == 3

    def test_beam_without_column_invalid(self):
        action = json.dumps({
            "action_type": "place_beam",
            "from_node_x": 0, "from_node_y": 0,
            "to_node_x": 10, "to_node_y": 0,
            "floor": 0, "section": "IPE300", "orientation": "x",
        })
        obs, _, _, _ = self.env.step(action)
        assert obs["last_action_result"] == "INVALID"

    def test_diagonal_beam_invalid(self):
        self.env.step(json.dumps({"action_type": "place_column", "grid_x": 5, "grid_y": 5, "floor": 0, "section": "HEB200"}))
        action = json.dumps({
            "action_type": "place_beam",
            "from_node_x": 0, "from_node_y": 0,
            "to_node_x": 5, "to_node_y": 5,
            "floor": 0, "section": "IPE300", "orientation": "x",
        })
        obs, _, _, _ = self.env.step(action)
        assert obs["last_action_result"] == "INVALID"

    def test_duplicate_beam_invalid(self):
        action = json.dumps({
            "action_type": "place_beam",
            "from_node_x": 0, "from_node_y": 0,
            "to_node_x": 5, "to_node_y": 0,
            "floor": 0, "section": "IPE300", "orientation": "x",
        })
        self.env.step(action)
        obs, _, _, _ = self.env.step(action)
        assert obs["last_action_result"] == "INVALID"


class TestUpgradeDowngrade:
    def setup_method(self):
        self.env = StructuralDesignEnv()
        self.env.reset(task_id="task1_warehouse")
        self.env.step(json.dumps({"action_type": "place_column", "grid_x": 0, "grid_y": 0, "floor": 0, "section": "HEB200"}))

    def test_upgrade_column(self):
        obs, _, _, _ = self.env.step(json.dumps({"action_type": "upgrade_section", "element_id": "col_0_0_0"}))
        assert obs["last_action_result"] == "UPDATED"
        # Section should be HEB240 now
        elem = self.env.graph.elements.get("col_0_0_0")
        assert elem.section == "HEB240"

    def test_downgrade_column(self):
        obs, _, _, _ = self.env.step(json.dumps({"action_type": "downgrade_section", "element_id": "col_0_0_0"}))
        assert obs["last_action_result"] == "UPDATED"
        elem = self.env.graph.elements.get("col_0_0_0")
        assert elem.section == "HEB160"

    def test_upgrade_at_max_invalid(self):
        for _ in range(10):  # upgrade many times to reach max
            self.env.step(json.dumps({"action_type": "upgrade_section", "element_id": "col_0_0_0"}))
        elem = self.env.graph.elements.get("col_0_0_0")
        assert elem.section == "HEB400"
        obs, _, _, _ = self.env.step(json.dumps({"action_type": "upgrade_section", "element_id": "col_0_0_0"}))
        assert obs["last_action_result"] == "INVALID"

    def test_downgrade_nonexistent_invalid(self):
        obs, _, _, _ = self.env.step(json.dumps({"action_type": "downgrade_section", "element_id": "col_99_99_0"}))
        assert obs["last_action_result"] == "INVALID"


class TestRemoveElement:
    def setup_method(self):
        self.env = StructuralDesignEnv()
        self.env.reset(task_id="task1_warehouse")
        self.env.step(json.dumps({"action_type": "place_column", "grid_x": 0, "grid_y": 0, "floor": 0, "section": "HEB200"}))

    def test_remove_existing(self):
        obs, _, _, _ = self.env.step(json.dumps({"action_type": "remove_element", "element_id": "col_0_0_0"}))
        assert obs["last_action_result"] == "REMOVED"
        assert obs["n_elements_placed"] == 0

    def test_remove_nonexistent_invalid(self):
        obs, _, _, _ = self.env.step(json.dumps({"action_type": "remove_element", "element_id": "col_99_99_0"}))
        assert obs["last_action_result"] == "INVALID"


class TestAddWall:
    def setup_method(self):
        self.env = StructuralDesignEnv()
        self.env.reset(task_id="task2_office")
        # Place two columns
        self.env.step(json.dumps({"action_type": "place_column", "grid_x": 0, "grid_y": 0, "floor": 0, "section": "HEB300"}))
        self.env.step(json.dumps({"action_type": "place_column", "grid_x": 5, "grid_y": 0, "floor": 0, "section": "HEB300"}))

    def test_add_wall_valid(self):
        action = json.dumps({
            "action_type": "add_wall",
            "from_node_x": 0, "from_node_y": 0,
            "to_node_x": 5, "to_node_y": 0,
            "floor": 0, "thickness_m": 0.2, "orientation": "x",
        })
        obs, _, _, _ = self.env.step(action)
        assert obs["last_action_result"] == "PLACED"

    def test_add_wall_no_column_invalid(self):
        action = json.dumps({
            "action_type": "add_wall",
            "from_node_x": 0, "from_node_y": 0,
            "to_node_x": 10, "to_node_y": 0,
            "floor": 0, "thickness_m": 0.2, "orientation": "x",
        })
        obs, _, _, _ = self.env.step(action)
        assert obs["last_action_result"] == "INVALID"


class TestDoneAction:
    def test_done_terminates(self):
        env = StructuralDesignEnv()
        env.reset(task_id="task1_warehouse")
        obs, reward, done, info = env.step(json.dumps({"action_type": "done"}))
        assert done is True
        assert "graded_score" in info

    def test_done_invalid_design_penalty(self):
        env = StructuralDesignEnv()
        env.reset(task_id="task1_warehouse")
        _, reward, done, _ = env.step(json.dumps({"action_type": "done"}))
        assert done
        # No elements placed → invalid → penalty
        assert reward < 0

    def test_done_after_done_noop(self):
        env = StructuralDesignEnv()
        env.reset(task_id="task1_warehouse")
        env.step(json.dumps({"action_type": "done"}))
        obs, reward, done2, _ = env.step(json.dumps({"action_type": "done"}))
        assert done2 is True
        assert reward == 0.0


class TestFullWarehouseEpisode:
    """A minimal but complete episode: place columns, beam, done."""

    def test_complete_episode(self):
        env = StructuralDesignEnv()
        env.reset(task_id="task1_warehouse")

        # Place 2 columns
        env.step(json.dumps({"action_type": "place_column", "grid_x": 0, "grid_y": 0, "floor": 0, "section": "HEB200"}))
        env.step(json.dumps({"action_type": "place_column", "grid_x": 5, "grid_y": 0, "floor": 0, "section": "HEB200"}))

        # Place beam
        env.step(json.dumps({
            "action_type": "place_beam",
            "from_node_x": 0, "from_node_y": 0,
            "to_node_x": 5, "to_node_y": 0,
            "floor": 0, "section": "IPE300", "orientation": "x",
        }))

        # Done
        obs, reward, done, info = env.step(json.dumps({"action_type": "done"}))

        assert done
        assert obs["n_elements_placed"] == 3
        assert obs["step_count"] == 4
        assert "graded_score" in info
        assert 0.0 <= info["graded_score"] <= 1.0

    def test_physics_runs_after_elements_placed(self):
        env = StructuralDesignEnv()
        env.reset(task_id="task1_warehouse")

        # Place enough structure for solver to run
        env.step(json.dumps({"action_type": "place_column", "grid_x": 0, "grid_y": 0, "floor": 0, "section": "HEB240"}))
        env.step(json.dumps({"action_type": "place_column", "grid_x": 5, "grid_y": 0, "floor": 0, "section": "HEB240"}))
        obs, _, _, _ = env.step(json.dumps({
            "action_type": "place_beam",
            "from_node_x": 0, "from_node_y": 0,
            "to_node_x": 5, "to_node_y": 0,
            "floor": 0, "section": "IPE360", "orientation": "x",
        }))

        # With a real structure, physics should provide UR values
        assert isinstance(obs["max_UR_bending"], float)
        assert isinstance(obs["total_steel_mass_kg"], float)
        assert obs["total_steel_mass_kg"] > 0


class TestObservationSchema:
    """Ensure observation dict is valid StructuralObservation."""

    def test_obs_parses_as_model(self):
        env = StructuralDesignEnv()
        obs_dict = env.reset(task_id="task1_warehouse")
        # Should not raise
        obs = StructuralObservation(**obs_dict)
        assert obs.task_id == "task1_warehouse"

    def test_message_field_present(self):
        env = StructuralDesignEnv()
        obs = env.reset()
        assert "message" in obs
        assert len(obs["message"]) > 10

    def test_step_obs_parses(self):
        env = StructuralDesignEnv()
        env.reset()
        obs_dict, _, _, _ = env.step(json.dumps({"action_type": "place_column", "grid_x": 3, "grid_y": 3, "floor": 0, "section": "HEB200"}))
        obs = StructuralObservation(**obs_dict)
        assert obs.n_elements_placed == 1
