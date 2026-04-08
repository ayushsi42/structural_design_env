"""
Tests for task graders.

Covers:
- task1 grade when invalid → partial credit
- task1 grade when valid + efficient → high score
- task2 grade with/without walls
- task3 grade with high seismic drift → reduced score
"""

import pytest

from structural_design_env.models import CriticalMember, StructuralObservation, TaskConfig
from structural_design_env.tasks.task1_warehouse import grade_task1, TASK1_CONFIG
from structural_design_env.tasks.task2_office import grade_task2, TASK2_CONFIG
from structural_design_env.tasks.task3_hospital import grade_task3, TASK3_CONFIG


def _make_obs(
    task_id="task1_warehouse",
    is_valid=True,
    n_violations=0,
    mass_kg=520.0,
    avg_ur=0.75,
    drift_ratio=0.5,
    has_wall_x=False,
    has_wall_y=False,
    n_floors=1,
):
    """Helper: build a minimal StructuralObservation for grader testing."""
    walls = []
    if has_wall_x:
        walls.append({"type": "wall", "orientation": "x"})
    if has_wall_y:
        walls.append({"type": "wall", "orientation": "y"})

    placed = [{"type": "column"}, {"type": "beam"}] + walls

    cm = CriticalMember(
        id="col_0_0_0",
        type="column",
        section="HEB200",
        length_m=4.0,
        UR_bending=avg_ur * 0.8,
        UR_shear=avg_ur * 0.3,
        UR_buckling=avg_ur,
        UR_deflection=0.0,
        max_UR=avg_ur,
        N_Ed_kN=-200.0,
        M_Ed_kNm=10.0,
        V_Ed_kN=5.0,
    )

    # grid_plan: List[floors][rows][cols] = List[List[List[str]]]
    grid_plan = [[["." for _ in range(20)] for _ in range(20)] for _ in range(max(n_floors, 1))]

    tc = (
        TASK1_CONFIG if task_id == "task1_warehouse"
        else TASK2_CONFIG if task_id == "task2_office"
        else TASK3_CONFIG
    )

    return StructuralObservation(
        site_width_m=tc.site_width_m,
        site_depth_m=tc.site_depth_m,
        n_floors=tc.n_floors,
        floor_height_m=tc.floor_height_m,
        dead_load_kPa=tc.dead_load_kPa,
        live_load_kPa=tc.live_load_kPa,
        wind_load_kN_per_m=tc.wind_load_kN_per_m,
        seismic_ag_g=tc.seismic_ag_g,
        task_id=task_id,
        grid_plan=grid_plan,
        placed_elements=placed,
        n_elements_placed=len(placed),
        critical_members=[cm],
        max_UR_bending=avg_ur * 0.8,
        max_UR_buckling=avg_ur,
        max_UR_shear=avg_ur * 0.3,
        max_deflection_mm=5.0,
        max_lateral_drift_ratio=drift_ratio,
        n_code_violations=n_violations,
        is_structurally_valid=is_valid,
        total_steel_mass_kg=mass_kg,
        material_efficiency_score=0.8,
        step_count=10,
        max_steps=tc.max_steps,
        last_action_result="PLACED",
        episode_id="test-ep",
        message="test",
    )


class TestTask1Grader:
    def test_invalid_gives_partial(self):
        obs = _make_obs(task_id="task1_warehouse", is_valid=False, n_violations=5)
        score = grade_task1(obs)
        assert 0.0 <= score < 0.2, f"Expected low partial score, got {score}"

    def test_zero_violations_gives_zero_partial(self):
        obs = _make_obs(task_id="task1_warehouse", is_valid=False, n_violations=0)
        score = grade_task1(obs)
        # 0 violations: (1 - 0/10) * 0.2 = 0.2
        assert score == pytest.approx(0.2, rel=0.01)

    def test_valid_efficient_design_high_score(self):
        obs = _make_obs(
            task_id="task1_warehouse",
            is_valid=True,
            n_violations=0,
            mass_kg=520.0,  # exactly reference
            avg_ur=0.77,    # sweet spot
        )
        score = grade_task1(obs)
        assert score >= 0.7, f"Expected high score, got {score}"

    def test_valid_overdesigned_reduced_score(self):
        obs_good = _make_obs(task_id="task1_warehouse", is_valid=True, mass_kg=520.0, avg_ur=0.77)
        obs_heavy = _make_obs(task_id="task1_warehouse", is_valid=True, mass_kg=1520.0, avg_ur=0.77)
        assert grade_task1(obs_heavy) < grade_task1(obs_good)

    def test_score_in_range(self):
        for mass in [400, 520, 800, 1500]:
            for ur in [0.3, 0.7, 0.85, 0.95]:
                obs = _make_obs(task_id="task1_warehouse", is_valid=True, mass_kg=mass, avg_ur=ur)
                score = grade_task1(obs)
                assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] for mass={mass}, ur={ur}"


class TestTask2Grader:
    def test_invalid_gives_low_score(self):
        obs = _make_obs(task_id="task2_office", is_valid=False, n_violations=10, n_floors=3)
        score = grade_task2(obs)
        assert score < 0.15

    def test_valid_no_walls_reduced_score(self):
        obs = _make_obs(
            task_id="task2_office", is_valid=True,
            mass_kg=3200.0, drift_ratio=0.5,
            has_wall_x=False, has_wall_y=False,
            n_floors=3,
        )
        score_no_walls = grade_task2(obs)

        obs_walls = _make_obs(
            task_id="task2_office", is_valid=True,
            mass_kg=3200.0, drift_ratio=0.5,
            has_wall_x=True, has_wall_y=True,
            n_floors=3,
        )
        score_walls = grade_task2(obs_walls)
        assert score_walls > score_no_walls

    def test_high_drift_reduces_score(self):
        obs_low = _make_obs(task_id="task2_office", is_valid=True, drift_ratio=0.3, n_floors=3)
        obs_high = _make_obs(task_id="task2_office", is_valid=True, drift_ratio=1.5, n_floors=3)
        assert grade_task2(obs_low) > grade_task2(obs_high)

    def test_score_range(self):
        obs = _make_obs(task_id="task2_office", is_valid=True, n_floors=3)
        score = grade_task2(obs)
        assert 0.0 <= score <= 1.0


class TestTask3Grader:
    def test_invalid_minimal_score(self):
        obs = _make_obs(task_id="task3_hospital", is_valid=False, n_violations=40, n_floors=3)
        score = grade_task3(obs, graph=None)
        assert score <= 0.05

    def test_valid_no_redundancy_lower_score(self):
        obs = _make_obs(task_id="task3_hospital", is_valid=True, mass_kg=14000.0, drift_ratio=0.5, n_floors=3)
        # Without graph, redundancy_score = 0
        score = grade_task3(obs, graph=None)
        # Max without redundancy (0.25 contribution) = 0.75
        assert score <= 0.76

    def test_high_seismic_drift_reduces_score(self):
        obs_ok = _make_obs(task_id="task3_hospital", is_valid=True, drift_ratio=0.8, n_floors=3)
        obs_bad = _make_obs(task_id="task3_hospital", is_valid=True, drift_ratio=2.0, n_floors=3)
        assert grade_task3(obs_ok, None) > grade_task3(obs_bad, None)

    def test_budget_exceeded_reduces_score(self):
        obs_ok = _make_obs(task_id="task3_hospital", is_valid=True, mass_kg=14000.0, n_floors=3)
        obs_over = _make_obs(task_id="task3_hospital", is_valid=True, mass_kg=30000.0, n_floors=3)
        assert grade_task3(obs_ok, None) >= grade_task3(obs_over, None)

    def test_score_in_range(self):
        obs = _make_obs(task_id="task3_hospital", is_valid=True, n_floors=3)
        score = grade_task3(obs, None)
        assert 0.0 <= score <= 1.0
