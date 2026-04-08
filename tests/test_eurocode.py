"""
Tests for Eurocode 3 code checks (eurocode3.py).

Covers:
- Bending utilisation ratio
- Shear utilisation ratio
- Buckling reduction factor
- Interaction formula
- Deflection check
- Wall elements pass trivially
"""

import math

import pytest

from structural_design_env.solver.eurocode3 import check_member, MemberChecks
from structural_design_env.solver.sections import (
    COLUMN_SECTIONS,
    BEAM_SECTIONS,
    F_Y_STEEL,
    E_STEEL,
)


class TestBendingCheck:
    def test_zero_moment_gives_zero_ur(self):
        props = COLUMN_SECTIONS["HEB200"]
        checks = check_member("column", props, {"N": 0, "V": 0, "M_max": 0, "delta_max_mm": 0}, L_m=3.5)
        assert checks.UR_bending == 0.0

    def test_at_yield_gives_ur_one(self):
        props = BEAM_SECTIONS["IPE300"]
        # M at yield = W_el_y * f_y
        M_yield = props["W_el_y"] * F_Y_STEEL
        checks = check_member("beam", props, {"N": 0, "V": 0, "M_max": M_yield, "delta_max_mm": 0}, L_m=5.0)
        assert checks.UR_bending == pytest.approx(1.0, rel=0.01)

    def test_overstressed_ur_above_one(self):
        props = BEAM_SECTIONS["IPE200"]
        M_yield = props["W_el_y"] * F_Y_STEEL
        checks = check_member("beam", props, {"N": 0, "V": 0, "M_max": 2.0 * M_yield, "delta_max_mm": 0}, L_m=5.0)
        assert checks.UR_bending > 1.0
        assert not checks.passes_all


class TestShearCheck:
    def test_zero_shear_gives_zero_ur(self):
        props = BEAM_SECTIONS["IPE360"]
        checks = check_member("beam", props, {"N": 0, "V": 0, "M_max": 0, "delta_max_mm": 0}, L_m=4.0)
        assert checks.UR_shear == 0.0

    def test_shear_at_capacity(self):
        props = BEAM_SECTIONS["IPE300"]
        V_Rd = F_Y_STEEL * props["A_v"] / math.sqrt(3)
        checks = check_member("beam", props, {"N": 0, "V": V_Rd, "M_max": 0, "delta_max_mm": 0}, L_m=5.0)
        assert checks.UR_shear == pytest.approx(1.0, rel=0.01)


class TestBucklingCheck:
    def test_columns_have_buckling_check(self):
        props = COLUMN_SECTIONS["HEB200"]
        # Apply significant compressive load
        N_Ed = 500e3  # 500 kN
        checks = check_member(
            "column", props,
            {"N": -N_Ed, "V": 0, "M_max": 0, "delta_max_mm": 0},
            L_m=3.5,
        )
        assert checks.UR_buckling > 0.0

    def test_beams_have_zero_buckling(self):
        props = BEAM_SECTIONS["IPE400"]
        checks = check_member(
            "beam", props,
            {"N": -100e3, "V": 50e3, "M_max": 50e3, "delta_max_mm": 5.0},
            L_m=6.0,
        )
        assert checks.UR_buckling == 0.0

    def test_short_column_chi_near_one(self):
        """Very short column should have chi close to 1.0."""
        props = COLUMN_SECTIONS["HEB300"]
        A = props["A"]
        # Squash load
        N_b_Rd_max = A * F_Y_STEEL
        # Apply small load → UR_buckling should be small
        checks = check_member(
            "column", props,
            {"N": -N_b_Rd_max * 0.1, "V": 0, "M_max": 0, "delta_max_mm": 0},
            L_m=1.0,  # very short
        )
        assert checks.UR_buckling < 0.15

    def test_slender_column_higher_buckling_ur(self):
        props = COLUMN_SECTIONS["HEB140"]
        A = props["A"]
        N_Ed = A * F_Y_STEEL * 0.5  # 50% of squash load
        checks_short = check_member(
            "column", props,
            {"N": -N_Ed, "V": 0, "M_max": 0, "delta_max_mm": 0},
            L_m=2.0,
        )
        checks_tall = check_member(
            "column", props,
            {"N": -N_Ed, "V": 0, "M_max": 0, "delta_max_mm": 0},
            L_m=8.0,
        )
        assert checks_tall.UR_buckling > checks_short.UR_buckling


class TestDeflectionCheck:
    def test_zero_deflection_gives_zero_ur(self):
        props = BEAM_SECTIONS["IPE300"]
        checks = check_member("beam", props, {"N": 0, "V": 0, "M_max": 0, "delta_max_mm": 0}, L_m=5.0)
        assert checks.UR_deflection == 0.0

    def test_at_limit_gives_ur_one(self):
        props = BEAM_SECTIONS["IPE400"]
        L = 6.0  # m
        defl_limit_mm = L * 1000 / 300.0  # = 20.0 mm
        checks = check_member(
            "beam", props,
            {"N": 0, "V": 0, "M_max": 0, "delta_max_mm": defl_limit_mm},
            L_m=L,
        )
        assert checks.UR_deflection == pytest.approx(1.0, rel=0.01)

    def test_columns_have_zero_deflection_ur(self):
        props = COLUMN_SECTIONS["HEB240"]
        checks = check_member(
            "column", props,
            {"N": -300e3, "V": 0, "M_max": 0, "delta_max_mm": 50.0},
            L_m=3.5,
        )
        assert checks.UR_deflection == 0.0


class TestWallCheck:
    def test_wall_always_passes(self):
        checks = check_member(
            "wall", {},
            {"N": 0, "V": 0, "M_max": 0, "delta_max_mm": 0},
            L_m=5.0,
        )
        assert checks.passes_all
        assert checks.max_UR == 0.0


class TestInteraction:
    def test_combined_axial_bending_higher_ur(self):
        props = COLUMN_SECTIONS["HEB200"]
        A = props["A"]
        W_pl = props["W_pl_y"]

        # Pure bending at 50% capacity
        M_50 = 0.5 * W_pl * F_Y_STEEL
        checks_M_only = check_member(
            "column", props,
            {"N": 0, "V": 0, "M_max": M_50, "delta_max_mm": 0},
            L_m=3.5,
        )

        # Add axial load on top
        N_Ed = 0.3 * A * F_Y_STEEL
        checks_combined = check_member(
            "column", props,
            {"N": -N_Ed, "V": 0, "M_max": M_50, "delta_max_mm": 0},
            L_m=3.5,
        )

        assert checks_combined.UR_interaction >= checks_M_only.UR_interaction


class TestMaxUR:
    def test_max_ur_is_maximum_of_all_checks(self):
        props = BEAM_SECTIONS["IPE200"]
        L = 3.0
        M_80pct = 0.80 * props["W_el_y"] * F_Y_STEEL
        defl_50pct = L * 1000 / 300.0 * 0.5

        checks = check_member(
            "beam", props,
            {"N": 0, "V": 0, "M_max": M_80pct, "delta_max_mm": defl_50pct},
            L_m=L,
        )
        expected_max = max(
            checks.UR_bending,
            checks.UR_shear,
            checks.UR_buckling,
            checks.UR_interaction,
            checks.UR_deflection,
        )
        assert checks.max_UR == pytest.approx(expected_max, rel=0.001)
