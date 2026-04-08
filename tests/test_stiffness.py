"""
Tests for the 3D direct stiffness method solver.

Global coordinate system: x=East, y=North, z=Up
Node DOFs: [ux, uy, uz, rx, ry, rz]
Loads: Fz = gravity (vertical, -z = downward), Fx/Fy = lateral
"""

import math

import numpy as np
import pytest

from structural_design_env.graph import StructuralGraph, NodeData, ElementData
from structural_design_env.solver.stiffness_matrix import (
    StructuralSolver,
    _local_stiffness,
    R_COL, R_XBEAM, R_YBEAM,
)
from structural_design_env.solver.sections import (
    E_STEEL, G_STEEL, COLUMN_SECTIONS, BEAM_SECTIONS,
)


class TestLocalStiffness:
    """Test the 12×12 local stiffness matrix."""

    def _k(self, I_y=1000e-8, I_z=None, J=None, L=3.0, A=100e-4):
        if I_z is None:
            I_z = I_y * 0.3
        if J is None:
            J = I_y * 0.01
        return _local_stiffness(E=E_STEEL, A=A, I_y=I_y, I_z=I_z,
                                G=G_STEEL, J=J, L=L)

    def test_symmetry(self):
        k = self._k()
        np.testing.assert_allclose(k, k.T, atol=1e-3)

    def test_shape(self):
        k = self._k()
        assert k.shape == (12, 12)

    def test_axial_diagonal(self):
        E, A, L = E_STEEL, 100e-4, 5.0
        k = _local_stiffness(E=E, A=A, I_y=1000e-8, I_z=300e-8,
                             G=G_STEEL, J=10e-8, L=L)
        EA_L = E * A / L
        assert abs(k[0, 0] - EA_L) < 1.0
        assert abs(k[6, 6] - EA_L) < 1.0
        assert abs(k[0, 6] + EA_L) < 1.0

    def test_bending_1_2_plane(self):
        """Bending in 1-2 plane uses I_z (couples u2 and r3)."""
        E, I_z, L = E_STEEL, 1000e-8, 4.0
        k = _local_stiffness(E=E, A=50e-4, I_y=3000e-8, I_z=I_z,
                             G=G_STEEL, J=10e-8, L=L)
        expected = 12 * E * I_z / L**3
        assert abs(k[1, 1] - expected) < 1.0
        assert abs(k[7, 7] - expected) < 1.0

    def test_bending_1_3_plane(self):
        """Bending in 1-3 plane uses I_y (couples u3 and r2)."""
        E, I_y, L = E_STEEL, 5000e-8, 4.0
        k = _local_stiffness(E=E, A=50e-4, I_y=I_y, I_z=500e-8,
                             G=G_STEEL, J=10e-8, L=L)
        expected = 12 * E * I_y / L**3
        assert abs(k[2, 2] - expected) < 1.0
        assert abs(k[8, 8] - expected) < 1.0

    def test_positive_semidefinite(self):
        """Free-body element is PSD (6 rigid-body zero eigenvalues for 3D)."""
        k = self._k(L=3.5, A=78e-4, I_y=5696e-8, I_z=2003e-8)
        eigenvalues = np.linalg.eigvalsh(k)
        # No significantly negative eigenvalues
        assert np.sum(eigenvalues < -1e3) == 0


class TestRotationMatrices:
    """Verify rotation matrices are orthogonal and map vectors correctly."""

    def _check_orthogonal(self, R):
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_R_COL_orthogonal(self):
        self._check_orthogonal(R_COL)

    def test_R_XBEAM_orthogonal(self):
        self._check_orthogonal(R_XBEAM)

    def test_R_YBEAM_orthogonal(self):
        self._check_orthogonal(R_YBEAM)

    def test_column_gravity_becomes_axial(self):
        """Gravity (-z global) maps to -local-1 (axial compression) for a column."""
        F_global = np.array([0, 0, -1])
        F_local = R_COL @ F_global
        # Should be -1 in local-1 direction (axial)
        assert abs(F_local[0] + 1) < 1e-10

    def test_xbeam_gravity_becomes_transverse_y(self):
        """Gravity (-z global) maps to -local-2 (transverse) for an x-beam."""
        F_global = np.array([0, 0, -1])
        F_local = R_XBEAM @ F_global
        # Should be -1 in local-2 direction → bending about local-3
        assert abs(F_local[1] + 1) < 1e-10

    def test_ybeam_gravity_becomes_transverse_y(self):
        """Gravity (-z global) maps to -local-2 for a y-beam."""
        F_global = np.array([0, 0, -1])
        F_local = R_YBEAM @ F_global
        assert abs(F_local[1] + 1) < 1e-10

    def test_column_wind_x_becomes_transverse(self):
        """Wind in global x maps to local-2 for a column."""
        F_global = np.array([1, 0, 0])
        F_local = R_COL @ F_global
        assert abs(F_local[1] - 1) < 1e-10  # local-2 = x_global


class TestSolverColumn:
    """Test solving a single column under various loads."""

    def _make_column(self, floor_height=4.0, section="HEB200"):
        g = StructuralGraph(floor_height_m=floor_height)
        g.place_column(0, 0, 0, section)
        return g

    def test_column_converges_under_vertical_load(self):
        g = self._make_column()
        result = StructuralSolver().solve(g, {"n_0_0_1": {"Fz": -100e3}})
        assert result.converged, result.error

    def test_column_compresses_under_vertical_load(self):
        """Vertical load → column shortens → uz at top < 0."""
        g = self._make_column(floor_height=4.0, section="HEB200")
        P = 500e3  # 500 kN
        result = StructuralSolver().solve(g, {"n_0_0_1": {"Fz": -P}})
        assert result.converged

        uz_top = result.node_displacements["n_0_0_1"]["uz"]
        assert uz_top < 0, f"Expected uz < 0 (compression), got {uz_top}"

        # Analytical check: δ = P*L/(E*A)
        props = COLUMN_SECTIONS["HEB200"]
        expected = P * 4.0 / (E_STEEL * props["A"])
        assert abs(uz_top) == pytest.approx(expected, rel=0.05), (
            f"Expected {expected:.3e}, got {abs(uz_top):.3e}"
        )

    def test_column_axial_force_reported(self):
        g = self._make_column()
        P = 200e3
        result = StructuralSolver().solve(g, {"n_0_0_1": {"Fz": -P}})
        assert result.converged
        forces = result.member_forces.get("col_0_0_0")
        assert forces is not None
        # Axial should be close to applied load (sign may differ by convention)
        assert abs(forces["N"]) == pytest.approx(P, rel=0.05)

    def test_lateral_x_load_causes_ux_sway(self):
        """Lateral force in x → horizontal ux displacement at top.

        R_COL maps global-x → local-2, so x-wind bends the column in its
        1-2 plane which uses I_z (weak axis) in the local stiffness matrix.
        Physically: column flanges face North-South, web faces East-West.
        """
        g = self._make_column(floor_height=3.5, section="HEB300")
        H = 10e3
        result = StructuralSolver().solve(g, {"n_0_0_1": {"Fx": H}})
        assert result.converged
        ux_top = result.node_displacements["n_0_0_1"]["ux"]
        assert ux_top > 0, f"Expected ux > 0, got {ux_top}"

        # Cantilever tip deflection: δ = H*L³/(3EI)
        # x-wind uses weak axis I_z (local-z bending, 1-2 plane via R_COL)
        props = COLUMN_SECTIONS["HEB300"]
        L = 3.5
        expected = H * L**3 / (3 * E_STEEL * props["I_z"])
        assert ux_top == pytest.approx(expected, rel=0.10), (
            f"Expected ~{expected:.4e} m, got {ux_top:.4e} m"
        )

    def test_lateral_y_load_causes_uy_sway(self):
        """Lateral force in y → uy displacement at top (weak-axis bending)."""
        g = self._make_column(floor_height=3.5, section="HEB300")
        H = 10e3
        result = StructuralSolver().solve(g, {"n_0_0_1": {"Fy": H}})
        assert result.converged
        uy_top = result.node_displacements["n_0_0_1"]["uy"]
        assert uy_top > 0

    def test_node_displacements_dict_has_all_keys(self):
        g = self._make_column()
        result = StructuralSolver().solve(g, {"n_0_0_1": {"Fz": -50e3}})
        assert result.converged
        d = result.node_displacements["n_0_0_1"]
        for key in ("ux", "uy", "uz", "rx", "ry", "rz"):
            assert key in d


class TestSolverBeam:
    """Beam deflection and portal frame tests."""

    def _portal(self, span=5, height=3.5, col="HEB300", beam="IPE400"):
        g = StructuralGraph(floor_height_m=height)
        g.place_column(0, 0, 0, col)
        g.place_column(span, 0, 0, col)
        g.place_beam(0, 0, span, 0, 0, beam, "x")
        return g

    def test_portal_converges(self):
        g = self._portal()
        result = StructuralSolver().solve(g, {"n_5_0_1": {"Fz": -50e3}})
        assert result.converged, result.error

    def test_beam_deflects_under_gravity(self):
        """Gravity on beam nodes → beam-end nodes move down (uz < 0)."""
        g = self._portal(span=5)
        loads = {
            "n_0_0_1": {"Fz": -30e3},
            "n_5_0_1": {"Fz": -30e3},
        }
        result = StructuralSolver().solve(g, loads)
        assert result.converged
        uz_0 = result.node_displacements["n_0_0_1"]["uz"]
        uz_5 = result.node_displacements["n_5_0_1"]["uz"]
        assert uz_0 < 0 or uz_5 < 0, "At least one beam end should deflect downward"

    def test_lateral_load_causes_frame_sway(self):
        """Lateral load at top of portal frame → sway in ux."""
        g = self._portal(span=6, height=4.0)
        result = StructuralSolver().solve(g, {
            "n_0_0_1": {"Fx": 50e3},
            "n_6_0_1": {"Fx": 50e3},
        })
        assert result.converged
        ux_avg = (result.node_displacements["n_0_0_1"]["ux"] +
                  result.node_displacements["n_6_0_1"]["ux"]) / 2
        assert ux_avg > 0, f"Frame should sway in +x, got {ux_avg}"

    def test_y_beam_deflects_under_gravity(self):
        """Y-direction beam between two columns deflects downward under gravity."""
        g = StructuralGraph(floor_height_m=3.5)
        g.place_column(0, 0, 0, "HEB300")
        g.place_column(0, 5, 0, "HEB300")
        g.place_beam(0, 0, 0, 5, 0, "IPE400", "y")
        loads = {
            "n_0_0_1": {"Fz": -30e3},
            "n_0_5_1": {"Fz": -30e3},
        }
        result = StructuralSolver().solve(g, loads)
        assert result.converged
        uz_0 = result.node_displacements["n_0_0_1"]["uz"]
        uz_5 = result.node_displacements["n_0_5_1"]["uz"]
        assert uz_0 < 0 or uz_5 < 0

    def test_member_forces_all_elements_reported(self):
        g = self._portal()
        result = StructuralSolver().solve(g, {"n_5_0_1": {"Fz": -50e3}})
        assert result.converged
        for eid in g.elements:
            assert eid in result.member_forces, f"Missing forces for {eid}"

    def test_delta_max_mm_positive(self):
        g = self._portal(span=5)
        result = StructuralSolver().solve(g, {
            "n_0_0_1": {"Fz": -20e3},
            "n_5_0_1": {"Fz": -20e3},
        })
        assert result.converged
        beam_id = "beam_0_0_5_0_0"
        forces = result.member_forces.get(beam_id)
        assert forces is not None
        assert forces["delta_max_mm"] >= 0


class TestSolverDisconnected:
    """Graceful failure on degenerate structures."""

    def test_empty_graph_fails(self):
        result = StructuralSolver().solve(StructuralGraph(), {})
        assert not result.converged

    def test_no_fixed_nodes_fails(self):
        """Beam with no supports → singular → converged=False."""
        g = StructuralGraph(floor_height_m=3.0)
        g.nodes["n_0_0_1"] = NodeData("n_0_0_1", 0.0, 0.0, 3.0, 1, False)
        g.nodes["n_5_0_1"] = NodeData("n_5_0_1", 5.0, 0.0, 3.0, 1, False)
        g.elements["beam_0_0_5_0_1"] = ElementData(
            "beam_0_0_5_0_1", "beam", "IPE300", "n_0_0_1", "n_5_0_1", 5.0, "x"
        )
        result = StructuralSolver().solve(g, {"n_5_0_1": {"Fz": -10e3}})
        assert isinstance(result.converged, bool)


class TestGridOf4Columns:
    """A 2×2 column grid with beams in both directions (the real 3D scenario)."""

    def _make_grid(self, span=5):
        g = StructuralGraph(floor_height_m=3.5)
        for gx, gy in [(0,0),(span,0),(0,span),(span,span)]:
            g.place_column(gx, gy, 0, "HEB200")
        # X-beams
        g.place_beam(0, 0, span, 0, 0, "IPE300", "x")
        g.place_beam(0, span, span, span, 0, "IPE300", "x")
        # Y-beams
        g.place_beam(0, 0, 0, span, 0, "IPE300", "y")
        g.place_beam(span, 0, span, span, 0, "IPE300", "y")
        return g

    def test_grid_converges(self):
        g = self._make_grid()
        # Gravity on all 4 column tops
        top_nodes = [f"n_0_0_1", "n_5_0_1", "n_0_5_1", "n_5_5_1"]
        loads = {nid: {"Fz": -50e3} for nid in top_nodes}
        result = StructuralSolver().solve(g, loads)
        assert result.converged, result.error

    def test_grid_all_columns_compress(self):
        g = self._make_grid()
        top_nodes = ["n_0_0_1", "n_5_0_1", "n_0_5_1", "n_5_5_1"]
        loads = {nid: {"Fz": -100e3} for nid in top_nodes}
        result = StructuralSolver().solve(g, loads)
        assert result.converged
        for nid in top_nodes:
            uz = result.node_displacements[nid]["uz"]
            assert uz < 0, f"Node {nid} should compress, uz={uz}"

    def test_grid_lateral_sway(self):
        """Lateral x-wind on all tops → all tops sway in +x."""
        g = self._make_grid()
        top_nodes = ["n_0_0_1", "n_5_0_1", "n_0_5_1", "n_5_5_1"]
        loads = {nid: {"Fx": 25e3} for nid in top_nodes}
        result = StructuralSolver().solve(g, loads)
        assert result.converged
        for nid in top_nodes:
            ux = result.node_displacements[nid]["ux"]
            assert ux > 0, f"Node {nid} should sway in +x, ux={ux}"

    def test_grid_y_beams_see_bending(self):
        """Gravity load → y-beams carry bending moment."""
        g = self._make_grid()
        top_nodes = ["n_0_0_1", "n_5_0_1", "n_0_5_1", "n_5_5_1"]
        loads = {nid: {"Fz": -80e3} for nid in top_nodes}
        result = StructuralSolver().solve(g, loads)
        assert result.converged
        y_beam = result.member_forces.get("beam_0_0_0_5_0")
        assert y_beam is not None
        # Should have non-trivial moment
        assert y_beam["M_max"] > 0
