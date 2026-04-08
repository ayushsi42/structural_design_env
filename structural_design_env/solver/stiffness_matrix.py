"""
Direct Stiffness Method solver for 3D steel frames.

Global coordinate system: x=East, y=North, z=Up
Each node has 6 DOFs: [ux, uy, uz, rx, ry, rz]
Global DOF index for node i: [6i, 6i+1, 6i+2, 6i+3, 6i+4, 6i+5]

Elements: 3D Euler-Bernoulli beam-column with 12x12 local stiffness.
Local DOF order per node: [u1(axial), u2(transv 1-2), u3(transv 1-3), r1(torsion), r2(bend 1-3), r3(bend 1-2)]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from structural_design_env.solver.sections import (
    COLUMN_SECTIONS,
    BEAM_SECTIONS,
    E_STEEL,
    G_STEEL,
)

G_CONCRETE = 12.5e9  # Pa, concrete shear modulus


@dataclass
class SolverResult:
    u: np.ndarray                     # full displacement vector (6*n_nodes,)
    member_forces: Dict[str, dict]    # element_id -> {N, V, M_max, delta_max_mm}
    node_displacements: Dict[str, dict]  # node_id -> {ux, uy, uz, rx, ry, rz}
    converged: bool
    error: Optional[str] = None


def _local_stiffness(E: float, A: float, I_y: float, I_z: float, G: float, J: float, L: float) -> np.ndarray:
    """
    12x12 local stiffness matrix for a 3D Euler-Bernoulli beam element.

    Local DOF order (i then j):
      0: u1 axial
      1: u2 transverse in 1-2 plane
      2: u3 transverse in 1-3 plane
      3: r1 torsion
      4: r2 bending about local-2 axis (couples with u3)
      5: r3 bending about local-3 axis (couples with u2)
      6: u1 at j
      7: u2 at j
      8: u3 at j
      9: r1 at j
     10: r2 at j
     11: r3 at j

    I_y: moment of inertia about local y (bending in 1-3 plane, r2/u3 coupling)
         Also called I_major for the plane that produces My
    I_z: moment of inertia about local z (bending in 1-2 plane, r3/u2 coupling)
         Also called I_minor
    """
    EA_L = E * A / L
    # Bending in 1-2 plane (u2, r3): uses I_z
    EIz = E * I_z
    EIz_L3 = EIz / L**3
    EIz_L2 = EIz / L**2
    EIz_L  = EIz / L
    # Bending in 1-3 plane (u3, r2): uses I_y
    EIy = E * I_y
    EIy_L3 = EIy / L**3
    EIy_L2 = EIy / L**2
    EIy_L  = EIy / L
    # Torsion
    GJ_L = G * J / L

    k = np.zeros((12, 12))

    # Axial (u1: 0 and 6)
    k[0, 0] =  EA_L;  k[0, 6] = -EA_L
    k[6, 0] = -EA_L;  k[6, 6] =  EA_L

    # Bending in 1-2 plane (u2: 1,7 and r3: 5,11)
    k[1, 1]  =  12*EIz_L3;  k[1, 7]  = -12*EIz_L3
    k[7, 1]  = -12*EIz_L3;  k[7, 7]  =  12*EIz_L3
    k[1, 5]  =   6*EIz_L2;  k[1, 11] =   6*EIz_L2
    k[5, 1]  =   6*EIz_L2;  k[11, 1] =   6*EIz_L2
    k[7, 5]  =  -6*EIz_L2;  k[7, 11] =  -6*EIz_L2
    k[5, 7]  =  -6*EIz_L2;  k[11, 7] =  -6*EIz_L2
    k[5, 5]  =   4*EIz_L;   k[11, 11] =  4*EIz_L
    k[5, 11] =   2*EIz_L;   k[11, 5]  =  2*EIz_L

    # Bending in 1-3 plane (u3: 2,8 and r2: 4,10)
    # Note: sign convention for r2 coupling with u3 is opposite to r3/u2
    k[2, 2]  =  12*EIy_L3;  k[2, 8]  = -12*EIy_L3
    k[8, 2]  = -12*EIy_L3;  k[8, 8]  =  12*EIy_L3
    k[2, 4]  =  -6*EIy_L2;  k[2, 10] =  -6*EIy_L2
    k[4, 2]  =  -6*EIy_L2;  k[10, 2] =  -6*EIy_L2
    k[8, 4]  =   6*EIy_L2;  k[8, 10] =   6*EIy_L2
    k[4, 8]  =   6*EIy_L2;  k[10, 8] =   6*EIy_L2
    k[4, 4]  =   4*EIy_L;   k[10, 10] =  4*EIy_L
    k[4, 10] =   2*EIy_L;   k[10, 4]  =  2*EIy_L

    # Torsion (r1: 3 and 9)
    k[3, 3] =  GJ_L;  k[3, 9] = -GJ_L
    k[9, 3] = -GJ_L;  k[9, 9] =  GJ_L

    return k


def _build_T(R: np.ndarray) -> np.ndarray:
    """Build 12x12 transformation matrix from 3x3 rotation matrix R (GLOBAL->LOCAL)."""
    T = np.zeros((12, 12))
    for block in range(4):
        T[3*block:3*block+3, 3*block:3*block+3] = R
    return T


# Rotation matrices: R maps GLOBAL -> LOCAL (local_vec = R @ global_vec)

# Column: along global z; local-x = z_global (axial up)
#   local 1 = z_global, local 2 = x_global, local 3 = y_global
R_COL = np.array([
    [0, 0, 1],   # x_local = z_global
    [1, 0, 0],   # y_local = x_global
    [0, 1, 0],   # z_local = y_global
], dtype=float)

# X-beam: along global x; local-x = x_global (axial east)
#   gravity is -z_global, which maps to -y_local  => local-2 = z_global
R_XBEAM = np.array([
    [1,  0, 0],   # x_local = x_global
    [0,  0, 1],   # y_local = z_global  (up; gravity in -y_local)
    [0, -1, 0],   # z_local = -y_global
], dtype=float)

# Y-beam: along global y; local-x = y_global (axial north)
#   gravity is -z_global => local-2 = z_global
R_YBEAM = np.array([
    [0, 1, 0],   # x_local = y_global
    [0, 0, 1],   # y_local = z_global  (up; gravity in -y_local)
    [1, 0, 0],   # z_local = x_global
], dtype=float)

T_COL   = _build_T(R_COL)
T_XBEAM = _build_T(R_XBEAM)
T_YBEAM = _build_T(R_YBEAM)


def _get_wall_section_props(thickness_m: float, floor_height_m: float):
    """
    Section properties for a concrete shear wall spanning horizontally.
    The wall cross-section depth = floor_height_m, width = thickness_m.
    """
    h = floor_height_m
    t = thickness_m
    A = t * h
    I_major = t * h**3 / 12.0   # bending in vertical plane (strong)
    I_minor = h * t**3 / 12.0   # bending out of plane (weak)
    GJ = G_CONCRETE * t**3 * h / 3.0
    E_wall = 30e9  # concrete
    return A, I_major, I_minor, GJ, E_wall


class StructuralSolver:
    """Assembles and solves the 3D global stiffness matrix (6 DOF/node)."""

    def solve(self, graph, loads: dict) -> SolverResult:
        """
        Solve the structural system.

        Parameters
        ----------
        graph : StructuralGraph
        loads : dict of node_id -> {"Fx": float, "Fy": float, "Fz": float}
                Forces in Newtons (Fz = vertical/gravity).

        Returns
        -------
        SolverResult
        """
        nodes = graph.nodes
        elements = graph.elements
        floor_height_m = graph.floor_height_m

        if len(nodes) < 2:
            return SolverResult(
                u=np.zeros(6),
                member_forces={},
                node_displacements={},
                converged=False,
                error="Insufficient nodes (need at least 2)",
            )

        # Build node index map
        node_ids = sorted(nodes.keys())
        idx_map = {nid: i for i, nid in enumerate(node_ids)}
        n_nodes = len(node_ids)
        n_dof = 6 * n_nodes

        # Assemble global stiffness matrix (COO format)
        rows, cols_list, vals = [], [], []

        def add_to_K(dof_i_start: int, dof_j_start: int, k_global: np.ndarray):
            dofs = list(range(dof_i_start, dof_i_start + 6)) + list(range(dof_j_start, dof_j_start + 6))
            for li in range(12):
                for lj in range(12):
                    rows.append(dofs[li])
                    cols_list.append(dofs[lj])
                    vals.append(k_global[li, lj])

        for eid, elem in elements.items():
            if elem.node_i not in idx_map or elem.node_j not in idx_map:
                continue

            ni = idx_map[elem.node_i]
            nj = idx_map[elem.node_j]
            L = elem.length_m

            if L <= 0:
                continue

            if elem.element_type == "wall":
                thickness = elem.thickness_m or 0.2
                A, I_major, I_minor, GJ, E = _get_wall_section_props(thickness, floor_height_m)
                G = G_CONCRETE
                T = T_XBEAM if (elem.orientation == "x") else T_YBEAM
            elif elem.element_type == "column":
                props = COLUMN_SECTIONS.get(elem.section) or BEAM_SECTIONS.get(elem.section)
                if props is None:
                    continue
                A = props["A"]
                I_major = props["I_y"]
                I_minor = props.get("I_z", I_major * 0.3)  # fallback
                J = props.get("J", I_major * 0.01)
                GJ = G_STEEL * J
                E = E_STEEL
                G = G_STEEL
                T = T_COL
            else:
                # beam
                props = COLUMN_SECTIONS.get(elem.section) or BEAM_SECTIONS.get(elem.section)
                if props is None:
                    continue
                A = props["A"]
                I_major = props["I_y"]   # strong axis (gravity bending)
                I_minor = props.get("I_z", I_major * 0.05)  # weak axis
                J = props.get("J", I_major * 0.01)
                GJ = G_STEEL * J
                E = E_STEEL
                G = G_STEEL
                orient = elem.orientation or "x"
                T = T_XBEAM if (orient == "x") else T_YBEAM

            # For columns: I_y in local frame is strong axis (bending in 1-3 plane)
            # For beams: I_y in local frame is strong axis (gravity, bending in 1-2 plane → I_z_local = I_major)
            # Local stiffness: I_y param = bending in 1-3 plane, I_z param = bending in 1-2 plane
            if elem.element_type == "column":
                # Column local: 1-2 plane = xz_global, 1-3 plane = yz_global
                # strong axis (I_y of section) → whichever is larger; use I_major for 1-3, I_minor for 1-2
                k_local = _local_stiffness(E, A, I_major, I_minor, G_STEEL if elem.element_type != "wall" else G_CONCRETE, J if elem.element_type != "wall" else GJ / G_STEEL, L)
                # Fix: for walls we handle separately
            elif elem.element_type == "wall":
                # GJ already computed; pass J=GJ/G_CONCRETE
                J_eff = GJ / G_CONCRETE
                k_local = _local_stiffness(E, A, I_major, I_minor, G_CONCRETE, J_eff, L)
            else:
                # Beam: gravity bending in 1-2 plane → I_z_local = I_major (I_y of section)
                #        lateral bending in 1-3 plane → I_y_local = I_minor (I_z of section)
                J_eff = props.get("J", I_major * 0.01)
                k_local = _local_stiffness(E, A, I_minor, I_major, G_STEEL, J_eff, L)

            k_global = T.T @ k_local @ T

            add_to_K(6 * ni, 6 * nj, k_global)

        if not vals:
            return SolverResult(
                u=np.zeros(n_dof),
                member_forces={},
                node_displacements={},
                converged=False,
                error="No elements assembled",
            )

        K = sp.coo_matrix((vals, (rows, cols_list)), shape=(n_dof, n_dof)).tocsr()

        # Build force vector
        F = np.zeros(n_dof)
        for nid, force in loads.items():
            if nid in idx_map:
                i = idx_map[nid]
                F[6*i]   += force.get("Fx", 0.0)
                F[6*i+1] += force.get("Fy", 0.0)
                F[6*i+2] += force.get("Fz", 0.0)

        # Identify fixed DOFs (floor == 0 → all 6 fixed)
        fixed_dofs = set()
        for nid, nd in nodes.items():
            if nd.is_fixed_base:
                i = idx_map[nid]
                fixed_dofs.update(range(6*i, 6*i+6))

        all_dofs = set(range(n_dof))
        free_dofs = sorted(all_dofs - fixed_dofs)

        if not free_dofs:
            return SolverResult(
                u=np.zeros(n_dof),
                member_forces={},
                node_displacements={},
                converged=False,
                error="All DOFs are fixed — no free DOFs to solve",
            )

        # Partition and solve
        free_arr = np.array(free_dofs)
        K_ff = K[free_arr, :][:, free_arr]
        F_f = F[free_arr]

        try:
            u_f = spla.spsolve(K_ff.tocsr(), F_f)
            if np.any(np.isnan(u_f)) or np.any(np.isinf(u_f)):
                raise ValueError("Solver returned NaN/Inf — likely singular matrix")
        except Exception as exc:
            return SolverResult(
                u=np.zeros(n_dof),
                member_forces={},
                node_displacements={},
                converged=False,
                error=str(exc),
            )

        # Reconstruct full displacement vector
        u_full = np.zeros(n_dof)
        u_full[free_arr] = u_f

        # Compute member forces
        member_forces: Dict[str, dict] = {}
        for eid, elem in elements.items():
            if elem.node_i not in idx_map or elem.node_j not in idx_map:
                continue

            ni = idx_map[elem.node_i]
            nj = idx_map[elem.node_j]
            L = elem.length_m

            if L <= 0:
                continue

            if elem.element_type == "wall":
                thickness = elem.thickness_m or 0.2
                A, I_major, I_minor, GJ, E = _get_wall_section_props(thickness, floor_height_m)
                G = G_CONCRETE
                J_eff = GJ / G_CONCRETE
                k_local = _local_stiffness(E, A, I_major, I_minor, G_CONCRETE, J_eff, L)
                T = T_XBEAM if (elem.orientation == "x") else T_YBEAM
            elif elem.element_type == "column":
                props = COLUMN_SECTIONS.get(elem.section) or BEAM_SECTIONS.get(elem.section)
                if props is None:
                    continue
                A = props["A"]; I_major = props["I_y"]; I_minor = props.get("I_z", I_major*0.3)
                J = props.get("J", I_major*0.01)
                k_local = _local_stiffness(E_STEEL, A, I_major, I_minor, G_STEEL, J, L)
                T = T_COL
            else:
                props = COLUMN_SECTIONS.get(elem.section) or BEAM_SECTIONS.get(elem.section)
                if props is None:
                    continue
                A = props["A"]; I_major = props["I_y"]; I_minor = props.get("I_z", I_major*0.05)
                J = props.get("J", I_major*0.01)
                k_local = _local_stiffness(E_STEEL, A, I_minor, I_major, G_STEEL, J, L)
                orient = elem.orientation or "x"
                T = T_XBEAM if (orient == "x") else T_YBEAM

            u_elem_g = np.concatenate([u_full[6*ni:6*ni+6], u_full[6*nj:6*nj+6]])
            u_elem_l = T @ u_elem_g
            f_local = k_local @ u_elem_l

            N = f_local[0]   # axial at i
            Vy_i = f_local[1]
            Vz_i = f_local[2]
            V = np.sqrt(Vy_i**2 + Vz_i**2)
            My_i = f_local[4]
            Mz_i = f_local[5]
            My_j = f_local[10]
            Mz_j = f_local[11]
            M_i = np.sqrt(My_i**2 + Mz_i**2)
            M_j = np.sqrt(My_j**2 + Mz_j**2)
            M_max = max(M_i, M_j)

            # Deflection: for beams use vertical (uz_global) at endpoints
            # For beams: uz_global = u_full[6*n+2]
            if elem.element_type == "beam":
                uz_i = u_full[6*ni + 2]
                uz_j = u_full[6*nj + 2]
                delta_max_mm = max(abs(uz_i), abs(uz_j)) * 1000.0
            else:
                # Column: lateral displacement
                ux_i = u_full[6*ni]
                ux_j = u_full[6*nj]
                uy_i = u_full[6*ni+1]
                uy_j = u_full[6*nj+1]
                drift = np.sqrt((ux_j-ux_i)**2 + (uy_j-uy_i)**2)
                delta_max_mm = drift * 1000.0

            member_forces[eid] = {
                "N": N,
                "V": float(V),
                "M_max": float(M_max),
                "delta_max_mm": float(delta_max_mm),
            }

        # Build node displacement map
        node_displacements = {}
        for nid in node_ids:
            i = idx_map[nid]
            node_displacements[nid] = {
                "ux": float(u_full[6*i]),
                "uy": float(u_full[6*i+1]),
                "uz": float(u_full[6*i+2]),
                "rx": float(u_full[6*i+3]),
                "ry": float(u_full[6*i+4]),
                "rz": float(u_full[6*i+5]),
            }

        return SolverResult(
            u=u_full,
            member_forces=member_forces,
            node_displacements=node_displacements,
            converged=True,
        )
