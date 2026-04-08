"""
Eurocode 3 (EN 1993-1-1) member code checks + RC shear wall checks (EC2).

Checks implemented:
- Bending (clause 6.2.5)
- Shear (clause 6.2.6)
- Flexural buckling for columns (clause 6.3.1, curve b)
- Combined axial + bending interaction (clause 6.3.3 simplified)
- Deflection for beams (L/300 limit)
- RC shear wall in-plane shear + bending (EC2 simplified, C25/30)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from structural_design_env.solver.sections import E_STEEL, F_Y_STEEL

# Material — steel
E = E_STEEL        # Pa
f_y = F_Y_STEEL    # Pa

# Material — concrete (C25/30, EC2)
F_CK = 25e6        # Pa  characteristic compressive strength
F_CD = F_CK / 1.5  # Pa  design compressive strength (~16.67 MPa)
F_CTD = 1.2e6      # Pa  design tensile strength (Table 3.1 approx)


@dataclass
class MemberChecks:
    UR_bending: float       # sigma_Ed / f_y  (or M_Ed/M_Rd for walls)
    UR_shear: float         # V_Ed / V_Rd
    UR_buckling: float      # N_Ed / N_b_Rd (columns), 0 for beams/walls
    UR_interaction: float   # combined axial+bending for columns
    UR_deflection: float    # delta_max / (L/300) for beams
    passes_all: bool
    max_UR: float


def check_member(
    element_type: str,       # "column" | "beam" | "wall"
    section_props: dict,     # from COLUMN_SECTIONS or BEAM_SECTIONS (ignored for walls)
    forces: dict,            # {"N": float, "V": float, "M_max": float, "delta_max_mm": float}
    L_m: float,              # element length [m]
    L_eff_factor: float = 1.0,   # effective length factor (buckling)
    floor_height_m: float = 3.5, # wall panel depth [m]
    thickness_m: float = 0.2,    # wall panel thickness [m]
) -> MemberChecks:
    """
    Perform code checks on a structural member.

    Parameters
    ----------
    element_type : "column", "beam", or "wall"
    section_props : section property dict (steel sections); not used for walls
    forces : solver output forces dict
    L_m : physical length [m]
    L_eff_factor : effective length factor for column buckling
                   0.7  → braced (non-sway) frame   [EC3 §6.3.1]
                   1.0  → fixed-pinned approx
                   1.5  → unbraced (sway) frame
    floor_height_m : wall panel in-plane depth [m]  (walls only)
    thickness_m    : wall panel out-of-plane width [m] (walls only)

    Returns
    -------
    MemberChecks dataclass
    """
    # ------------------------------------------------------------------
    # RC Shear wall: in-plane shear + bending check (EC2 simplified)
    # ------------------------------------------------------------------
    if element_type == "wall":
        h_w = floor_height_m   # in-plane depth
        t_w = thickness_m      # panel thickness

        A_w = t_w * h_w                   # in-plane cross-section area [m²]
        W_w = t_w * h_w ** 2 / 6.0       # elastic modulus about in-plane axis [m³]

        V_Ed = abs(forces.get("V", 0.0))
        M_Ed = abs(forces.get("M_max", 0.0))
        N_Ed = forces.get("N", 0.0)       # axial (compression negative conv.)

        # In-plane shear capacity (diagonal tension, EC2 §6.2)
        V_Rd = 0.5 * F_CTD * A_w
        UR_shear = V_Ed / V_Rd if V_Rd > 0 else 0.0

        # In-plane bending + axial: combined fiber stress vs f_cd
        sigma_M = M_Ed / W_w if W_w > 0 else 0.0
        sigma_N = abs(N_Ed) / A_w if A_w > 0 else 0.0
        sigma_Ed = sigma_M + sigma_N
        UR_bending = sigma_Ed / F_CD

        all_URs = [UR_bending, UR_shear]
        max_UR = max(all_URs)

        return MemberChecks(
            UR_bending=round(UR_bending, 4),
            UR_shear=round(UR_shear, 4),
            UR_buckling=0.0,
            UR_interaction=0.0,
            UR_deflection=0.0,
            passes_all=max_UR <= 1.0,
            max_UR=round(max_UR, 4),
        )

    # ------------------------------------------------------------------
    # Steel section checks
    # ------------------------------------------------------------------
    A = section_props["A"]           # m²
    I_y = section_props["I_y"]       # m⁴
    W_el_y = section_props["W_el_y"] # m³
    W_pl_y = section_props["W_pl_y"] # m³
    A_v = section_props["A_v"]       # m² shear area

    N_Ed = forces.get("N", 0.0)      # N (positive = tension)
    V_Ed = forces.get("V", 0.0)      # N
    M_max = forces.get("M_max", 0.0) # N·m
    delta_max_mm = forces.get("delta_max_mm", 0.0)  # mm

    L_eff = L_m * L_eff_factor  # effective length [m]

    # ------------------------------------------------------------------
    # 1. Bending check: EN 1993-1-1 cl. 6.2.5
    # ------------------------------------------------------------------
    sigma_Ed = abs(M_max) / W_el_y if W_el_y > 0 else 0.0
    UR_bending = sigma_Ed / f_y

    # ------------------------------------------------------------------
    # 2. Shear check: EN 1993-1-1 cl. 6.2.6
    # ------------------------------------------------------------------
    V_Rd = f_y * A_v / math.sqrt(3.0)
    UR_shear = abs(V_Ed) / V_Rd if V_Rd > 0 else 0.0

    # ------------------------------------------------------------------
    # 3. Flexural buckling (columns only): EN 1993-1-1 cl. 6.3.1, curve b
    # ------------------------------------------------------------------
    UR_buckling = 0.0
    UR_interaction = 0.0

    if element_type == "column" and A > 0 and I_y > 0:
        # Slenderness ratio: lambda_1 = pi * sqrt(E/f_y) ≈ 76.4 for S355
        lambda_1 = math.pi * math.sqrt(E / f_y)

        # Radius of gyration (strong axis governs for HEB — I_y > I_z)
        r = math.sqrt(I_y / A)

        # Non-dimensional slenderness (EN 1993-1-1 §6.3.1.3)
        lambda_bar = (L_eff / r) / lambda_1
        lambda_bar = max(lambda_bar, 0.001)

        # Imperfection factor: curve b for HEB hot-rolled sections
        alpha = 0.34

        # Reduction factor chi (EN 1993-1-1 eq. 6.49)
        Phi = 0.5 * (1.0 + alpha * (lambda_bar - 0.2) + lambda_bar ** 2)
        chi = 1.0 / (Phi + math.sqrt(max(Phi ** 2 - lambda_bar ** 2, 0.0)))
        chi = min(chi, 1.0)
        chi = max(chi, 0.01)

        # Buckling resistance
        N_b_Rd = chi * A * f_y
        UR_buckling = abs(N_Ed) / N_b_Rd if N_b_Rd > 0 else 0.0

        # 4. Combined axial + bending: EN 1993-1-1 cl. 6.3.3 (simplified)
        denom_N = chi * A * f_y
        k_yy = 1.0
        if denom_N > 0 and lambda_bar > 0:
            k_yy = min(1.0 + (lambda_bar - 0.2) * abs(N_Ed) / denom_N, 1.5)

        M_Rd = W_pl_y * f_y   # plastic moment resistance
        N_Rd = A * f_y        # squash load

        term_N = abs(N_Ed) / N_Rd if N_Rd > 0 else 0.0
        term_M = k_yy * abs(M_max) / M_Rd if M_Rd > 0 else 0.0
        UR_interaction = term_N + term_M

    # ------------------------------------------------------------------
    # 5. Deflection check (beams only): L/300
    # ------------------------------------------------------------------
    UR_deflection = 0.0
    if element_type == "beam" and L_m > 0:
        defl_limit_mm = (L_m * 1000.0) / 300.0
        UR_deflection = delta_max_mm / defl_limit_mm if defl_limit_mm > 0 else 0.0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_URs = [UR_bending, UR_shear, UR_buckling, UR_interaction, UR_deflection]
    max_UR = max(all_URs)
    passes_all = max_UR <= 1.0

    return MemberChecks(
        UR_bending=round(UR_bending, 4),
        UR_shear=round(UR_shear, 4),
        UR_buckling=round(UR_buckling, 4),
        UR_interaction=round(UR_interaction, 4),
        UR_deflection=round(UR_deflection, 4),
        passes_all=passes_all,
        max_UR=round(max_UR, 4),
    )
