"""
Eurocode 3 (EN 1993-1-1) member code checks.

Checks implemented:
- Bending (clause 6.2.5)
- Shear (clause 6.2.6)
- Flexural buckling for columns (clause 6.3.1, curve b)
- Combined axial + bending interaction (clause 6.3.3 simplified)
- Deflection for beams (L/300 limit)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from structural_design_env.solver.sections import E_STEEL, F_Y_STEEL

# Material
E = E_STEEL        # Pa
f_y = F_Y_STEEL    # Pa


@dataclass
class MemberChecks:
    UR_bending: float       # sigma_Ed / f_y
    UR_shear: float         # V_Ed / V_Rd
    UR_buckling: float      # N_Ed / N_b_Rd (columns), 0 for beams
    UR_interaction: float   # combined axial+bending for columns
    UR_deflection: float    # delta_max / (L/300) for beams
    passes_all: bool
    max_UR: float


def check_member(
    element_type: str,      # "column" | "beam" | "wall"
    section_props: dict,    # from COLUMN_SECTIONS or BEAM_SECTIONS
    forces: dict,           # {"N": float, "V": float, "M_max": float, "delta_max_mm": float}
    L_m: float,             # element length [m]
    L_eff_factor: float = 1.0,  # effective length factor (1.0 for fixed-pinned approx)
) -> MemberChecks:
    """
    Perform Eurocode 3 code checks on a member.

    Parameters
    ----------
    element_type : "column", "beam", or "wall"
    section_props : section property dict
    forces : solver output forces dict
    L_m : physical length [m]
    L_eff_factor : effective length factor for buckling

    Returns
    -------
    MemberChecks dataclass
    """
    if element_type == "wall":
        # Walls are not individually code-checked — always pass
        return MemberChecks(
            UR_bending=0.0,
            UR_shear=0.0,
            UR_buckling=0.0,
            UR_interaction=0.0,
            UR_deflection=0.0,
            passes_all=True,
            max_UR=0.0,
        )

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
    # 3. Flexural buckling (columns only): EN 1993-1-1 cl. 6.3.1
    # ------------------------------------------------------------------
    UR_buckling = 0.0
    UR_interaction = 0.0

    if element_type == "column" and A > 0 and I_y > 0:
        # Slenderness ratio: lambda_1 = pi * sqrt(E/f_y)
        lambda_1 = math.pi * math.sqrt(E / f_y)  # ~76.4 for S355

        # Radius of gyration
        r = math.sqrt(I_y / A)

        # Non-dimensional slenderness
        lambda_bar = (L_eff / r) / lambda_1
        lambda_bar = max(lambda_bar, 0.001)  # avoid zero

        # Imperfection factor alpha for curve b (HEB sections)
        alpha = 0.34

        # Reduction factor chi
        Phi = 0.5 * (1.0 + alpha * (lambda_bar - 0.2) + lambda_bar**2)
        chi = 1.0 / (Phi + math.sqrt(max(Phi**2 - lambda_bar**2, 0.0)))
        chi = min(chi, 1.0)
        chi = max(chi, 0.01)

        # Buckling resistance
        N_b_Rd = chi * A * f_y
        UR_buckling = abs(N_Ed) / N_b_Rd if N_b_Rd > 0 else 0.0

        # 4. Combined axial + bending interaction: EN 1993-1-1 cl. 6.3.3
        denom_N = chi * A * f_y
        k_yy = 1.0
        if denom_N > 0 and lambda_bar > 0:
            k_yy = min(1.0 + (lambda_bar - 0.2) * abs(N_Ed) / denom_N, 1.5)

        M_Rd = W_pl_y * f_y  # plastic moment resistance
        N_Rd = A * f_y       # squash load

        term_N = abs(N_Ed) / N_Rd if N_Rd > 0 else 0.0
        term_M = k_yy * abs(M_max) / M_Rd if M_Rd > 0 else 0.0
        UR_interaction = term_N + term_M

    # ------------------------------------------------------------------
    # 5. Deflection check (beams only)
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
