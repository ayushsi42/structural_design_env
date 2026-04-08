"""
Seismic analysis per EN 1998-1 (Eurocode 8).
Equivalent lateral force method using the design spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SeismicResult:
    T1: float               # Fundamental period [s]
    Sd: float               # Spectral acceleration [g]
    F_b_kN: float           # Base shear [kN]
    floor_forces_kN: List[float]  # Per-floor force (index 0 = ground+1)


def compute_seismic_shear(
    W_kN: float,
    H_m: float,
    ag_g: float,
    n_floors: int = 1,
    gamma_I: float = 1.0,
) -> SeismicResult:
    """
    Compute seismic base shear and floor distribution per EN 1998-1.

    Parameters
    ----------
    W_kN : total seismic weight [kN]
    H_m  : total building height [m]
    ag_g : design ground acceleration [fraction of g]
    n_floors : number of floors
    gamma_I  : importance factor

    Returns
    -------
    SeismicResult
    """
    if ag_g <= 0.0 or W_kN <= 0.0:
        return SeismicResult(
            T1=0.0,
            Sd=0.0,
            F_b_kN=0.0,
            floor_forces_kN=[0.0] * max(n_floors, 1),
        )

    # Fundamental period estimate (Ct method, EN 1998 eq. 4.6)
    T1 = 0.085 * H_m**0.75  # for steel moment frames

    # EN 1998 Type 1 spectrum, soil type B
    TB = 0.15
    TC = 0.60
    TD = 2.00
    S = 1.20   # soil factor for type B
    eta = 1.0  # damping correction (5%)

    if T1 < TB:
        Sd = ag_g * S * (2.0 / 3.0 + T1 / TB * (2.5 * eta - 2.0 / 3.0))
    elif T1 <= TC:
        Sd = ag_g * S * 2.5 * eta
    elif T1 <= TD:
        Sd = ag_g * S * 2.5 * eta * (TC / T1)
    else:
        Sd = ag_g * S * 2.5 * eta * (TC * TD / T1**2)

    # Apply importance factor
    Sd *= gamma_I

    # Base shear correction factor (lambda)
    lambda_factor = 0.85 if n_floors >= 2 else 1.0

    F_b_kN = Sd * W_kN * lambda_factor  # kN (since W is in kN)

    # Distribute as inverted triangle (EN 1998 eq. 4.11)
    # F_i = F_b * (z_i * m_i) / sum(z_j * m_j)
    # Assume uniform mass per floor: m_i = W_kN / n_floors
    # z_i = floor height * i (floor levels: 1..n_floors)
    floor_heights = [float(i + 1) for i in range(n_floors)]
    sum_z = sum(floor_heights)

    if sum_z <= 0:
        floor_forces_kN = [F_b_kN / max(n_floors, 1)] * n_floors
    else:
        floor_forces_kN = [F_b_kN * z / sum_z for z in floor_heights]

    return SeismicResult(
        T1=T1,
        Sd=Sd,
        F_b_kN=F_b_kN,
        floor_forces_kN=floor_forces_kN,
    )
