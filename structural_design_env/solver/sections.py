"""
Steel section database with real Eurocode values.
HEB series for columns, IPE series for beams.
All values in SI units (m, Pa, kg).
"""

COLUMN_SECTIONS = {
    "HEB140": {"A": 42.96e-4, "I_y": 1509e-8,  "I_z": 549.7e-8,  "J": 20.1e-8,  "W_el_y": 215.6e-6, "W_pl_y": 245.4e-6, "mass_kg_m": 33.7,  "A_v": 18.0e-4},
    "HEB160": {"A": 54.25e-4, "I_y": 2492e-8,  "I_z": 889.2e-8,  "J": 31.2e-8,  "W_el_y": 311.5e-6, "W_pl_y": 354.0e-6, "mass_kg_m": 42.6,  "A_v": 22.6e-4},
    "HEB200": {"A": 78.08e-4, "I_y": 5696e-8,  "I_z": 2003e-8,   "J": 59.3e-8,  "W_el_y": 569.6e-6, "W_pl_y": 642.5e-6, "mass_kg_m": 61.3,  "A_v": 32.0e-4},
    "HEB240": {"A": 106.0e-4, "I_y": 11260e-8, "I_z": 3923e-8,   "J": 103.0e-8, "W_el_y": 938.3e-6, "W_pl_y": 1053e-6,  "mass_kg_m": 83.2,  "A_v": 43.1e-4},
    "HEB300": {"A": 149.0e-4, "I_y": 25170e-8, "I_z": 8563e-8,   "J": 185.0e-8, "W_el_y": 1678e-6,  "W_pl_y": 1869e-6,  "mass_kg_m": 117.0, "A_v": 59.6e-4},
    "HEB360": {"A": 180.6e-4, "I_y": 43190e-8, "I_z": 15110e-8,  "J": 263.0e-8, "W_el_y": 2400e-6,  "W_pl_y": 2683e-6,  "mass_kg_m": 142.0, "A_v": 71.8e-4},
    "HEB400": {"A": 197.8e-4, "I_y": 57680e-8, "I_z": 17630e-8,  "J": 355.0e-8, "W_el_y": 2884e-6,  "W_pl_y": 3232e-6,  "mass_kg_m": 155.0, "A_v": 78.4e-4},
}

COLUMN_SECTION_ORDER = [
    "HEB140",
    "HEB160",
    "HEB200",
    "HEB240",
    "HEB300",
    "HEB360",
    "HEB400",
]

BEAM_SECTIONS = {
    "IPE200": {"A": 28.48e-4, "I_y": 1943e-8,  "I_z": 142.4e-8, "J": 7.0e-8,  "W_el_y": 194.3e-6, "W_pl_y": 220.6e-6, "mass_kg_m": 22.4, "A_v": 12.0e-4},
    "IPE240": {"A": 39.12e-4, "I_y": 3892e-8,  "I_z": 283.6e-8, "J": 12.9e-8, "W_el_y": 324.3e-6, "W_pl_y": 366.6e-6, "mass_kg_m": 30.7, "A_v": 16.2e-4},
    "IPE300": {"A": 53.81e-4, "I_y": 8356e-8,  "I_z": 604.0e-8, "J": 20.1e-8, "W_el_y": 557.1e-6, "W_pl_y": 628.4e-6, "mass_kg_m": 42.2, "A_v": 22.0e-4},
    "IPE360": {"A": 72.73e-4, "I_y": 16270e-8, "I_z": 1043e-8,  "J": 37.3e-8, "W_el_y": 903.6e-6, "W_pl_y": 1019e-6,  "mass_kg_m": 57.1, "A_v": 30.1e-4},
    "IPE400": {"A": 84.46e-4, "I_y": 23130e-8, "I_z": 1318e-8,  "J": 51.1e-8, "W_el_y": 1156e-6,  "W_pl_y": 1307e-6,  "mass_kg_m": 66.3, "A_v": 35.5e-4},
    "IPE450": {"A": 98.82e-4, "I_y": 33740e-8, "I_z": 1676e-8,  "J": 66.9e-8, "W_el_y": 1500e-6,  "W_pl_y": 1702e-6,  "mass_kg_m": 77.6, "A_v": 41.4e-4},
    "IPE500": {"A": 116.0e-4, "I_y": 48200e-8, "I_z": 2142e-8,  "J": 89.3e-8, "W_el_y": 1928e-6,  "W_pl_y": 2194e-6,  "mass_kg_m": 90.7, "A_v": 48.0e-4},
}

BEAM_SECTION_ORDER = [
    "IPE200",
    "IPE240",
    "IPE300",
    "IPE360",
    "IPE400",
    "IPE450",
    "IPE500",
]

# Material constants
E_STEEL = 210e9       # Pa (Young's modulus)
F_Y_STEEL = 355e6     # Pa (yield strength S355)
RHO_STEEL = 7850.0    # kg/m³
G_STEEL = 80.77e9     # Pa (shear modulus for S355 steel)


def get_section_props(section_name: str) -> dict:
    """Return section properties dict for given section name."""
    if section_name in COLUMN_SECTIONS:
        return COLUMN_SECTIONS[section_name]
    if section_name in BEAM_SECTIONS:
        return BEAM_SECTIONS[section_name]
    raise ValueError(f"Unknown section: {section_name}")


def upgrade_section(section_name: str) -> str | None:
    """Return next larger section, or None if already at max."""
    if section_name in COLUMN_SECTION_ORDER:
        idx = COLUMN_SECTION_ORDER.index(section_name)
        if idx < len(COLUMN_SECTION_ORDER) - 1:
            return COLUMN_SECTION_ORDER[idx + 1]
        return None
    if section_name in BEAM_SECTION_ORDER:
        idx = BEAM_SECTION_ORDER.index(section_name)
        if idx < len(BEAM_SECTION_ORDER) - 1:
            return BEAM_SECTION_ORDER[idx + 1]
        return None
    raise ValueError(f"Unknown section: {section_name}")


def downgrade_section(section_name: str) -> str | None:
    """Return next smaller section, or None if already at min."""
    if section_name in COLUMN_SECTION_ORDER:
        idx = COLUMN_SECTION_ORDER.index(section_name)
        if idx > 0:
            return COLUMN_SECTION_ORDER[idx - 1]
        return None
    if section_name in BEAM_SECTION_ORDER:
        idx = BEAM_SECTION_ORDER.index(section_name)
        if idx > 0:
            return BEAM_SECTION_ORDER[idx - 1]
        return None
    raise ValueError(f"Unknown section: {section_name}")


def get_section_family(section_name: str) -> str:
    """Return 'column' or 'beam'."""
    if section_name in COLUMN_SECTIONS:
        return "column"
    if section_name in BEAM_SECTIONS:
        return "beam"
    raise ValueError(f"Unknown section: {section_name}")
