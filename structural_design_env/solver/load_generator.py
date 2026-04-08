"""
Load generator: converts task config + structural graph into nodal force vectors.

Gravity (dead + live), wind, and seismic loads are computed and assembled
into a dict {node_id: {"Fx": float, "Fy": float, "Fz": float}}.

Coordinate system: x=East, y=North, z=Up
  Fz = gravity load (downward = negative Z)
  Fx = wind/seismic in x direction
  Fy = wind/seismic in y direction
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from structural_design_env.solver.seismic import compute_seismic_shear


def _parse_node_coords(node_id: str) -> Tuple[int, int, int]:
    """Parse 'n_{x}_{y}_{floor}' -> (x, y, floor)."""
    parts = node_id.split("_")
    # Format: n_{x}_{y}_{floor}
    x = int(parts[1])
    y = int(parts[2])
    floor = int(parts[3])
    return x, y, floor


def _compute_tributary_areas(graph) -> Dict[str, float]:
    """
    Compute tributary floor area (m²) for each node at the TOP of each column.

    For each column at plan position (cx, cy), search ALL other columns
    regardless of grid alignment:
      dx_left  = (cx - max(all_x < cx)) / 2  or  cx/2 if none
      dx_right = (min(all_x > cx) - cx) / 2  or  (19-cx)/2 if none
      dy_below = (cy - max(all_y < cy)) / 2  or  cy/2 if none
      dy_above = (min(all_y > cy) - cy) / 2  or  (19-cy)/2 if none
      area = (dx_left + dx_right) * (dy_below + dy_above)
    """
    # Group column TOP nodes by floor (floor+1 level)
    floor_columns: Dict[int, list] = {}  # floor_level -> [(x, y, node_id)]

    for eid, elem in graph.elements.items():
        if elem.element_type != "column":
            continue
        nid = elem.node_j
        nd = graph.nodes.get(nid)
        if nd is None:
            continue
        floor_level = nd.floor
        x, y, _ = _parse_node_coords(nid)
        floor_columns.setdefault(floor_level, []).append((x, y, nid))

    trib: Dict[str, float] = {}

    for floor_level, cols in floor_columns.items():
        # All unique x and y positions at this floor (from ALL columns)
        all_xs = sorted(set(c[0] for c in cols))
        all_ys = sorted(set(c[1] for c in cols))

        for cx, cy, nid in cols:
            # X direction: search ALL columns regardless of y
            left_xs  = [x for x in all_xs if x < cx]
            right_xs = [x for x in all_xs if x > cx]
            dx_left  = (cx - max(left_xs))  / 2.0 if left_xs  else cx / 2.0
            dx_right = (min(right_xs) - cx) / 2.0 if right_xs else (19 - cx) / 2.0

            # Y direction: search ALL columns regardless of x
            below_ys = [y for y in all_ys if y < cy]
            above_ys = [y for y in all_ys if y > cy]
            dy_below = (cy - max(below_ys)) / 2.0 if below_ys else cy / 2.0
            dy_above = (min(above_ys) - cy)  / 2.0 if above_ys else (19 - cy) / 2.0

            trib[nid] = (dx_left + dx_right) * (dy_below + dy_above)

    return trib


def generate_loads(graph, task_config) -> Dict[str, Dict[str, float]]:
    """
    Generate nodal loads from task configuration.

    Parameters
    ----------
    graph : StructuralGraph
    task_config : TaskConfig

    Returns
    -------
    dict of node_id -> {"Fx": float, "Fy": float, "Fz": float}  (values in Newtons)
    """
    loads: Dict[str, Dict[str, float]] = {
        nid: {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0} for nid in graph.nodes
    }

    n_floors = task_config.n_floors
    floor_height = task_config.floor_height_m
    dead_kPa = task_config.dead_load_kPa
    live_kPa = task_config.live_load_kPa
    wind_kN_m = task_config.wind_load_kN_per_m
    ag_g = task_config.seismic_ag_g
    gamma_I = task_config.seismic_gamma_I
    site_width = task_config.site_width_m

    # ------------------------------------------------------------------
    # 1. Gravity loads (Fz, downward = negative)
    # ------------------------------------------------------------------
    trib_areas = _compute_tributary_areas(graph)
    total_load_kPa = dead_kPa + live_kPa  # kN/m²

    for nid, area_m2 in trib_areas.items():
        nd = graph.nodes.get(nid)
        if nd is None:
            continue
        # Count how many floors are above this node
        floors_above = n_floors - nd.floor
        if floors_above <= 0:
            floors_above = 1
        # Force in Newtons (kN/m² * m² * 1000 N/kN), downward = negative Fz
        Fz = -(total_load_kPa * area_m2 * floors_above) * 1000.0
        loads.setdefault(nid, {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0})
        loads[nid]["Fz"] += Fz

    # ------------------------------------------------------------------
    # 2. Wind loads (Fx horizontal, floor-by-floor)
    # ------------------------------------------------------------------
    if wind_kN_m > 0.0:
        for floor_level in range(1, n_floors + 1):
            floor_nodes = [
                (nid, nd)
                for nid, nd in graph.nodes.items()
                if nd.floor == floor_level
            ]
            if not floor_nodes:
                continue

            # Total wind force per floor [N]
            F_wind_floor_N = wind_kN_m * floor_height * site_width * 1000.0
            Fx_per_node = F_wind_floor_N / len(floor_nodes)
            for nid, _ in floor_nodes:
                loads.setdefault(nid, {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0})
                loads[nid]["Fx"] += Fx_per_node

    # ------------------------------------------------------------------
    # 3. Seismic loads (Fx horizontal)
    # ------------------------------------------------------------------
    if ag_g > 0.0:
        total_area_m2 = sum(trib_areas.values()) if trib_areas else site_width * task_config.site_depth_m
        W_kN = dead_kPa * total_area_m2

        H_m = n_floors * floor_height
        seismic_result = compute_seismic_shear(
            W_kN=W_kN,
            H_m=H_m,
            ag_g=ag_g,
            n_floors=n_floors,
            gamma_I=gamma_I,
        )

        for floor_idx in range(n_floors):
            floor_level = floor_idx + 1
            F_floor_kN = seismic_result.floor_forces_kN[floor_idx]
            floor_nodes = [
                nid
                for nid, nd in graph.nodes.items()
                if nd.floor == floor_level
            ]
            if not floor_nodes:
                continue
            Fx_per_node = (F_floor_kN * 1000.0) / len(floor_nodes)
            for nid in floor_nodes:
                loads.setdefault(nid, {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0})
                loads[nid]["Fx"] += Fx_per_node

    return loads
