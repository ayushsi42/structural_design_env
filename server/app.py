"""
FastAPI server for StructuralDesignEnv (OpenEnv HTTP interface).

Endpoints:
  GET  /health
  GET  /tasks
  POST /reset
  POST /step
  GET  /state?session_id=...
  GET  /action_schema
  GET  /query_forces?session_id=...&element_id=...
  POST /what_if_remove
  GET  /render?session_id=...&floor=0
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from structural_design_env.env import StructuralDesignEnv
from structural_design_env.tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, StructuralDesignEnv] = {}

    def create(self, session_id: Optional[str] = None) -> tuple[str, StructuralDesignEnv]:
        sid = session_id or str(uuid.uuid4())
        env = StructuralDesignEnv()
        self._sessions[sid] = env
        return sid, env

    def get(self, session_id: str) -> Optional[StructuralDesignEnv]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str]) -> tuple[str, StructuralDesignEnv]:
        if session_id and session_id in self._sessions:
            return session_id, self._sessions[session_id]
        return self.create(session_id)


session_manager = SessionManager()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="StructuralDesignEnv",
    description="OpenEnv API for steel frame structural engineering RL environment.",
    version="1.0.0",
)

DEMO_HTML_PATH = Path(__file__).with_name("interactive_demo.html")

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1_warehouse"
    session_id: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    session_id: Optional[str] = None
    message: str  # JSON-encoded StructuralAction


class ResetResponse(BaseModel):
    session_id: str
    observation: dict


class StepResponse(BaseModel):
    session_id: str
    observation: dict
    reward: float
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _load_demo_html() -> str:
    return DEMO_HTML_PATH.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def root():
    return _load_demo_html()


@app.get("/demo", response_class=HTMLResponse)
def demo():
    return _load_demo_html()


@app.get("/health")
def health():
    return {"status": "ok", "env": "StructuralDesignEnv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    tasks = []
    for tid, (cfg, _) in TASK_REGISTRY.items():
        tasks.append({
            "id": tid,
            "name": cfg.name,
            "difficulty": cfg.difficulty,
            "max_steps": cfg.max_steps,
            "n_floors": cfg.n_floors,
            "site_width_m": cfg.site_width_m,
            "site_depth_m": cfg.site_depth_m,
            "description": _task_description(tid),
        })
    return {"tasks": tasks}


@app.post("/reset", response_model=ResetResponse)
def reset_env(req: ResetRequest):
    if req.task_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Valid: {list(TASK_REGISTRY)}",
        )
    sid, env = session_manager.get_or_create(req.session_id)
    obs = env.reset(task_id=req.task_id, seed=req.seed)
    return ResetResponse(session_id=sid, observation=obs)


@app.post("/step", response_model=StepResponse)
def step_env(req: StepRequest):
    sid, env = session_manager.get_or_create(req.session_id)
    obs, reward, done, info = env.step(req.message)
    return StepResponse(
        session_id=sid,
        observation=obs,
        reward=round(reward, 6),
        done=done,
        info=info,
    )


@app.get("/state")
def get_state(session_id: Optional[str] = None):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    env = session_manager.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return env.state()


@app.get("/action_schema")
def action_schema():
    return {
        "description": "Actions for StructuralDesignEnv. Send as JSON string in the 'message' field of /step.",
        "actions": [
            {
                "action_type": "place_column",
                "description": "Place a steel column at a grid position on a floor.",
                "fields": {
                    "grid_x": "int [0-19] — horizontal grid position (1 cell = 1m)",
                    "grid_y": "int [0-19] — depth grid position",
                    "floor": "int [0..n_floors-1] — floor index (0=ground)",
                    "section": "str — one of HEB140, HEB160, HEB200, HEB240, HEB300, HEB360, HEB400",
                },
                "example": {
                    "action_type": "place_column",
                    "grid_x": 5,
                    "grid_y": 0,
                    "floor": 0,
                    "section": "HEB200",
                },
            },
            {
                "action_type": "place_beam",
                "description": "Place a steel beam connecting two column nodes at the same floor.",
                "fields": {
                    "from_node_x": "int — x of start column",
                    "from_node_y": "int — y of start column",
                    "to_node_x": "int — x of end column (must share x OR y with start)",
                    "to_node_y": "int — y of end column",
                    "floor": "int — floor where both columns sit",
                    "section": "str — one of IPE200, IPE240, IPE300, IPE360, IPE400, IPE450, IPE500",
                    "orientation": "'x' (horizontal) or 'y' (depth direction)",
                },
                "example": {
                    "action_type": "place_beam",
                    "from_node_x": 0,
                    "from_node_y": 0,
                    "to_node_x": 5,
                    "to_node_y": 0,
                    "floor": 0,
                    "section": "IPE300",
                    "orientation": "x",
                },
            },
            {
                "action_type": "upgrade_section",
                "description": "Upgrade an element to the next larger standard section.",
                "fields": {
                    "element_id": "str — element ID from placed_elements list (e.g. 'col_5_0_0')",
                },
                "example": {"action_type": "upgrade_section", "element_id": "col_5_0_0"},
            },
            {
                "action_type": "downgrade_section",
                "description": "Downgrade an element to the next smaller standard section.",
                "fields": {
                    "element_id": "str — element ID",
                },
                "example": {"action_type": "downgrade_section", "element_id": "beam_0_0_5_0_0"},
            },
            {
                "action_type": "remove_element",
                "description": "Remove an element from the structure.",
                "fields": {"element_id": "str — element ID"},
                "example": {"action_type": "remove_element", "element_id": "col_5_0_0"},
            },
            {
                "action_type": "add_wall",
                "description": "Add a concrete shear wall between two column nodes.",
                "fields": {
                    "from_node_x": "int",
                    "from_node_y": "int",
                    "to_node_x": "int",
                    "to_node_y": "int",
                    "floor": "int",
                    "thickness_m": "float — 0.2 or 0.3",
                    "orientation": "'x' or 'y'",
                },
                "example": {
                    "action_type": "add_wall",
                    "from_node_x": 0,
                    "from_node_y": 0,
                    "to_node_x": 5,
                    "to_node_y": 0,
                    "floor": 0,
                    "thickness_m": 0.2,
                    "orientation": "x",
                },
            },
            {
                "action_type": "done",
                "description": "Signal that the design is complete. Triggers final grading.",
                "example": {"action_type": "done"},
            },
        ],
        "sections": {
            "columns": ["HEB140", "HEB160", "HEB200", "HEB240", "HEB300", "HEB360", "HEB400"],
            "beams": ["IPE200", "IPE240", "IPE300", "IPE360", "IPE400", "IPE450", "IPE500"],
        },
    }


# ---------------------------------------------------------------------------
# Research endpoints
# ---------------------------------------------------------------------------

@app.get("/query_forces")
def query_forces(session_id: str, element_id: str):
    """Return solver forces (N, V, M_max, delta_max_mm) for a specific element."""
    env = session_manager.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    sr = env._last_solver_result
    if sr is None or not sr.converged:
        raise HTTPException(status_code=400, detail="No converged solver result available")
    forces = sr.member_forces.get(element_id)
    if forces is None:
        raise HTTPException(status_code=404, detail=f"Element '{element_id}' not found in solver result")
    elem = env.graph.elements.get(element_id)
    return {
        "element_id": element_id,
        "element_type": elem.element_type if elem else "unknown",
        "section": elem.section if elem else "unknown",
        "length_m": elem.length_m if elem else None,
        "forces": {
            "N_kN": round(forces.get("N", 0.0) / 1000.0, 3),
            "V_kN": round(forces.get("V", 0.0) / 1000.0, 3),
            "M_max_kNm": round(forces.get("M_max", 0.0) / 1000.0, 3),
            "delta_max_mm": round(forces.get("delta_max_mm", 0.0), 3),
        },
    }


class WhatIfRemoveRequest(BaseModel):
    session_id: str
    element_id: str


@app.post("/what_if_remove")
def what_if_remove(req: WhatIfRemoveRequest):
    """
    Re-solve the frame without the specified element and return the change in max UR.
    Useful for identifying which members are critical to structural performance.
    """
    env = session_manager.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found")
    if req.element_id not in env.graph.elements:
        raise HTTPException(status_code=404, detail=f"Element '{req.element_id}' not found")

    # Current max UR
    current_obs = env._current_obs
    current_max_UR = max(
        (m.max_UR for m in current_obs.critical_members), default=0.0
    ) if current_obs else 0.0

    # Clone graph, remove element, re-solve
    graph_copy = env.graph.copy()
    graph_copy.remove_element(req.element_id)

    if len(graph_copy.elements) == 0:
        return {
            "element_id": req.element_id,
            "current_max_UR": current_max_UR,
            "counterfactual_max_UR": None,
            "delta_UR": None,
            "verdict": "structure_would_collapse",
        }

    try:
        from structural_design_env.solver import generate_loads, StructuralSolver
        from structural_design_env.solver.sections import COLUMN_SECTIONS, BEAM_SECTIONS
        from structural_design_env.solver import check_member

        loads = generate_loads(graph_copy, env.task_config)
        sr = StructuralSolver().solve(graph_copy, loads)

        if not sr.converged:
            return {
                "element_id": req.element_id,
                "current_max_UR": current_max_UR,
                "counterfactual_max_UR": None,
                "delta_UR": None,
                "verdict": "structure_would_collapse",
            }

        is_braced = any(e.element_type == "wall" for e in graph_copy.elements.values())
        L_eff = 0.7 if is_braced else 1.5
        tc = env.task_config

        new_URs = []
        for eid, elem in graph_copy.elements.items():
            forces = sr.member_forces.get(eid, {"N": 0, "V": 0, "M_max": 0, "delta_max_mm": 0})
            if elem.element_type == "wall":
                chk = check_member("wall", {}, forces, elem.length_m,
                                   floor_height_m=tc.floor_height_m,
                                   thickness_m=elem.thickness_m or 0.2)
            else:
                props = COLUMN_SECTIONS.get(elem.section) or BEAM_SECTIONS.get(elem.section)
                if props is None:
                    continue
                eff = L_eff if elem.element_type == "column" else 1.0
                chk = check_member(elem.element_type, props, forces, elem.length_m, eff)
            new_URs.append(chk.max_UR)

        new_max_UR = max(new_URs, default=0.0)
        delta = round(new_max_UR - current_max_UR, 4)

        return {
            "element_id": req.element_id,
            "current_max_UR": round(current_max_UR, 4),
            "counterfactual_max_UR": round(new_max_UR, 4),
            "delta_UR": delta,
            "verdict": (
                "critical" if new_max_UR > 1.0 and current_max_UR <= 1.0
                else "load_redistributes" if delta > 0.1
                else "redundant"
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _ur_color(ur: float) -> str:
    """Map utilization ratio to hex color."""
    if ur < 0.0:
        return "#9E9E9E"   # grey — not checked
    if ur < 0.60:
        return "#4CAF50"   # green — underutilized
    if ur < 0.85:
        return "#8BC34A"   # light green — good
    if ur < 1.0:
        return "#FFC107"   # amber — near limit
    return "#F44336"       # red — violation


@app.get("/render")
def render_frame(session_id: str, floor: int = 0):
    """
    Return an SVG plan view of the structural frame at the given floor.
    Columns are circles, beams are lines, walls are thick lines.
    Color indicates utilization ratio: green < 0.6 < light-green < 0.85 < amber < 1.0 < red.
    """
    env = session_manager.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    tc = env.task_config
    graph = env.graph
    current_obs = env._current_obs

    # Build UR lookup from last observation
    ur_map: Dict[str, float] = {}
    if current_obs:
        for cm in current_obs.critical_members:
            ur_map[cm.id] = cm.max_UR

    W = tc.site_width_m
    D = tc.site_depth_m
    SVG_SIZE = 520
    MARGIN = 40
    scale = (SVG_SIZE - 2 * MARGIN) / max(W, D)

    def px(x_m: float) -> float:
        return MARGIN + x_m * scale

    def py(y_m: float) -> float:
        return SVG_SIZE - MARGIN - y_m * scale  # flip Y

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_SIZE}" height="{SVG_SIZE}" '
        f'style="background:#1a1a2e;font-family:monospace">',
        # Site boundary
        f'<rect x="{px(0):.1f}" y="{py(D):.1f}" '
        f'width="{W*scale:.1f}" height="{D*scale:.1f}" '
        f'fill="none" stroke="#444" stroke-width="1"/>',
        # Floor label
        f'<text x="{MARGIN}" y="{MARGIN-8}" fill="#aaa" font-size="12">'
        f'Floor {floor} | {tc.name} | step {env.step_count}</text>',
    ]

    # Grid dots (faint)
    for xi in range(0, int(W) + 1, 5):
        for yi in range(0, int(D) + 1, 5):
            lines.append(
                f'<circle cx="{px(xi):.1f}" cy="{py(yi):.1f}" r="1" fill="#333"/>'
            )

    # Elements at this floor
    for eid, elem in graph.elements.items():
        ur = ur_map.get(eid, -1.0)
        color = _ur_color(ur)

        if elem.element_type == "column":
            # Columns span from floor → floor+1; node_i is at `floor` level
            node = graph.nodes.get(elem.node_i)
            if node is None or node.floor != floor:
                continue
            r = max(4.0, scale * 0.4)
            lines.append(
                f'<circle cx="{px(node.x_m):.1f}" cy="{py(node.y_m):.1f}" '
                f'r="{r:.1f}" fill="{color}" stroke="#fff" stroke-width="0.5" '
                f'opacity="0.9"><title>{eid} UR={ur:.3f}</title></circle>'
            )

        elif elem.element_type in ("beam", "wall"):
            # Beams/walls connect nodes at floor+1 level
            ni = graph.nodes.get(elem.node_i)
            nj = graph.nodes.get(elem.node_j)
            if ni is None or nj is None:
                continue
            if ni.floor != floor + 1:
                continue
            sw = 6.0 if elem.element_type == "wall" else 2.5
            opacity = "0.95" if elem.element_type == "wall" else "0.85"
            wall_color = "#9C27B0" if elem.element_type == "wall" else color
            lines.append(
                f'<line x1="{px(ni.x_m):.1f}" y1="{py(ni.y_m):.1f}" '
                f'x2="{px(nj.x_m):.1f}" y2="{py(nj.y_m):.1f}" '
                f'stroke="{wall_color}" stroke-width="{sw}" opacity="{opacity}">'
                f'<title>{eid} UR={ur:.3f}</title></line>'
            )

    # Legend
    legend_items = [
        ("#4CAF50", "UR<0.60"),
        ("#8BC34A", "0.60-0.85"),
        ("#FFC107", "0.85-1.0"),
        ("#F44336", "UR≥1.0"),
        ("#9C27B0", "Wall"),
    ]
    lx = SVG_SIZE - 110
    for i, (col, label) in enumerate(legend_items):
        ly = MARGIN + i * 18
        lines.append(
            f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{col}"/>'
            f'<text x="{lx+16}" y="{ly+10}" fill="#ccc" font-size="10">{label}</text>'
        )

    lines.append("</svg>")
    return Response(content="\n".join(lines), media_type="image/svg+xml")


def _task_description(tid: str) -> str:
    descriptions = {
        "task1_warehouse": "Single-story 20×10m warehouse. No lateral loads. Score by validity + material efficiency.",
        "task2_office": "3-story 20×20m office with wind and light seismic. Score by drift + efficiency + torsional balance.",
        "task3_hospital": "3-story hospital in seismic Zone 3. Score by seismic drift + budget + redundancy + utilisation.",
    }
    return descriptions.get(tid, "")


def main() -> None:
    """Entry point for the OpenEnv server (used by project.scripts)."""
    import os
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
