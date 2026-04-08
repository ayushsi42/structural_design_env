"""
FastAPI server for StructuralDesignEnv (OpenEnv HTTP interface).

Endpoints:
  GET  /health
  GET  /tasks
  POST /reset
  POST /step
  GET  /state?session_id=...
  GET  /action_schema
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import os
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

@app.get("/health")
def health():
    return {"status": "ok", "env": "StructuralDesignEnv", "version": "1.0.0"}


@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
def serve_demo():
    demo_path = os.path.join(os.path.dirname(__file__), "demo.html")
    if os.path.exists(demo_path):
        with open(demo_path, "r", encoding="utf-8") as f:
            return f.read()
    return "demo.html not found."


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
def reset_env(req: Optional[ResetRequest] = None):
    req = req or ResetRequest()
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


def _task_description(tid: str) -> str:
    descriptions = {
        "task1_warehouse": "Single-story 20×10m warehouse. No lateral loads. Score by validity + material efficiency.",
        "task2_office": "3-story 20×20m office with wind and light seismic. Score by drift + efficiency + torsional balance.",
        "task3_hospital": "3-story hospital in seismic Zone 3. Score by seismic drift + budget + redundancy + utilisation.",
    }
    return descriptions.get(tid, "")
