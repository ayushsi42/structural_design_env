"""
PGSA FastAPI Server
---------------------
Exposes OpenEnv-compatible HTTP endpoints:
  POST /reset      - Initialize a new episode
  POST /step       - Execute one action
  GET  /state      - Get current episode state
  GET  /health     - Health check
  GET  /tasks      - List available tasks
  GET  /web        - Interactive web UI (when enabled)
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pgsa_env.environment import PGSAEnvironment

# ─── APP INIT ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PGSA — Physically-Grounded Scientific Architect",
    description=(
        "OpenEnv environment where LLM agents design structurally sound, "
        "functionally complete buildings while discovering physics through interaction."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global environment instance (single-user for hackathon; production would use sessions)
_env = PGSAEnvironment()

# ─── REQUEST / RESPONSE MODELS ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "easy"   # "easy" | "medium" | "hard" | task_id
    seed: Optional[int] = None


class StepRequest(BaseModel):
    message: str               # JSON action string from agent


class ResetResponse(BaseModel):
    observation: dict
    info: dict


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


class StateResponse(BaseModel):
    state: dict

# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "pgsa-env", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "id": "stable_shelter",
                "difficulty": "easy",
                "curriculum_level": 1,
                "description": "Build a structurally stable single-room shelter (8×8×8 grid).",
                "budget": 200,
                "action_budget": 500,
                "grader": "grade_easy",
            },
            {
                "id": "functional_office",
                "difficulty": "medium",
                "curriculum_level": 2,
                "description": "Build a 2–3 room functional office with airflow and lighting (16×12×16 grid).",
                "budget": 1000,
                "action_budget": 500,
                "grader": "grade_medium",
            },
            {
                "id": "material_detective",
                "difficulty": "hard",
                "curriculum_level": 3,
                "description": "Build a 4–5 room complex with hidden material properties — probe to infer (16×16×16 grid).",
                "budget": 2000,
                "action_budget": 500,
                "probe_budget": 50,
                "grader": "grade_hard",
            },
        ]
    }


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = ResetRequest()):
    """Start a new episode."""
    try:
        obs = _env.reset(difficulty=request.difficulty, seed=request.seed)
        return ResetResponse(
            observation=obs,
            info={
                "episode_id": obs["info"]["episode_id"],
                "task_description": obs["task_description"],
                "difficulty": request.difficulty,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """Execute one action in the current episode."""
    if _env._done and _env._episode_id == "":
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first."
        )
    try:
        obs, reward, done, info = _env.step(request.message)
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=StateResponse)
async def state():
    """Get current episode state metadata."""
    return StateResponse(state=_env.state())


# ─── INTERACTIVE WEB UI ───────────────────────────────────────────────────────

ENABLE_WEB = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes")


@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    if not ENABLE_WEB:
        return HTMLResponse(
            "<h2>Web interface disabled. Set ENABLE_WEB_INTERFACE=true to enable.</h2>",
            status_code=200
        )
    return HTMLResponse(content=_WEB_UI_HTML)


_WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PGSA — Interactive Environment</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', monospace; background: #0d1117; color: #c9d1d9; height: 100vh; display: flex; flex-direction: column; }
  header { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 20px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 18px; color: #58a6ff; }
  header span { font-size: 12px; color: #8b949e; }
  .main { display: flex; flex: 1; overflow: hidden; }
  .panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; border-right: 1px solid #30363d; }
  .panel-title { background: #161b22; padding: 8px 16px; font-size: 12px; color: #8b949e; border-bottom: 1px solid #30363d; text-transform: uppercase; letter-spacing: 1px; }
  .panel-content { flex: 1; overflow-y: auto; padding: 12px 16px; font-size: 13px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
  .task-select { display: flex; gap: 8px; padding: 12px 16px; background: #161b22; border-top: 1px solid #30363d; }
  select, button, textarea { font-family: inherit; font-size: 13px; border-radius: 6px; border: 1px solid #30363d; background: #21262d; color: #c9d1d9; padding: 6px 12px; }
  button { background: #238636; border-color: #238636; color: #fff; cursor: pointer; }
  button:hover { background: #2ea043; }
  button.danger { background: #da3633; border-color: #da3633; }
  .input-area { display: flex; gap: 8px; padding: 12px 16px; background: #161b22; border-top: 1px solid #30363d; }
  textarea { flex: 1; resize: none; height: 68px; }
  .msg-agent { color: #58a6ff; }
  .msg-env { color: #3fb950; }
  .msg-error { color: #f85149; }
  .msg-reward { color: #d29922; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; margin-left: 6px; }
  .badge-easy { background: #238636; }
  .badge-medium { background: #9a6700; }
  .badge-hard { background: #da3633; }
</style>
</head>
<body>
<header>
  <h1>⚡ PGSA Environment</h1>
  <span>Physically-Grounded Scientific Architect — OpenEnv Interactive Shell</span>
</header>
<div class="main">
  <div class="panel" style="max-width:400px">
    <div class="panel-title">🏗 Agent Actions</div>
    <div class="panel-content" id="history">Welcome! Start by clicking Reset to begin an episode.</div>
    <div class="task-select">
      <select id="difficulty">
        <option value="easy">🟢 Easy — Stable Shelter</option>
        <option value="medium">🟡 Medium — Functional Office</option>
        <option value="hard">🔴 Hard — Material Detective</option>
      </select>
      <button onclick="doReset()">↺ Reset</button>
    </div>
    <div class="input-area">
      <textarea id="action-input" placeholder='{"action_type": "PLACE_ELEMENT", "x": 2, "y": 1, "z": 2, "element_type": "WALL", "material_id": 0}'></textarea>
      <button onclick="doStep()">▶ Step</button>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">🔬 Environment State</div>
    <div class="panel-content" id="state-panel">No episode active.</div>
  </div>
</div>

<script>
const hist = document.getElementById('history');
const statePanel = document.getElementById('state-panel');
const actionInput = document.getElementById('action-input');

function append(el, text, cls) {
  const d = document.createElement('div');
  d.className = cls || '';
  d.textContent = text;
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
}

async function doReset() {
  hist.innerHTML = '';
  const diff = document.getElementById('difficulty').value;
  try {
    const r = await fetch('/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({difficulty: diff})});
    const data = await r.json();
    append(hist, `[RESET] Task: ${diff}`, 'msg-env');
    append(hist, data.observation.message, 'msg-env');
    updateState();
  } catch(e) { append(hist, `Error: ${e}`, 'msg-error'); }
}

async function doStep() {
  const msg = actionInput.value.trim();
  if (!msg) return;
  append(hist, `➤ ${msg}`, 'msg-agent');
  try {
    const r = await fetch('/step', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: msg})});
    const data = await r.json();
    append(hist, data.observation.message, data.done ? 'msg-reward' : 'msg-env');
    append(hist, `reward: ${data.reward >= 0 ? '+' : ''}${data.reward.toFixed(4)} | done: ${data.done}`, 'msg-reward');
    if (data.done && data.info.graded_score !== null) {
      append(hist, `🏆 FINAL SCORE: ${(data.info.graded_score * 100).toFixed(1)}%`, 'msg-reward');
    }
    updateState();
  } catch(e) { append(hist, `Error: ${e}`, 'msg-error'); }
}

async function updateState() {
  try {
    const r = await fetch('/state');
    const data = await r.json();
    const s = data.state;
    statePanel.textContent = JSON.stringify(s, null, 2);
  } catch(e) {}
}

actionInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) doStep();
});

// Pre-fill example actions
const examples = [
  '{"action_type": "PLACE_ELEMENT", "x": 1, "y": 1, "z": 1, "element_type": "WALL", "material_id": 0}',
  '{"action_type": "ANNOTATE_ROOM", "x1": 1, "y1": 1, "z1": 1, "x2": 4, "y2": 3, "z2": 4, "room_type": "STORAGE"}',
  '{"action_type": "COMMIT_DESIGN"}',
];
let exIdx = 0;
document.addEventListener('keydown', e => {
  if (e.altKey && e.key === 'e') {
    actionInput.value = examples[exIdx % examples.length];
    exIdx++;
  }
});
</script>
</body>
</html>"""
