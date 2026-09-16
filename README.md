# StructuralDesignEnv
*An OpenEnv RL environment where LLM agents design steel building frames, graded by a real 3D direct-stiffness solver and Eurocode 3 checks — not heuristics.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-96%2F96%20passing-brightgreen.svg)](tests/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#current-status)

## Overview

StructuralDesignEnv is an [OpenEnv](https://pypi.org/project/openenv-core/) reinforcement-learning environment built for OpenEnv Hackathon Round 1, in which an LLM agent plays structural engineer: it places HEB columns, IPE beams, and concrete shear walls on a building grid and must converge on a design that survives real loads. Unlike toy RL environments, the grader is a genuine engineering pipeline — a 3D, 6-degree-of-freedom direct-stiffness-method (DSM) solver assembles and solves the structure's global stiffness matrix, then every member is checked against EN 1993-1-1 (Eurocode 3: bending, shear, flexural buckling, deflection), with EN 1998-1 seismic base-shear loading applied on the hardest task. There are three graded tasks of increasing difficulty (warehouse → office → hospital), each with its own weighted scoring formula over validity, efficiency, drift control, and redundancy. The environment is served over HTTP via FastAPI so it can be driven by any LLM through an OpenAI-compatible chat completion API.

## Key Features

- **Real physics, not heuristics** — a sparse 3D direct-stiffness solver (`solver/stiffness_matrix.py`) assembles 12×12 element stiffness matrices (Euler-Bernoulli beam theory, 6 DOF/node) and solves `Ku = F`; validated in `tests/test_stiffness.py` against known structural behaviors (axial compression, cantilever/portal-frame bending, lateral sway).
- **Eurocode 3 (EN 1993-1-1) member checks** — bending, shear, flexural buckling (buckling curve b, braced vs. sway effective-length factors), combined axial+bending interaction, and beam deflection limits, all against a real HEB/IPE section database with true Eurocode section properties (`solver/sections.py`).
- **EN 1998-1 seismic loading** for the hospital task — Type 1 elastic response spectrum, soil class C, inverted-triangle floor force distribution, base shear from `ag`/importance factor (`solver/seismic.py`).
- **Three tiered tasks with distinct grading rubrics** — warehouse (validity + efficiency), office (drift control + torsional balance + efficiency), hospital (validity + budget efficiency + redundancy + utilization), plus a progressive-collapse redundancy check (`solver/redundancy.py`).
- **Full OpenEnv-compliant HTTP server** — FastAPI app (`server/app.py`) exposing `/health`, `/metadata`, `/schema`, `/mcp` (JSON-RPC), plus simulation endpoints (`/reset`, `/step`, `/grade`) and research tooling (`/query_forces`, `/what_if_remove`, `/render` for SVG floor-plan visualization).
- **96/96 tests passing** across solver mechanics, Eurocode checks, task graders, full-episode integration, and server routes (see [Current Status](#current-status)).
- **Docker-ready** for one-command deployment (e.g. to a Hugging Face Space) with a health-checked container image.

## How It Works

Each episode is driven by a step loop: the agent sends a JSON action, the environment updates its structural graph, re-solves the physics, re-checks the code compliance, and returns a natural-language + structured observation the agent can reason over.

```
LLM agent
   │  {"action_type": "place_column", "grid_x":5, "grid_y":0, "floor":0, "section":"HEB200"}
   ▼
StructuralDesignEnv.step()          (structural_design_env/env.py)
   │  1. validate_action()          — geometry/adjacency/section checks (validation.py)
   │  2. graph.place_*()            — update node/element graph (graph.py, grid.py)
   │  3. generate_loads()           — tributary-area DL/LL + wind + seismic (solver/load_generator.py)
   │  4. StructuralSolver.solve()   — assemble & solve global 6-DOF stiffness matrix (solver/stiffness_matrix.py)
   │  5. check_member()             — EN 1993-1-1 bending/shear/buckling/deflection per member (solver/eurocode3.py)
   │  6. compute_reward()           — 3-stage shaped reward: member fixes → global drift/violations → terminal efficiency (reward.py)
   ▼
Observation (UR per member, drift ratio, mass, violations) + reward + done
   │  on "done" →  task-specific grader (tasks/task{1,2,3}_*.py) → graded_score in [0, 1]
```

## Project Structure

```
structural_design_env/
├── server/
│   ├── app.py                    # Canonical FastAPI server (OpenEnv /metadata, /schema, /mcp + sim + research endpoints)
│   └── interactive_demo.html     # Browser demo UI served at /demo
├── server.py                     # Legacy/simplified standalone server (superseded by server/app.py)
├── inference.py                  # Baseline LLM-agent runner (root — hackathon entry point)
├── scripts/inference.py          # Alternate inference script (Anthropic/OpenAI env-var conventions)
├── openenv.yaml                  # OpenEnv manifest (tasks, graders, runtime)
├── pyproject.toml                # Package metadata & dependencies
├── Dockerfile                    # Container image (port 7860, runs server.app:app)
├── structural_design_env/
│   ├── env.py                    # StructuralDesignEnv: reset/step/state
│   ├── models.py                 # Pydantic: StructuralAction, StructuralObservation, TaskConfig
│   ├── graph.py                  # Node/element graph (columns, beams, walls)
│   ├── grid.py                   # 2D plan-grid occupancy state
│   ├── reward.py                 # Three-stage shaped physics reward
│   ├── validation.py             # Action geometry/adjacency validator
│   ├── solver/
│   │   ├── stiffness_matrix.py   # 3D direct-stiffness method (6 DOF/node, 12×12 elements)
│   │   ├── load_generator.py     # Tributary-area dead/live load + wind + seismic loads
│   │   ├── eurocode3.py          # EN 1993-1-1 member checks
│   │   ├── seismic.py            # EN 1998-1 response spectrum & base shear
│   │   ├── redundancy.py         # Progressive-collapse redundancy check
│   │   └── sections.py           # HEB/IPE section database (real Eurocode properties)
│   └── tasks/
│       ├── task1_warehouse.py    # Easy: single-story, gravity-only
│       ├── task2_office.py       # Medium: 3-story, wind + light seismic
│       └── task3_hospital.py     # Hard: 3-story, seismic Zone 3, redundancy
└── tests/
    ├── test_stiffness.py         # Solver unit tests (31 tests)
    ├── test_eurocode.py          # Eurocode check unit tests (15 tests)
    ├── test_graders.py           # Grader determinism/range tests (14 tests)
    ├── test_env.py               # Full-episode integration tests (32 tests)
    └── test_server_routes.py     # FastAPI route tests (4 tests)
```

## Getting Started

### Tasks & Grading

| Task | Difficulty | Site | Loads | Max Steps | Score Formula |
|------|-----------|------|-------|-----------|---------------|
| `task1_warehouse` | Easy | 20×10m, 1 floor | DL+LL only | 25 | `0.40·validity + 0.35·efficiency + 0.25·utilization_quality` |
| `task2_office` | Medium | 20×20m, 3 floors | Wind 1.5 kN/m, light seismic | 55 | `0.35·validity + 0.30·drift_control + 0.20·torsional_balance + 0.15·efficiency` |
| `task3_hospital` | Hard | 20×20m, 3 floors | Wind 2.0 kN/m, seismic Zone 3 (ag=0.25g, γI=1.5) | 85 | `0.30·validity + 0.30·budget_efficiency + 0.25·redundancy + 0.15·utilization` |

### API Endpoints (`server/app.py`, the canonical server run by Docker)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/metadata` | OpenEnv metadata (name, description) |
| GET | `/schema` | Action/observation/state JSON schema |
| POST | `/mcp` | JSON-RPC 2.0 interface |
| GET | `/tasks` | List all 3 tasks |
| POST | `/reset` | Start new episode `{"task_id": "task1_warehouse"}` |
| POST | `/step` | Execute an action `{"session_id": "...", "message": "{...json...}"}` |
| POST | `/grade` | Explicitly grade the current episode state |
| GET | `/state` | Episode state `?session_id=...` |
| GET | `/action_schema` | Human-readable action reference |
| GET | `/query_forces` | Member forces for a placed element |
| POST | `/what_if_remove` | Simulate removing an element without committing |
| GET | `/render` | SVG floor-plan visualization, colored by utilization ratio |
| GET | `/demo` | Browser-based interactive demo UI |

### Requirements

- Python 3.10+
- `numpy`, `scipy`, `fastapi`, `uvicorn`, `pydantic`, `httpx`, `openai`, `python-dotenv` (installed automatically)

### Installation

```bash
git clone https://github.com/ayushsi42/structural_design_env
cd structural_design_env
pip install -e ".[dev]"
```

### Usage

**Run the test suite:**
```bash
pytest tests/ -q
```

**Start the server locally:**
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
# or, via the console-script entry point:
server
```

**Run the baseline LLM agent against it:**
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
python inference.py task1_warehouse   # or task2_office / task3_hospital
# with no argument, inference.py runs all 3 tasks in sequence
```

**Docker:**
```bash
docker build -t structural-design-env .
docker run -p 7860:7860 structural-design-env
curl http://localhost:7860/health
```

## Current Status

Verified locally on 2026-09-16 (`pip install -e ".[dev]"` then `pytest tests/ -q`): **96/96 tests pass, 0 failures** — 31 solver-mechanics tests, 15 Eurocode-check tests, 14 grader tests, 32 full-episode integration tests, and 4 FastAPI route tests. The solver tests exercise real structural behavior (axial compression, cantilever and portal-frame bending, lateral sway, multi-bay grid frames) rather than only checking that the code runs, and the Eurocode tests confirm utilization ratios cross 1.0 exactly at the analytically expected capacity. One benign `MatrixRankWarning` is emitted (expected: a deliberately disconnected/unsupported test structure is exactly singular, which the solver correctly detects and reports as non-converged).

The repo carries **two server implementations**: `server/app.py` is the actively developed, OpenEnv-compliant one (used by the Dockerfile and the `server` console script); root-level `server.py` is an earlier, simpler version kept for reference and is not what ships in the container. Similarly, root `inference.py` (OpenAI-style env vars, hackathon entry point) and `scripts/inference.py` (Anthropic/OpenAI-agnostic env vars) overlap in purpose — root `inference.py` is the one referenced by the packaging metadata. The HF Space YAML front-matter and git history (`Add root / endpoint to fix HF Space 404`, `Add HTML landing page for HF Space App tab`) indicate this was deployed to a Hugging Face Space at some point during development; no HF remote is configured in this local checkout, so current live status of that Space was not verified as part of this pass. No end-to-end LLM-agent run with logged scores is checked into the repo yet — `inference.py` exists and runs episodes, but a recorded baseline result (which model, which task, what score) is not yet published.

## Roadmap

Near-term priorities are proving the environment out with an actual LLM baseline run and tightening the project down to one canonical server/inference entry point; medium-term work extends the physics (more section types, P-delta effects) and grading depth; longer-term work aims at a public leaderboard across multiple LLMs. See [ROADMAP.md](ROADMAP.md) for the full plan.

## Tech Stack

Python 3.10+, NumPy, SciPy (sparse linear solve), FastAPI, Pydantic, Uvicorn, OpenAI SDK (LLM client), pytest, Docker.

## License

MIT — see [LICENSE](LICENSE).

## Author

Ayush Singh — [GitHub](https://github.com/ayushsi42) · [LinkedIn](https://www.linkedin.com/in/ayush-singh-40539522b/) · ayushsingh73920@gmail.com
