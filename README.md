# StructuralDesignEnv

> **OpenEnv Hackathon Round 1 Submission**
> An RL environment where LLM agents design structurally sound steel building frames using real structural engineering physics (Eurocode 3, direct stiffness method).

---

## Overview

StructuralDesignEnv is a physics-informed [OpenEnv](https://pypi.org/project/openenv-core/) environment where an AI agent acts as a structural engineer. Given a building task specification, the agent:

1. **Places structural elements** — columns (HEB sections), beams (IPE sections), shear walls
2. **Receives physics analysis** — utilization ratios, deflections, lateral drift from the Direct Stiffness Method
3. **Upgrades/downgrades sections** based on Eurocode 3 compliance checks
4. **Signals done** when all members satisfy code requirements

The solver runs **real structural analysis** (not heuristics): assembles a 6-DOF sparse stiffness matrix, solves Ku=F, checks bending, shear, flexural buckling, and deflection per EN 1993-1-1.

---

## Tasks

| Task | Difficulty | Site | Loads | Max Steps | Description |
|------|-----------|------|-------|-----------|-------------|
| `task1_warehouse` | Easy | 20×10m, 1 floor | DL+LL only, no lateral | 25 | Single-story warehouse — place column grid, add beams |
| `task2_office` | Medium | 20×20m, 3 floors | Wind 1.5 kN/m, light seismic | 55 | Multi-story office — control drift + efficiency |
| `task3_hospital` | Hard | 20×20m, 3 floors | Wind 2.0 kN/m, seismic Zone 3 (ag=0.25g, γI=1.5) | 85 | Seismic hospital — progressive collapse, budget, redundancy |

### Grading (all tasks return scores in 0.0–1.0)

**Task 1 (Warehouse):**
```
score = 0.40 × validity + 0.35 × efficiency + 0.25 × utilization_quality
```

**Task 2 (Office):**
```
score = 0.35 × validity + 0.30 × drift_control + 0.20 × torsional_balance + 0.15 × efficiency
```

**Task 3 (Hospital):**
```
score = 0.30 × validity + 0.30 × budget_efficiency + 0.25 × redundancy + 0.15 × utilization
```

---

## Action Space

Actions are sent as JSON in the `message` field of `/step`. Supported `action_type` values:

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `place_column` | `grid_x`, `grid_y`, `floor`, `section` | Place a steel column (HEB140–HEB400) |
| `place_beam` | `node_i`, `node_j`, `section` | Place a beam between two column nodes (IPE200–IPE500) |
| `place_wall` | `node_i`, `node_j` | Place a concrete shear wall panel |
| `upgrade_section` | `element_id` | Upgrade element to next larger section |
| `downgrade_section` | `element_id` | Downgrade element to next smaller section |
| `done` | — | Finalize design and trigger grader |

**Column sections (HEB):** HEB140, HEB160, HEB200, HEB240, HEB300, HEB360, HEB400

**Beam sections (IPE):** IPE200, IPE240, IPE300, IPE360, IPE400, IPE450, IPE500

### Example Actions

```json
{"action_type": "place_column", "grid_x": 0, "grid_y": 0, "floor": 0, "section": "HEB200"}

{"action_type": "place_beam", "from_node_x": 0, "from_node_y": 0, "to_node_x": 5, "to_node_y": 0, "floor": 0, "section": "IPE300", "orientation": "x"}

{"action_type": "place_wall", "from_node_x": 0, "from_node_y": 0, "to_node_x": 0, "to_node_y": 5, "floor": 0}

{"action_type": "upgrade_section", "element_id": "col_n_0_0_0_n_0_0_1"}

{"action_type": "done"}
```

---

## Observation Space

Each step returns a JSON object with:

| Field | Type | Description |
|-------|------|-------------|
| `message` | str | Natural-language physics summary for the agent |
| `task_description` | str | Full task specification |
| `grid_plan` | list | 2D grid showing column positions |
| `placed_elements` | list | All elements with section names |
| `critical_members` | list | Members with highest utilization ratios |
| `n_code_violations` | int | Number of members with UR > 1.0 |
| `max_utilization_ratio` | float | Worst member demand/capacity ratio |
| `max_deflection_mm` | float | Maximum beam deflection |
| `lateral_drift_ratio` | float | Max story drift / (height/500) |
| `is_structurally_valid` | bool | All URs < 1.0 and drift OK |
| `total_steel_mass_kg` | float | Total steel mass placed |
| `step_count` | int | Current step number |
| `last_action_error` | str\|null | Validation error if action was rejected |

---

## Reward Function

Three-stage physics reward (clipped to −1.0–1.0):

```
Stage 1 (member physics):
  +0.05 per violation fixed, −0.05 per new violation, +0.10 if fully compliant

Stage 2 (global structure):
  +0.30 × (1 − drift_ratio) if drift improved
  +0.20 if zero violations
  −0.15 per invalid action

Stage 3 (efficiency + terminal):
  +0.15 × (1 − mass/reference_mass) on "done" if valid
  −0.30 × failure_severity on "done" if invalid
```

---

## Setup & Usage

### Requirements

- Python 3.10+
- `pip install -e .` (installs numpy, scipy, fastapi, pydantic, uvicorn)

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Yes | Model identifier (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | Yes | Hugging Face / OpenAI API key |
| `TASK_ID` | No | Task to run (default: `task1_warehouse`) |
| `SPACE_URL` | No | Server URL (default: `http://localhost:7860`) |

### Local Development

```bash
# Clone & install
git clone <repo>
cd structural_design_env
pip install -e .

# Start server
uvicorn server:app --host 0.0.0.0 --port 7860

# In another terminal — run inference
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
python inference.py task1_warehouse
```

### Docker

```bash
# Build
docker build -t structural-design-env .

# Run server
docker run -p 7860:7860 structural-design-env

# Test health
curl http://localhost:7860/health

# Run inference against running server
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
python inference.py
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check → `{"status": "ok"}` |
| GET | `/tasks` | List all 3 tasks |
| POST | `/reset` | Start new episode `{"task_id": "task1_warehouse"}` |
| POST | `/step` | Execute action `{"session_id": "...", "message": "{...json...}"}` |
| GET | `/state` | Current episode state `?session_id=...` |
| GET | `/action_schema` | Full action JSON schema |
| GET | `/docs` | OpenAPI documentation |

---

## Project Structure

```
structural-design-env/
├── inference.py                      # Baseline inference script (root — required)
├── server.py                         # FastAPI server
├── openenv.yaml                      # OpenEnv manifest
├── pyproject.toml                    # Package dependencies
├── Dockerfile                        # Container image (port 7860)
├── structural_design_env/
│   ├── env.py                        # Main environment: reset/step/state
│   ├── models.py                     # Pydantic: Action, Observation, TaskConfig
│   ├── graph.py                      # Structural graph (nodes + elements)
│   ├── grid.py                       # 2D grid state
│   ├── reward.py                     # Three-stage physics reward
│   ├── validation.py                 # Action geometry validator
│   ├── solver/
│   │   ├── stiffness_matrix.py       # 3D Direct Stiffness (6-DOF, 12×12 elements)
│   │   ├── load_generator.py         # Tributary area + wind + seismic loads
│   │   ├── eurocode3.py              # EN 1993-1-1: bending, shear, buckling
│   │   ├── seismic.py                # EN 1998-1: Type 1 spectrum, base shear
│   │   ├── redundancy.py             # Progressive collapse check
│   │   └── sections.py               # HEB/IPE section database (real EC values)
│   └── tasks/
│       ├── task1_warehouse.py        # Easy task config + grader
│       ├── task2_office.py           # Medium task config + grader
│       └── task3_hospital.py         # Hard task config + grader
└── tests/
    ├── test_stiffness.py             # Solver unit tests (analytical solutions)
    ├── test_eurocode.py              # Eurocode check unit tests
    ├── test_graders.py               # Grader determinism + range tests
    └── test_env.py                   # Integration: full episode tests
```

---

## Physics Details

### Direct Stiffness Method (3D, 6 DOF per node)

The solver assembles a sparse global stiffness matrix with **6 degrees of freedom per node**: [ux, uy, uz, rx, ry, rz] (translations + rotations in x=East, y=North, z=Up). Each element produces a 12×12 local stiffness matrix (Euler-Bernoulli beam theory) transformed to global coordinates via element-type rotation matrices.

### Eurocode 3 (EN 1993-1-1) Checks

Every member is checked for:
- **Bending** — M_Ed / M_c,Rd ≤ 1.0
- **Shear** — V_Ed / V_pl,Rd ≤ 1.0
- **Flexural buckling** — N_Ed / (χ × N_pl,Rd) ≤ 1.0 (buckling curve b, columns only)
- **Combined axial + bending** — interaction formula
- **Deflection** — δ_max ≤ L/300 (beams only)

### Seismic Loading (Task 3)

EN 1998-1 Type 1 elastic spectrum, soil class C, ag=0.25g, γI=1.5. Base shear distributed via inverted-triangle floor force distribution.

---

## License

MIT
