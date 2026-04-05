# PGSA — Physically-Grounded Scientific Architect

> **OpenEnv Hackathon Round 1 Submission**  
> An RL environment where LLM agents design structurally sound, functionally complete buildings by discovering physical laws through interaction.

---

## Overview

PGSA is a text-based [OpenEnv](https://pypi.org/project/openenv-core/) environment where an AI agent acts as an architect. Given a building task specification, the agent:

1. **Places structural elements** (beams, walls, floors, windows, doors) in a 3D voxel grid
2. **Probes material physics** to infer hidden mechanical properties (yield strength, elastic modulus)
3. **Annotates enclosed rooms** with functional types (OFFICE, BEDROOM, KITCHEN…)
4. **Commits the design** when done — triggering full grader evaluation

The environment enforces real structural physics principles heuristically:
- Elements must be **connected to the foundation** or they fail
- Tall structures under high loads require **stronger materials**
- Rooms need **doors** (access), **windows** (light), and **airflow paths** (ventilation)
- Egress distances must be within NFPA 101-inspired limits

---

## Why PGSA?

Most RL environments for LLMs test language/reasoning in isolation. PGSA is unique because:
- **Physics is discovered, not taught** — the agent never receives symbolic laws
- **Hidden material properties** at hard difficulty force the agent to perform scientific inquiry via `PROBE_PHYSICS` actions
- **Dense reward signal** guides every step — structural, functional, physics accuracy, cost, and efficiency components
- **Real-world task** modeling genuine architectural design trade-offs

---

## Tasks

| Task | Difficulty | Grid | Description | Score Range |
|------|-----------|------|-------------|-------------|
| `stable_shelter` | 🟢 Easy | 8×8×8 | Build a structurally stable single-room shelter | 0.0 → 1.0 |
| `functional_office` | 🟡 Medium | 16×12×16 | Design 2–3 rooms with airflow, lighting, connectivity | 0.0 → 1.0 |
| `material_detective` | 🔴 Hard | 16×16×16 | Build 4–5 rooms with hidden material properties | 0.0 → 1.0 |

### Task Grading

**Easy grader** — `grade_easy(episode_state: dict) → float`:
```
score = 0.5 × stability_score + 0.5 × room_score
```
- `stability_score`: 1.0 if STABLE, 0.4 if SERVICEABILITY_FAIL, 0.0 if COLLAPSE
- `room_score`: 1.0 if ≥1 valid enclosed room with door, 0.0 otherwise

**Medium grader** — `grade_medium(episode_state: dict) → float`:
```
score = 0.35 × room_completion
      + 0.20 × connectivity
      + 0.15 × airflow
      + 0.15 × lighting
      + 0.10 × egress
      + 0.05 × density
      ± structural_bonus
```

**Hard grader** — `grade_hard(episode_state: dict) → float`:
Same as medium but reduced weight + physics accuracy bonus proportional to PROBE_PHYSICS usage.

---

## Action Space

Actions are sent as JSON strings in the `message` field. Supported `action_type` values:

| Action | Key Parameters | Description |
|--------|---------------|-------------|
| `PLACE_ELEMENT` | x, y, z, element_type, material_id, orientation | Place one structural element |
| `REMOVE_ELEMENT` | x, y, z | Remove element at position |
| `REPLACE_MATERIAL` | x, y, z, new_material_id | Swap material of existing element |
| `PLACE_BATCH` | x, y, z, motif_id, scale | Place predefined structural motif |
| `PROBE_PHYSICS` | x, y, z, load_kn, direction | Test-load an element to observe stress/deformation |
| `ANNOTATE_ROOM` | x1,y1,z1, x2,y2,z2, room_type | Flood-fill a room from bounding box |
| `QUERY_BELIEF` | — | Get physics belief state summary |
| `COMMIT_DESIGN` | — | Finalize design, trigger grader evaluation |

**Elements**: BEAM, WALL, FLOOR, WINDOW, DOOR, JOINT  
**Materials**: 0=WOOD(1), 1=CONCRETE(3), 2=STEEL(8), 3=GLASS(5), 4=COMPOSITE(20) ← (cost/voxel)  
**Room types**: OFFICE, BEDROOM, LIVING, KITCHEN, BATHROOM, STORAGE, CORRIDOR, GENERIC

### Example Actions

```json
{"action_type": "PLACE_ELEMENT", "x": 3, "y": 1, "z": 3, "element_type": "WALL", "material_id": 0, "orientation": 0}

{"action_type": "PLACE_ELEMENT", "x": 3, "y": 1, "z": 6, "element_type": "DOOR", "material_id": 0}

{"action_type": "ANNOTATE_ROOM", "x1": 2, "y1": 1, "z1": 2, "x2": 6, "y2": 4, "z2": 7, "room_type": "OFFICE"}

{"action_type": "PROBE_PHYSICS", "x": 3, "y": 1, "z": 3, "load_kn": 10.0, "direction": "Y"}

{"action_type": "COMMIT_DESIGN"}
```

---

## Observation Space

Each step returns a dict with:

| Field | Type | Description |
|-------|------|-------------|
| `message` | str | Natural-language summary for LLM agents |
| `task_description` | str | Full task spec (constant per episode) |
| `grid_summary` | str | Current voxel grid state as text |
| `constraint_status` | dict | `{stability_class, n_valid_rooms, room_score, airflow_score, light_score, egress_score, connected_fraction}` |
| `reward_breakdown` | dict | Per-component reward values |
| `budget_remaining` | int | Remaining material cost units |
| `probe_budget_remaining` | int | Remaining PROBE_PHYSICS uses (hard task) |
| `done` | bool | Episode ended flag |
| `graded_score` | float\|null | Final score after COMMIT_DESIGN |

---

## Reward Function

```
R_t = w0 × R_structural(s_t)
    + w1 × R_functional(s_t)
    + w2 × R_physics(s_t)
    - w3 × R_cost(s_t)
    + w4 × R_efficiency(s_t)
```

| Level | w0 struct | w1 funct | w2 physics | w3 cost | w4 effic |
|-------|-----------|---------|----------|---------|---------|
| Easy  | 3.0 | 0.5 | 0.3 | 0.5 | 0.3 |
| Medium | 2.0 | 1.5 | 0.5 | 0.5 | 0.3 |
| Hard | 2.0 | 1.5 | 1.5 | 0.5 | 0.3 |

**Terminal events:**
- COLLAPSE → −0.5 penalty (episode ends)
- COMMIT_DESIGN → graded score (0.0–1.0) applied as final reward

---

## Baseline Scores

Baseline uses a simple LLM agent (3–5 step wall-building strategy):

| Task | Baseline Score | Notes |
|------|---------------|-------|
| `stable_shelter` | ~0.35 | Walls placed, usually no enclosed room detected |
| `functional_office` | ~0.15 | Structural only, functional scores low |
| `material_detective` | ~0.10 | No probing, material properties missed |

> Scores from rule-based fallback (no LLM). With a capable LLM agent, expect ~0.5–0.7 on easy, ~0.3–0.5 on medium.

---

## Setup & Usage

### Local Development

```bash
# Clone & install
git clone https://github.com/ayushsi42/pgsa
cd pgsa
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env: set API_BASE_URL, MODEL_NAME, HF_TOKEN

# Start server
cd pgsa
PYTHONPATH=. uvicorn pgsa_env.server.app:app --host 0.0.0.0 --port 7860 --reload

# In another terminal — run inference
python inference.py --task all --steps 30

# Run only easy task
python inference.py --task easy

# Enable web UI for manual testing
ENABLE_WEB_INTERFACE=true uvicorn pgsa_env.server.app:app --port 7860
# Open http://localhost:7860/web
```

### Docker

```bash
# Build
docker build -f Dockerfile . -t pgsa-env

# Run
docker run -p 7860:7860 -e ENABLE_WEB_INTERFACE=true pgsa-env

# Test
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/tasks` | List all 3 tasks |
| POST | `/reset` | Start new episode `{"difficulty": "easy"\|"medium"\|"hard"}` |
| POST | `/step` | Execute action `{"message": "{...json...}"}` |
| GET | `/state` | Current episode metadata |
| GET | `/web` | Interactive web UI (if enabled) |
| GET | `/docs` | OpenAPI documentation |

---

## Project Structure

```
pgsa/
├── openenv.yaml              # Environment manifest
├── pyproject.toml            # Package dependencies
├── .env.example              # Environment variable template
├── README.md                 # This file
├── pgsa_env/                 # Core Python package
│   ├── __init__.py           
│   ├── models.py             # Pydantic Action/Observation/State models
│   ├── client.py             # Sync/Async Python client
│   ├── environment.py        # Core step/reset loop
│   └── server/               # FastAPI + graders + physics wrapper
│       ├── app.py
│       ├── physics_sim.py
│       ├── task_generator.py
│       └── graders.py
├── scripts/
│   └── inference.py          # Baseline evaluation script
└── Dockerfile                # Container image (port 7860)
```

---

## Technical Details

### Physics Engine

The physics engine in `pgsa_env/server/physics_sim.py` provides a **lightweight heuristic approximation** of the full PGSA physics pipeline:

- **Structural**: BFS connectivity check from foundation + column-load stress estimation vs. material yield strength → `STABLE | SERVICEABILITY_FAIL | PARTIAL_COLLAPSE | COLLAPSE`
- **Airflow**: Checks whether enclosed rooms have DOOR access to exterior or ventilated spaces
- **Lighting**: Counts WINDOW elements on room boundaries, weighted by room type importance
- **Egress**: BFS-like distance from room centroids to nearest exterior exit (≤30m NFPA 101 proxy)

### Hidden Material Properties (Hard Task)

At curriculum level 3, true material `yield_mpa` and `E_gpa` are drawn from log-normal distributions (σ_scale=0.30). The agent must:
1. Place a test element
2. Call `PROBE_PHYSICS` to observe stress/deformation at a given load
3. Infer material strength from the probe response
4. Choose appropriate materials for structural elements

---

## License

MIT License — see [LICENSE](LICENSE)
