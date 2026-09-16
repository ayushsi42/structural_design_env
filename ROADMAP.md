# Roadmap — StructuralDesignEnv

## Current State

- The physics core is solid: a 3D 6-DOF direct-stiffness solver plus EN 1993-1-1 (Eurocode 3) member checks and EN 1998-1 seismic loading, verified by 96/96 passing tests (`tests/test_stiffness.py`, `tests/test_eurocode.py`, `tests/test_graders.py`, `tests/test_env.py`, `tests/test_server_routes.py`) run and confirmed locally.
- Three tasks (`task1_warehouse`, `task2_office`, `task3_hospital`) are fully wired end-to-end: reset → step → grade, each with its own scoring formula.
- The FastAPI server (`server/app.py`) implements the full OpenEnv contract (`/metadata`, `/schema`, `/mcp`) plus useful research endpoints (`/query_forces`, `/what_if_remove`, `/render`).
- Duplication exists that should be resolved rather than left as tech debt: `server.py` (root) vs. `server/app.py`, and `inference.py` (root) vs. `scripts/inference.py` — both pairs overlap in purpose with slightly different env-var conventions.
- No LLM-agent baseline run with logged, reproducible scores is checked into the repo — `inference.py` runs episodes and prints `[START]/[STEP]/[END]` logs, but no result table exists yet.
- There is no CI configured (no `.github/workflows/`), so `pytest` only runs when a human remembers to run it locally.
- HF Space deployment history exists in git log (`Add root / endpoint to fix HF Space 404`, HF Space YAML front-matter in the README) but no HF remote is configured in this checkout, so current live status is unverified from this repo alone.

## Phase 1 — Near-term (weeks)

- **Run and publish a real baseline**: execute `inference.py` against a real LLM (e.g. `gpt-4o-mini` or a local Ollama model) for all three tasks, capture the `[END] success=... steps=... score=...` lines, and commit a results table (model, task, score, steps, wall-clock) — turning the "environment exists" claim into a demonstrated one.
- **Add GitHub Actions CI**: a workflow that runs `pip install -e ".[dev]"` and `pytest tests/ -q` on every push/PR, so the 96-test suite is enforced automatically rather than relying on manual runs.
- **Consolidate the duplicate server/inference implementations**: pick `server/app.py` and root `inference.py` as canonical (they already are, per packaging metadata and the Dockerfile), and either delete or clearly deprecate `server.py` and `scripts/inference.py` to remove ambiguity for new contributors.
- **Verify and document HF Space status**: either confirm the Space is live and link it from the README, or remove the stale HF-specific commentary if it's been decommissioned.
- **Expand `test_server_routes.py` coverage**: it currently has only 4 tests versus 32 for `test_env.py`; add route-level tests for `/grade`, `/query_forces`, `/what_if_remove`, and `/render` (currently exercised only indirectly, if at all).
- **Add a `/leaderboard`-style results artifact**: even a static `RESULTS.md` populated by the Phase 1 baseline runs, tracked per task/model.

## Phase 2 — Medium-term

- **Add P-delta (second-order) effects** to `solver/stiffness_matrix.py` for the sway-sensitive `task2_office` and `task3_hospital` cases, where first-order drift may understate real lateral response.
- **Expand the section database** in `solver/sections.py` beyond HEB/IPE (e.g. HEA columns, UB/UC for agents trained on non-European conventions) to widen the design space.
- **Multi-agent collaborative design**: split the design into role-specialized agents (e.g. one proposes column grid, another sizes beams, a third handles lateral bracing) coordinating through the same `/step` loop.
- **Benchmark multiple LLMs head-to-head** on all three tasks using the existing `inference.py`/`scripts/inference.py` harness, extending the Phase 1 results table into a genuine comparison.
- **Torsional response detail**: `env.py`'s `_compute_lateral_drift` already captures per-column drift (catching torsion implicitly); add an explicit torsional irregularity check per EN 1998-1 §4.2.3.2 for `task2_office`'s "torsional_balance" score component.

## Phase 3 — Stretch

- **Calibrate against a real-world dataset**: compare solver output (drift, member forces) against published structural analysis results or textbook worked examples beyond the current analytical unit tests, to strengthen the "real physics" claim with external validation.
- **Public leaderboard**: a hosted page (e.g. an HF Space or static site) showing live-updated scores across models/tasks, fed by scheduled `inference.py` runs.
- **Write up the environment and baseline results** (blog post or short paper) for the OpenEnv community — the combination of DSM + Eurocode 3 + seismic loading in an LLM-gradable RL environment is a distinctive contribution worth documenting properly.
- **4th task tier**: a mixed-use podium tower reusing `tasks/task3_hospital.py`'s seismic loading machinery but with irregular plan geometry, testing agents on non-rectangular grids.

## Success Metrics

- **Phase 1** is done when: CI is green on every push, at least one LLM has a logged, reproducible score for all 3 tasks in a committed results file, and there is exactly one canonical server file and one canonical inference script referenced consistently across README/Dockerfile/pyproject.
- **Phase 2** is done when: P-delta effects are covered by new solver tests analogous to `test_stiffness.py`'s existing analytical checks, at least 2 additional section families are selectable by the agent, and a documented multi-model comparison table exists.
- **Phase 3** is done when: solver output has been checked against at least one external/published reference case, a public leaderboard URL is live and linked from the README, and a written report/post is published and linked.
