#!/usr/bin/env python3
"""
DEPRECATED — legacy inference script, kept only as a backward-compatible shim.

The canonical LLM-agent runner is the root-level `inference.py` (referenced by
`pyproject.toml`'s packaging metadata, the README's "Getting Started" section,
and used to produce `results/gpt-4o-mini_baseline.json`). This script
originally targeted Anthropic-style env vars (`ENV_URL`, `INFERENCE_MODEL`,
`ANTHROPIC_API_KEY`, `OPENAI_BASE_URL`) with a slightly different prompt and
logging format, predating the OpenAI-style env-var convention
(`SERVER_URL`, `MODEL_NAME`, `HF_TOKEN`, `API_BASE_URL`) that `inference.py`
now uses.

To avoid maintaining two divergent agent loops, this shim maps the old env
vars onto the new ones (without overriding anything already set) and then
delegates to the canonical `inference.run_episode`.

Usage (unchanged): python scripts/inference.py [task_id]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_OLD_TO_NEW_ENV = {
    "ENV_URL": "SERVER_URL",
    "INFERENCE_MODEL": "MODEL_NAME",
    "OPENAI_BASE_URL": "API_BASE_URL",
}
for _old, _new in _OLD_TO_NEW_ENV.items():
    if os.getenv(_old) and not os.getenv(_new):
        os.environ[_new] = os.environ[_old]

# HF_TOKEN is what inference.py calls the API key env var; fall back to
# whichever key-bearing var this script's old callers used.
if not os.getenv("HF_TOKEN"):
    for _key_var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if os.getenv(_key_var):
            os.environ["HF_TOKEN"] = os.environ[_key_var]
            break

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference import run_episode  # noqa: E402  canonical implementation, imported after path/env setup


if __name__ == "__main__":
    valid_tasks = {"task1_warehouse", "task2_office", "task3_hospital"}
    task = sys.argv[1] if len(sys.argv) > 1 else "task1_warehouse"
    if task not in valid_tasks:
        print(f"Unknown task '{task}'. Valid: {sorted(valid_tasks)}")
        sys.exit(1)

    print(
        "[DEPRECATED] scripts/inference.py is a shim — delegating to the canonical root inference.py",
        file=sys.stderr,
    )
    run_episode(task)
