#!/usr/bin/env python3
"""
PGSA Baseline Inference Script
--------------------------------
Runs an LLM agent against all 3 PGSA tasks using the OpenAI API client.
Emits structured [START] / [STEP] / [END] logs per OpenEnv hackathon spec.

Environment variables (can be set in .env):
  API_BASE_URL   - OpenAI-compatible API base URL (e.g. http://localhost:11434/v1)
  MODEL_NAME     - Model identifier (e.g. llama3.1, gpt-4o, etc.)
  HF_TOKEN       - Hugging Face API token (for HF Space hosting)
  SERVER_URL     - PGSA server URL (default: http://localhost:7860)

Usage:
  python inference.py
  python inference.py --task easy          # Run only easy task
  python inference.py --task all           # Run all 3 tasks (default)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import List, Optional

import httpx

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — use raw env vars

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL     = os.environ.get("API_BASE_URL", "http://localhost:11434/v1")
API_KEY          = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", "none"))
MODEL_NAME       = os.environ.get("MODEL_NAME", "llama3.1")
SERVER_URL       = os.environ.get("SERVER_URL", "http://localhost:7860")
MAX_STEPS        = int(os.environ.get("MAX_STEPS", "30"))   # Per task; keep under 20min
SUCCESS_THRESHOLD = 0.5
BENCHMARK        = "pgsa-env"

TASKS = ["stable_shelter", "functional_office", "material_detective"]
TASK_DIFFICULTY  = {
    "stable_shelter":    "easy",
    "functional_office": "medium",
    "material_detective": "hard",
}
# Max possible total reward per task (used to normalize score)
MAX_TOTAL_REWARD = {
    "stable_shelter":    MAX_STEPS * 0.2,
    "functional_office": MAX_STEPS * 0.2,
    "material_detective": MAX_STEPS * 0.2,
}

# ── Logging (mandatory [START]/[STEP]/[END] format) ───────────────────────────

def log_start(task: str, env: str, model: str):
    print(json.dumps({
        "event": "START",
        "task": task,
        "env": env,
        "model": model,
        "timestamp": time.time(),
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    print(json.dumps({
        "event": "STEP",
        "step": step,
        "action": action[:500] if action else "",
        "reward": round(reward, 4),
        "done": done,
        "error": error,
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(json.dumps({
        "event": "END",
        "success": success,
        "steps": steps,
        "score": round(score, 4),
        "rewards": [round(r, 4) for r in rewards],
        "total_reward": round(sum(rewards), 4),
    }), flush=True)


# ── OpenAI Client ─────────────────────────────────────────────────────────────

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    print("[DEBUG] openai package not installed. Install with: pip install openai", flush=True)


def get_llm_action(
    client,
    step: int,
    task_description: str,
    observation_message: str,
    action_history: List[str],
    last_reward: float,
) -> str:
    """
    Query the LLM to get the next action JSON.
    Returns the raw response text.
    """
    if not _HAS_OPENAI:
        # Fallback: always commit design after a few steps
        if step >= 3:
            return json.dumps({"action_type": "COMMIT_DESIGN"})
        # Place a simple wall
        return json.dumps({
            "action_type": "PLACE_ELEMENT",
            "x": step, "y": 1, "z": step,
            "element_type": "WALL", "material_id": 0
        })

    history_str = "\n".join(action_history[-5:]) if action_history else "None yet."

    system_prompt = (
        "You are an expert architect playing the PGSA environment. "
        "Your goal is to design a building that satisfies structural and functional requirements. "
        "At each step, you must output a SINGLE JSON action object and nothing else.\n\n"
        "Rules:\n"
        "1. Output ONLY a JSON object — no explanation, no markdown, no extra text.\n"
        "2. The JSON must have an 'action_type' field.\n"
        "3. Place elements one at a time using PLACE_ELEMENT.\n"
        "4. After building walls/floor for a room, use ANNOTATE_ROOM with the room corners.\n"
        "5. When done, output: {\"action_type\": \"COMMIT_DESIGN\"}\n\n"
        "Valid action types: PLACE_ELEMENT, REMOVE_ELEMENT, REPLACE_MATERIAL, "
        "PLACE_BATCH, PROBE_PHYSICS, ANNOTATE_ROOM, QUERY_BELIEF, COMMIT_DESIGN\n\n"
        f"TASK:\n{task_description}\n\n"
        f"Recent action history:\n{history_str}"
    )

    user_msg = (
        f"Step {step} | Last reward: {last_reward:+.4f}\n\n"
        f"CURRENT OBSERVATION:\n{observation_message}\n\n"
        "Output your next action as a JSON object:"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200,
            temperature=0.3,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else json.dumps({"action_type": "COMMIT_DESIGN"})
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return json.dumps({"action_type": "COMMIT_DESIGN"})


# ── HTTP helpers ──────────────────────────────────────────────────────────────

async def server_reset(http: httpx.AsyncClient, difficulty: str) -> dict:
    resp = await http.post("/reset", json={"difficulty": difficulty})
    resp.raise_for_status()
    return resp.json()


async def server_step(http: httpx.AsyncClient, message: str) -> dict:
    resp = await http.post("/step", json={"message": message})
    resp.raise_for_status()
    return resp.json()


# ── Task runner ───────────────────────────────────────────────────────────────

async def run_task(
    task_name: str,
    client,
    http: httpx.AsyncClient,
) -> float:
    """Run a single task. Returns graded score [0.0, 1.0]."""
    difficulty = TASK_DIFFICULTY[task_name]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    graded_score = None

    try:
        # Reset
        reset_data = await server_reset(http, difficulty)
        obs_data = reset_data["observation"]
        task_description = obs_data.get("task_description", "")
        last_obs_message = obs_data.get("message", "")
        last_reward = 0.0
        is_done = obs_data.get("done", False)

        for step in range(1, MAX_STEPS + 1):
            if is_done:
                break

            # Get LLM action
            action_str = get_llm_action(
                client=client,
                step=step,
                task_description=task_description,
                observation_message=last_obs_message,
                action_history=history,
                last_reward=last_reward,
            )

            # Execute
            step_data = await server_step(http, action_str)
            obs_data  = step_data["observation"]
            reward    = step_data.get("reward", 0.0)
            is_done   = step_data.get("done", False)
            info      = step_data.get("info", {})

            rewards.append(reward)
            steps_taken = step
            last_obs_message = obs_data.get("message", "")
            last_reward = reward

            # Extract graded score if available
            if info.get("graded_score") is not None:
                graded_score = info["graded_score"]
            if obs_data.get("graded_score") is not None:
                graded_score = obs_data["graded_score"]

            log_step(step=step, action=action_str, reward=reward, done=is_done, error=None)
            history.append(f"Step {step}: {action_str[:80]!r} → reward {reward:+.4f}")

            if is_done:
                break

        # Final score: use graded_score if available, else normalized cumulative reward
        if graded_score is not None:
            score = graded_score
        else:
            max_r = MAX_TOTAL_REWARD.get(task_name, MAX_STEPS * 0.2)
            score = min(max(sum(rewards) / max(max_r, 0.01), 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        score = 0.0
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(tasks_to_run: List[str]) -> None:
    # Init OpenAI client (works with any OpenAI-compatible endpoint)
    client = None
    if _HAS_OPENAI:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    else:
        print("[DEBUG] Running without openai package — using fallback actions.", flush=True)

    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=60.0) as http:
        # Verify server health
        try:
            health = await http.get("/health")
            health.raise_for_status()
            print(f"[DEBUG] Server online: {health.json()}", flush=True)
        except Exception as e:
            print(f"[ERROR] Cannot reach PGSA server at {SERVER_URL}: {e}", flush=True)
            print("[ERROR] Start the server first: uvicorn pgsa_env.server.app:app --port 7860", flush=True)
            sys.exit(1)

        all_scores = {}
        for task_name in tasks_to_run:
            print(f"\n{'='*60}", flush=True)
            print(f"Running task: {task_name} ({TASK_DIFFICULTY[task_name]})", flush=True)
            print('='*60, flush=True)
            score = await run_task(task_name, client, http)
            all_scores[task_name] = score
            print(f"[RESULT] {task_name}: {score:.4f}", flush=True)

        print(f"\n{'='*60}", flush=True)
        print("SUMMARY", flush=True)
        print('='*60, flush=True)
        for task, score in all_scores.items():
            status = "✓ PASS" if score >= SUCCESS_THRESHOLD else "✗ FAIL"
            print(f"  {status}  {task:30s}  {score:.4f}", flush=True)

        overall = sum(all_scores.values()) / max(len(all_scores), 1)
        print(f"\n  OVERALL AVERAGE SCORE: {overall:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGSA Baseline Inference")
    parser.add_argument(
        "--task", default="all",
        choices=["all", "easy", "medium", "hard",
                 "stable_shelter", "functional_office", "material_detective"],
        help="Which task(s) to run"
    )
    parser.add_argument("--steps", type=int, default=MAX_STEPS,
                        help="Max steps per task")
    args = parser.parse_args()

    MAX_STEPS = args.steps

    if args.task == "all":
        tasks = TASKS
    elif args.task in ("easy", "stable_shelter"):
        tasks = ["stable_shelter"]
    elif args.task in ("medium", "functional_office"):
        tasks = ["functional_office"]
    elif args.task in ("hard", "material_detective"):
        tasks = ["material_detective"]
    else:
        tasks = TASKS

    asyncio.run(main(tasks))
