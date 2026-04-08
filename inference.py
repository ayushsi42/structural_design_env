#!/usr/bin/env python3
"""
Inference script for StructuralDesignEnv — OpenEnv Hackathon Round 1 submission.

An LLM agent designs a steel building frame step-by-step, receiving physics
analysis feedback (utilization ratios, deflections, drift) at each step.

Required environment variables:
    API_BASE_URL  — LLM API endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME    — Model identifier (e.g. gpt-4o-mini)
    HF_TOKEN      — Hugging Face / API key

Optional:
    TASK_ID       — task1_warehouse | task2_office | task3_hospital (default: task1_warehouse)
    SPACE_URL     — URL of the running OpenEnv server (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# --------------------------------------------------------------------------
# Hackathon-required env vars (exact spec: only API_BASE_URL and MODEL_NAME have defaults)
# --------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")                 # no default — required at runtime
LOCAL_IMAGE_NAME: Optional[str] = os.getenv("LOCAL_IMAGE_NAME")  # optional, for from_docker_image()

# --------------------------------------------------------------------------
# Optional config
# --------------------------------------------------------------------------
TASK_ID: str = os.getenv("TASK_ID", "task1_warehouse")
SPACE_URL: str = os.getenv("SPACE_URL", "http://localhost:7860")
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 4096  # Qwen3.5 needs extra tokens for thinking before JSON

BENCHMARK = "structural_design_env"

SYSTEM_PROMPT = textwrap.dedent("""
    You are a structural engineer designing a building frame step-by-step.
    Each response must be ONE JSON object and nothing else — no prose, no markdown fences.

    ACTION TYPES (exact field names required):

    1. Place column:
    {"action_type":"place_column","grid_x":5,"grid_y":0,"floor":0,"section":"HEB200"}
    grid_x: 0 to site_width-1, grid_y: 0 to site_depth-1, floor: 0 to n_floors-1
    sections: HEB140 HEB160 HEB200 HEB240 HEB300 HEB360 HEB400

    2. Place beam (connect two existing columns on same floor):
    {"action_type":"place_beam","from_node_x":0,"from_node_y":0,"to_node_x":5,"to_node_y":0,"floor":0,"section":"IPE300","orientation":"x"}
    orientation "x" = east-west beam, "y" = north-south beam
    sections: IPE200 IPE240 IPE300 IPE360 IPE400 IPE450 IPE500

    3. Add shear wall (both endpoint columns must already exist):
    {"action_type":"add_wall","from_node_x":0,"from_node_y":0,"to_node_x":0,"to_node_y":5,"floor":0,"thickness_m":0.2,"orientation":"y"}

    4. Upgrade section (next larger):
    {"action_type":"upgrade_section","element_id":"col_5_0_0"}

    5. Downgrade section (next smaller):
    {"action_type":"downgrade_section","element_id":"col_5_0_0"}

    6. Finish design:
    {"action_type":"done"}

    PHYSICS RULES:
    - UR (utilization ratio) = demand/capacity. All URs must be < 1.0.
    - Beam deflection limit: span/300. Longer spans need bigger IPE sections.
    - Lateral drift limit: story_height/500. Add shear walls if drift is exceeded.

    DESIGN STRATEGY:
    1. Place columns at 4-6m spacing (check site_width_m and site_depth_m in the observation)
    2. Connect columns with beams in both x and y directions
    3. If wind or seismic loads exist, add shear walls
    4. Upgrade any member with UR > 1.0
    5. Send {"action_type":"done"} when all URs < 1.0 or you are satisfied
""").strip()

# --------------------------------------------------------------------------
# Required logging functions (hackathon spec — exact format mandatory)
# --------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# --------------------------------------------------------------------------
# LLM helper
# --------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def build_user_prompt(step: int, obs_message: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Environment feedback:
        {obs_message}
        Previous actions:
        {history_block}
        Send your next action as a JSON object.
    """).strip()


def get_model_action(step: int, obs_message: str, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs_message, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip thinking preamble from reasoning models (Qwen3.5, DeepSeek-R1, etc.)
        # vllm may strip the opening <think> tag but keep </think>, so split on it
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        elif "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("```")
            text = lines[1] if len(lines) > 1 else text
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return text if text else '{"action_type": "done"}'
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return '{"action_type": "done"}'


# --------------------------------------------------------------------------
# Episode runner
# --------------------------------------------------------------------------

def run_episode(task_id: str) -> float:
    env = httpx.Client(base_url=SPACE_URL, timeout=60)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    # Reset
    try:
        resp = env.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
    except Exception as exc:
        print(f"[DEBUG] Reset failed: {exc}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

    data = resp.json()
    session_id = data["session_id"]
    obs = data["observation"]
    max_steps: int = obs.get("max_steps", 100)

    history: List[str] = []
    rewards: List[float] = []
    last_reward = 0.0
    score = 0.0
    success = False
    steps_taken = 0
    done = False

    for step in range(1, max_steps + 1):
        if done:
            break

        # Query LLM
        action_str = get_model_action(step, obs.get("message", ""), last_reward, history)
        history.append(action_str)

        # Parse action_type for compact logging
        try:
            action_obj = json.loads(action_str)
            action_label = action_obj.get("action_type", action_str[:40])
        except Exception:
            action_label = action_str[:40]

        # Step environment
        error_msg: Optional[str] = None
        try:
            resp = env.post("/step", json={"session_id": session_id, "message": action_str})
            resp.raise_for_status()
            step_data = resp.json()
        except Exception as exc:
            error_msg = str(exc)[:80]
            log_step(step, action_label, 0.0, False, error_msg)
            break

        obs = step_data["observation"]
        reward: float = step_data.get("reward", 0.0)
        done = step_data.get("done", False)
        info = step_data.get("info", {})

        if obs.get("last_action_error"):
            error_msg = obs["last_action_error"][:80]

        rewards.append(reward)
        last_reward = reward
        steps_taken = step

        log_step(step, action_label, reward, done, error_msg)

        if done:
            score = float(info.get("graded_score", 0.0))
            success = obs.get("is_structurally_valid", False)
            break

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else TASK_ID
    valid_tasks = {"task1_warehouse", "task2_office", "task3_hospital"}
    if task not in valid_tasks:
        print(f"[DEBUG] Unknown task '{task}'. Valid: {sorted(valid_tasks)}", flush=True)
        sys.exit(1)

    run_episode(task)
