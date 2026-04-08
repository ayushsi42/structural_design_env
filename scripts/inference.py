#!/usr/bin/env python3
"""
Inference script for StructuralDesignEnv.
LLM agent (Claude) designs a steel building frame step by step.

Usage:
    python scripts/inference.py [task_id]

    task_id: task1_warehouse (default) | task2_office | task3_hospital

Environment variables:
    ENV_URL          — base URL of the running server (default: http://localhost:7860)
    INFERENCE_MODEL  — model name (default: claude-opus-4-6)
    ANTHROPIC_API_KEY or OPENAI_API_KEY
    OPENAI_BASE_URL  — override API base URL
"""

import json
import os
import sys

import httpx
from openai import OpenAI

BASE_URL = os.getenv("ENV_URL", "http://localhost:7860")
MODEL = os.getenv("INFERENCE_MODEL", "claude-opus-4-6")

SYSTEM_PROMPT = """You are a structural engineer designing a building frame step-by-step.
You place columns, beams, and shear walls on a building grid, then receive
physics analysis showing whether your design is structurally safe.

PHYSICS RULES:
- Beams carry vertical load via bending: M = w*L^2/8. Longer spans need bigger sections.
- Columns carry vertical load via compression. More floors = higher axial load.
- Lateral loads (wind/seismic) require lateral resistance: shear walls or moment frames.
- Utilization ratio (UR) = demand/capacity. Must be < 1.0 for all members.
- UR=1.47 means 47% overstressed → upgrade section or reduce span.
- Deflection limit: maximum beam deflection < span/300.
- Lateral drift limit: story drift < height/500.

DESIGN STRATEGY:
1. Establish column grid (spacing 4-6m gives economical spans)
2. Add beams in both directions
3. Check physics → upgrade any UR > 1.0 members
4. Add shear walls if lateral drift > limit
5. Downgrade members with UR < 0.6 (wasteful)
6. Signal "done" only when all URs < 1.0

Respond with a single JSON action object matching the StructuralAction schema.
Do not include any text outside the JSON object."""

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.anthropic.com/v1"),
    api_key=os.getenv("ANTHROPIC_API_KEY", os.getenv("OPENAI_API_KEY", "")),
)


def run_episode(task_id: str = "task1_warehouse"):
    env = httpx.Client(base_url=BASE_URL, timeout=60)

    # Reset
    resp = env.post("/reset", json={"task_id": task_id})
    resp.raise_for_status()
    data = resp.json()
    session_id = data["session_id"]
    obs = data["observation"]

    print(f"\n{'=' * 60}")
    print(f"Task: {task_id}  |  Session: {session_id}")
    print(f"{'=' * 60}")
    print(obs["message"])

    messages = [{"role": "user", "content": obs["message"]}]
    done = False
    total_reward = 0.0
    step = 0
    max_steps = obs.get("max_steps", 100)

    while not done and step < max_steps + 5:
        # Query LLM
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                max_tokens=512,
                temperature=0.0,
            )
            action_str = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n[LLM error] {e}")
            break

        # Strip markdown code fences if present
        if action_str.startswith("```"):
            action_str = action_str.split("```")[1]
            if action_str.startswith("json"):
                action_str = action_str[4:]
            action_str = action_str.strip()

        print(f"\n[Step {step + 1}] Agent: {action_str}")
        messages.append({"role": "assistant", "content": action_str})

        # Step environment
        try:
            resp = env.post(
                "/step",
                json={"session_id": session_id, "message": action_str},
            )
            resp.raise_for_status()
            step_data = resp.json()
        except Exception as e:
            print(f"\n[HTTP error] {e}")
            break

        obs = step_data["observation"]
        reward = step_data["reward"]
        done = step_data["done"]
        info = step_data.get("info", {})

        total_reward += reward
        step += 1

        print(f"Reward: {reward:+.4f}  |  Total: {total_reward:+.4f}  |  Done: {done}")
        print(obs["message"])

        messages.append({"role": "user", "content": obs["message"]})

        if done:
            graded = info.get("graded_score", 0.0)
            print(f"\n{'=' * 60}")
            print(f"EPISODE COMPLETE")
            print(f"Steps: {step}  |  Total reward: {total_reward:.3f}  |  Score: {graded:.4f}")
            print(f"Valid: {obs.get('is_structurally_valid', False)}")
            print(f"Elements: {obs.get('n_elements_placed', 0)}")
            print(f"Steel mass: {obs.get('total_steel_mass_kg', 0):.0f} kg")
            print(f"{'=' * 60}\n")

    return total_reward


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "task1_warehouse"
    valid_tasks = {"task1_warehouse", "task2_office", "task3_hospital"}
    if task not in valid_tasks:
        print(f"Unknown task '{task}'. Valid: {sorted(valid_tasks)}")
        sys.exit(1)

    run_episode(task)
