"""
PGSA EnvClient
---------------
Client-side class for interacting with the PGSA OpenEnv environment.
Implements both async and sync interaction patterns compatible with
the OpenEnv framework (openenv-core >= 0.2.3).
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import httpx

from pgsa_env.models import PGSAAction, PGSAObservation, PGSAState


# ─── STEP RESULT ──────────────────────────────────────────────────────────────

class StepResult:
    """Encapsulates the result of a step() or reset() call."""

    def __init__(self, observation: PGSAObservation, reward: float = 0.0, done: bool = False, info: dict = None):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info or {}


# ─── SYNC WRAPPER ─────────────────────────────────────────────────────────────

class SyncPGSAEnv:
    """Synchronous wrapper around PGSAEnv."""

    def __init__(self, client: "PGSAEnv"):
        self._client = client

    def __enter__(self):
        return self

    def __exit__(self, *args):
        asyncio.run(self._client.close())

    def reset(self, difficulty: str = "easy", seed: Optional[int] = None) -> StepResult:
        return asyncio.run(self._client.reset(difficulty=difficulty, seed=seed))

    def step(self, action: PGSAAction) -> StepResult:
        return asyncio.run(self._client.step(action))

    def state(self) -> PGSAState:
        return asyncio.run(self._client.state())


# ─── ASYNC CLIENT ─────────────────────────────────────────────────────────────

class PGSAEnv:
    """
    Async OpenEnv client for the PGSA environment.

    Usage (async):
        async with PGSAEnv(base_url="http://localhost:7860") as env:
            result = await env.reset(difficulty="easy")
            print(result.observation.message)
            result = await env.step(PGSAAction(message='{"action_type": "COMMIT_DESIGN"}'))
            print(result.reward)

    Usage (sync):
        with PGSAEnv(base_url="http://localhost:7860").sync() as env:
            result = env.reset()
            result = env.step(PGSAAction(message='{"action_type": "COMMIT_DESIGN"}'))
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "PGSAEnv":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def sync(self) -> SyncPGSAEnv:
        """Get a synchronous wrapper."""
        return SyncPGSAEnv(self)

    def _client_or_raise(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self._client

    async def reset(self, difficulty: str = "easy", seed: Optional[int] = None) -> StepResult:
        """Initialize a new episode. Returns StepResult with initial observation."""
        client = self._client_or_raise()
        payload = {"difficulty": difficulty}
        if seed is not None:
            payload["seed"] = seed
        resp = client.post("/reset", json=payload) if not asyncio.iscoroutine(client.post("/reset", json=payload)) else await client.post("/reset", json=payload)
        # Handle both sync and async httpx
        if hasattr(resp, "__await__"):
            resp = await resp
        resp.raise_for_status()
        data = resp.json()
        obs = PGSAObservation(**data["observation"])
        return StepResult(observation=obs, reward=0.0, done=False, info=data.get("info", {}))

    async def step(self, action: PGSAAction) -> StepResult:
        """Execute one action. Returns StepResult."""
        client = self._client_or_raise()
        payload = {"message": action.message}
        resp = await client.post("/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        obs = PGSAObservation(**data["observation"])
        return StepResult(
            observation=obs,
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )

    async def state(self) -> PGSAState:
        """Get current episode state."""
        client = self._client_or_raise()
        resp = await client.get("/state")
        resp.raise_for_status()
        return PGSAState(**resp.json()["state"])

    @classmethod
    async def from_url(cls, base_url: str) -> "PGSAEnv":
        """Create a client connected to a running environment server."""
        instance = cls(base_url=base_url)
        instance._client = httpx.AsyncClient(base_url=base_url, timeout=instance._timeout)
        return instance
