"""
DEPRECATED — legacy standalone server, kept only as a backward-compatible shim.

The canonical FastAPI app lives at `server/app.py`. It is what the Dockerfile
runs (`uvicorn server.app:app`), what the `server` console-script entry point
in `pyproject.toml` points at (`server.app:main`), and what `tests/test_server_routes.py`
imports. This file originally contained an earlier, simpler standalone
implementation (fewer endpoints: no `/metadata`, `/schema`, `/mcp`, `/query_forces`,
`/what_if_remove`, `/render`) that predates `server/app.py` becoming canonical.

Note that because `server/` is a real package (it has `__init__.py`), a plain
`import server` already resolves to the package, not to this file — so this
module was already unreachable via import; it could only ever be invoked
directly as `python server.py`. That entry point is preserved below so any
old muscle memory / scripts still work, but it simply runs the canonical app.

New code should use one of:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
    server                       # console-script entry point
"""

from __future__ import annotations

from server.app import app  # noqa: F401  re-exported for any `from server import app`-style usage

if __name__ == "__main__":
    import sys

    import uvicorn

    print(
        "[DEPRECATED] server.py (root) is a shim — running the canonical server/app.py instead. "
        "Use `uvicorn server.app:app` or the `server` console script directly.",
        file=sys.stderr,
    )
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)
