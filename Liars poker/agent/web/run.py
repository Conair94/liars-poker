"""
Entry point for the Liar's Poker web interface.

Run from papers/Liars poker/:
    python agent/web/run.py

Then open http://localhost:8000 in your browser.
"""

import os
import sys

# Ensure the paper root is on sys.path so all imports resolve.
_HERE      = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_PAPER_DIR = os.path.abspath(os.path.join(_AGENT_DIR, ".."))
for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import uvicorn  # noqa: E402

if __name__ == "__main__":
    print("Starting Liar's Poker server at http://localhost:8000")
    uvicorn.run(
        "agent.web.backend.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[_PAPER_DIR],
    )
