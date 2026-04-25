"""
pytest conftest for the unified tests/ tree.

Adds src/ and src/training/probs/ to sys.path so tests can import
`game.bids`, `agents.heuristic.cfr_1v1`, `poker_math_exact`, etc.,
without per-test path-setup boilerplate.

Until pyproject.toml is rewritten to declare src/ as the package root
(commit P1.11), this is the canonical entry point for path discovery.
"""
import os
import sys

_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SRC_DIR   = os.path.join(_REPO_ROOT, "src")
_PROBS_DIR = os.path.join(_SRC_DIR, "training", "probs")

for _p in (_PROBS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
