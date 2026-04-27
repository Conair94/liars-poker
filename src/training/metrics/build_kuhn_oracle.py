"""Solve the Kuhn-sized Liar's Poker game and persist the reference policy.

Run from repo root:

    python3 -m training.metrics.build_kuhn_oracle [--iterations 5000]

Writes ``data/oracles/liars_poker_kuhn_policy.npz`` with two arrays:
  - ``infoset_keys`` (np.ndarray[str]): tabular-policy info-state strings
  - ``action_probs`` (np.ndarray[float32, (n, num_actions)]): policy rows

This is the reference oracle used to validate the project's CFR / R-NaD
implementations on a tractable variant.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.metrics.exploitability import (  # noqa: E402
    compute_exploitability,
    kuhn_cfr_plus_solve,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument(
        "--out",
        default=os.path.abspath(
            os.path.join(_HERE, "..", "..", "..", "data", "oracles",
                         "liars_poker_kuhn_policy.npz")
        ),
    )
    args = p.parse_args()

    game, policy = kuhn_cfr_plus_solve(iterations=args.iterations)
    expl = compute_exploitability(game, policy)
    print(f"final exploitability after {args.iterations} iters: {expl:.6f}")

    table = policy.action_probability_array
    keys = np.array(list(policy.state_lookup.keys()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        infoset_keys=keys,
        action_probs=table.astype(np.float32),
        exploitability=np.float32(expl),
        iterations=np.int64(args.iterations),
    )
    print(f"wrote oracle policy to {args.out} ({len(keys)} infosets)")


if __name__ == "__main__":
    main()
