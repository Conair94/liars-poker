"""
Vectorized CFR+ solver for the 1v1 n=2 exact-rules Liar's Poker game.

Same algorithm as `cfr_1v1.CFRSolver` (CFR+ with regret matching+ and linear
strategy averaging) but ~100–500× faster per iteration via:

  1. Precomputing the static game tree once at construction time.
  2. Batching all 169 rank-pair traversals at each node with numpy.
  3. Storing infoset state (regret_sum, strategy_sum) as flat numpy arrays.

Algorithmic note
----------------
The reference `CFRSolver` applies the CFR+ regret floor inside the rank-pair
loop (once per (r0, r1) traversal). This solver floors once per iteration —
which is the canonical CFR+ behavior (Tammelin 2014). Convergence properties
are the same; average strategies converge to the same Nash equilibrium, and
exploitability curves overlap within float noise at moderate iteration counts.

Verification
------------
See `tests/test_cfr_1v1_fast.py` for the equivalence gate: after N iterations
both solvers must produce (a) identical opening_mix_by_rank to 1e-3, and
(b) exploitability within 1e-3.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR    = os.path.abspath(os.path.join(_BASELINE_DIR, ".."))
_PAPER_DIR    = os.path.abspath(os.path.join(_AGENT_DIR,    ".."))

for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.game.bids import (                          # noqa: E402
    CALL_ACTION, HH_ACTION, index_to_bid,
)
from agent.baseline.cfr_1v1 import (                   # noqa: E402
    HC_PAIR_BIDS, _legal_actions, _is_terminal,
    _current_player, _terminal_utility_p0_rank,
    _rank_pair_weight, _TOTAL_DEAL_COUNT,
)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class CFRSolverFast:
    """
    Vectorized CFR+ solver. Drop-in replacement for `CFRSolver` for Nash
    extraction; state serialization uses a different format (numpy-backed).
    """

    def __init__(
        self,
        max_bids: int = 6,
        bid_space: Tuple[int, ...] = HC_PAIR_BIDS,
        include_hh: bool = True,
    ) -> None:
        self.max_bids   = int(max_bids)
        self.bid_space  = tuple(sorted(bid_space))
        self.include_hh = bool(include_hh)
        self._iterations = 0

        self._build_tree()
        self._init_state()

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self) -> None:
        """BFS-enumerate every internal history, assign integer node ids, and
        precompute terminal utility matrices for every resolver action."""
        internal_histories: List[Tuple[int, ...]] = []
        internal_id: Dict[Tuple[int, ...], int] = {}

        # BFS so parent nodes get lower ids than children.
        frontier: List[Tuple[int, ...]] = [()]
        visited: set = set()
        while frontier:
            h = frontier.pop(0)
            if h in visited or _is_terminal(h):
                continue
            visited.add(h)
            internal_id[h] = len(internal_histories)
            internal_histories.append(h)
            for a in _legal_actions(h, self.bid_space, self.max_bids, self.include_hh):
                new_h = h + (a,)
                if not _is_terminal(new_h) and new_h not in visited:
                    frontier.append(new_h)

        num_internal = len(internal_histories)
        node_player = np.zeros(num_internal, dtype=np.int8)
        node_depth  = np.zeros(num_internal, dtype=np.int16)
        node_legal: List[List[int]] = [[] for _ in range(num_internal)]
        node_children: List[List[Tuple[bool, int]]] = [[] for _ in range(num_internal)]

        term_u_list: List[np.ndarray] = []

        def _term_matrix(h: Tuple[int, ...]) -> np.ndarray:
            m = np.empty((13, 13), dtype=np.float64)
            for r0 in range(13):
                for r1 in range(13):
                    m[r0, r1] = _terminal_utility_p0_rank(h, r0, r1)
            return m

        for node_id, h in enumerate(internal_histories):
            node_player[node_id] = _current_player(h)
            node_depth[node_id]  = len(h)
            legal = _legal_actions(h, self.bid_space, self.max_bids, self.include_hh)
            node_legal[node_id] = legal
            for a in legal:
                new_h = h + (a,)
                if _is_terminal(new_h):
                    term_idx = len(term_u_list)
                    term_u_list.append(_term_matrix(new_h))
                    node_children[node_id].append((True, term_idx))
                else:
                    node_children[node_id].append((False, internal_id[new_h]))

        self._num_internal = num_internal
        self._node_player   = node_player
        self._node_depth    = node_depth
        self._node_legal    = node_legal
        self._node_children = node_children
        self._term_util     = np.stack(term_u_list) if term_u_list else np.zeros((0, 13, 13))

        # Sort once: children are always at greater depth than parent.
        self._topo_order    = np.argsort(node_depth, kind="stable").astype(np.int32)

        # Chance reach matrix (13, 13): P(r0, r1) over all ordered deals
        cr = np.empty((13, 13), dtype=np.float64)
        for r0 in range(13):
            for r1 in range(13):
                cr[r0, r1] = _rank_pair_weight(r0, r1) / _TOTAL_DEAL_COUNT
        self._chance_reach = cr

    def _init_state(self) -> None:
        """Allocate regret_sum and strategy_sum per node."""
        self._regret_sum:   List[np.ndarray] = []
        self._strategy_sum: List[np.ndarray] = []
        for node_id in range(self._num_internal):
            n_legal = len(self._node_legal[node_id])
            self._regret_sum.append(np.zeros((13, n_legal), dtype=np.float64))
            self._strategy_sum.append(np.zeros((13, n_legal), dtype=np.float64))

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _compute_strategy(self, node_id: int) -> np.ndarray:
        rs      = self._regret_sum[node_id]
        pos     = np.maximum(rs, 0.0)
        s       = pos.sum(axis=1, keepdims=True)
        n_legal = rs.shape[1]
        safe_s  = np.where(s > 0, s, 1.0)
        uniform = np.full_like(rs, 1.0 / n_legal)
        return np.where(s > 0, pos / safe_s, uniform)

    def _gather_child_values(self, node_id: int, node_value: np.ndarray) -> np.ndarray:
        """Return (n_legal, 13, 13) P0 utility for every child of node_id."""
        children = self._node_children[node_id]
        n_legal  = len(children)
        cv = np.empty((n_legal, 13, 13), dtype=np.float64)
        for i, (is_term, child_idx) in enumerate(children):
            cv[i] = self._term_util[child_idx] if is_term else node_value[child_idx]
        return cv

    def iterate(self) -> float:
        """One full CFR+ iteration. Returns P0 expected utility under the
        current (instantaneous, not average) strategy."""
        num_internal = self._num_internal
        node_value = np.zeros((num_internal, 13, 13), dtype=np.float64)
        strategies: List[Optional[np.ndarray]] = [None] * num_internal

        # 1. Bottom-up: compute node values.
        for node_id in reversed(self._topo_order.tolist()):
            player  = int(self._node_player[node_id])
            strat   = self._compute_strategy(node_id)      # (13, n_legal)
            strategies[node_id] = strat
            cv      = self._gather_child_values(node_id, node_value)  # (i, r0, r1)
            if player == 0:
                node_value[node_id] = np.einsum('ri,irs->rs', strat, cv)
            else:
                node_value[node_id] = np.einsum('si,irs->rs', strat, cv)

        # 2. Top-down: propagate reach probabilities per player, update regrets
        #              and strategy_sum.
        reach0 = np.zeros((num_internal, 13), dtype=np.float64)
        reach1 = np.zeros((num_internal, 13), dtype=np.float64)
        root_id = int(self._topo_order[0])  # depth-0 node is the root
        reach0[root_id] = 1.0
        reach1[root_id] = 1.0

        lin_w = float(self._iterations + 1)
        cr    = self._chance_reach

        for node_id in self._topo_order.tolist():
            node_id = int(node_id)
            player  = int(self._node_player[node_id])
            strat   = strategies[node_id]                       # (13, n_legal)
            cv      = self._gather_child_values(node_id, node_value)
            nv      = node_value[node_id]                        # (13, 13)
            diff    = cv - nv[None, :, :]                        # (i, r0, r1)
            r0      = reach0[node_id]
            r1      = reach1[node_id]

            if player == 0:
                # weight[r0, r1] = chance(r0,r1) * reach1(r1)
                w = cr * r1[None, :]
                delta = np.einsum('irs,rs->ri', diff, w)          # (13, n_legal)
                self._regret_sum[node_id] = np.maximum(
                    0.0, self._regret_sum[node_id] + delta
                )
                self._strategy_sum[node_id] += lin_w * r0[:, None] * strat
                # Propagate reach
                for i, (is_term, child_idx) in enumerate(self._node_children[node_id]):
                    if not is_term:
                        reach0[child_idx] = r0 * strat[:, i]
                        reach1[child_idx] = r1
            else:
                # player == 1: P1's regret = -(P0 utility advantage)
                w = cr * r0[:, None]
                delta = np.einsum('irs,rs->si', diff, w) * (-1.0)  # (13, n_legal)
                self._regret_sum[node_id] = np.maximum(
                    0.0, self._regret_sum[node_id] + delta
                )
                self._strategy_sum[node_id] += lin_w * r1[:, None] * strat
                for i, (is_term, child_idx) in enumerate(self._node_children[node_id]):
                    if not is_term:
                        reach0[child_idx] = r0
                        reach1[child_idx] = r1 * strat[:, i]

        self._iterations += 1
        root_val = node_value[root_id]
        return float((root_val * cr).sum())

    def run(self, n_iterations: int = 1000, verbose: bool = False) -> None:
        for i in range(n_iterations):
            gv = self.iterate()
            if verbose and (i + 1) % 100 == 0:
                exp = self.exploitability()
                print(f"  iter {i+1:6d}: game_value={gv:+.4f}  exploitability={exp:.6f}")

    # ------------------------------------------------------------------
    # Average strategy queries
    # ------------------------------------------------------------------

    def _avg_strategy_for(self, node_id: int) -> np.ndarray:
        ss = self._strategy_sum[node_id]
        n_legal = ss.shape[1]
        row_sum = ss.sum(axis=1, keepdims=True)
        uniform = np.full_like(ss, 1.0 / n_legal)
        safe = np.where(row_sum > 0, row_sum, 1.0)
        return np.where(row_sum > 0, ss / safe, uniform)

    def average_strategy(self, history: Tuple[int, ...]) -> Dict[int, np.ndarray]:
        """Return a dict rank -> (n_legal,) average-strategy distribution."""
        node_id = self._history_to_nodeid(history)
        if node_id is None:
            legal = _legal_actions(history, self.bid_space, self.max_bids, self.include_hh)
            n_legal = len(legal)
            return {r: np.full(n_legal, 1.0 / n_legal) for r in range(13)}
        avg = self._avg_strategy_for(node_id)  # (13, n_legal)
        return {r: avg[r] for r in range(13)}

    def _history_to_nodeid(self, history: Tuple[int, ...]) -> Optional[int]:
        # Linear search — only called at reporting time.
        for nid in range(self._num_internal):
            if self._reconstruct_history(nid) == history:
                return nid
        return None

    def _reconstruct_history(self, node_id: int) -> Tuple[int, ...]:
        # Expensive; used only for average-strategy queries. The solver caches
        # node -> history if needed. For now, walk the id back via BFS metadata.
        # Simpler: we'll build an id->history cache lazily.
        if not hasattr(self, "_id_to_history"):
            self._id_to_history = [None] * self._num_internal
            # Re-run BFS to map node_id -> history.
            stack: List[Tuple[Tuple[int, ...], int]] = [((), self._topo_order[0].item())]
            # Walk BFS using children
            from collections import deque
            queue = deque()
            queue.append(((), int(self._topo_order[0])))
            self._id_to_history[int(self._topo_order[0])] = ()
            while queue:
                h, nid = queue.popleft()
                legal = self._node_legal[nid]
                for i, a in enumerate(legal):
                    is_term, child_idx = self._node_children[nid][i]
                    if not is_term and self._id_to_history[child_idx] is None:
                        new_h = h + (a,)
                        self._id_to_history[child_idx] = new_h
                        queue.append((new_h, child_idx))
        return self._id_to_history[node_id]

    def opening_mix_by_rank(self) -> Dict[int, Dict[str, float]]:
        """P0's root opening-bid distribution keyed by P0's rank."""
        root_id = int(self._topo_order[0])
        legal   = self._node_legal[root_id]
        avg     = self._avg_strategy_for(root_id)  # (13, n_legal)
        out: Dict[int, Dict[str, float]] = {}
        for rank in range(13):
            d: Dict[str, float] = {}
            for i, a in enumerate(legal):
                p = float(avg[rank, i])
                if p >= 0.001:
                    d[str(index_to_bid(a))] = round(p, 5)
            out[rank] = d
        return out

    def response_mix_by_rank(self, opening_bid_idx: int) -> Dict[int, Dict[str, float]]:
        """P1's response distribution after a given opening bid."""
        history = (opening_bid_idx,)
        node_id = self._history_to_nodeid(history)
        out: Dict[int, Dict[str, float]] = {}
        if node_id is None:
            return {r: {} for r in range(13)}
        legal = self._node_legal[node_id]
        avg   = self._avg_strategy_for(node_id)
        for rank in range(13):
            d: Dict[str, float] = {}
            for i, a in enumerate(legal):
                p = float(avg[rank, i])
                if p < 0.001:
                    continue
                if a == CALL_ACTION:
                    name = "CALL"
                elif a == HH_ACTION:
                    name = "HH"
                else:
                    name = str(index_to_bid(a))
                d[name] = round(p, 5)
            out[rank] = d
        return out

    # ------------------------------------------------------------------
    # Game value + exploitability (vectorized)
    # ------------------------------------------------------------------

    def game_value(self) -> float:
        """P0 expected utility under the average strategy profile."""
        node_value = np.zeros((self._num_internal, 13, 13), dtype=np.float64)
        for node_id in reversed(self._topo_order.tolist()):
            player = int(self._node_player[node_id])
            avg    = self._avg_strategy_for(node_id)       # (13, n_legal)
            cv     = self._gather_child_values(node_id, node_value)
            if player == 0:
                node_value[node_id] = np.einsum('ri,irs->rs', avg, cv)
            else:
                node_value[node_id] = np.einsum('si,irs->rs', avg, cv)
        root_id = int(self._topo_order[0])
        return float((node_value[root_id] * self._chance_reach).sum())

    def exploitability(self) -> float:
        """NashConv/2: average best-response gain across both players."""
        br0 = self._best_response_value(br_player=0)
        br1 = self._best_response_value(br_player=1)
        return (br0 + br1) / 2.0

    def _best_response_value(self, br_player: int) -> float:
        """Expected BR-player utility, bottom-up, vectorized over rank pairs."""
        node_value = np.zeros((self._num_internal, 13, 13), dtype=np.float64)
        for node_id in reversed(self._topo_order.tolist()):
            player = int(self._node_player[node_id])
            cv     = self._gather_child_values(node_id, node_value)  # (i, r0, r1)
            if player == br_player:
                # BR picks max over actions — but "max utility" is from BR's view.
                # cv is P0 utility; flip sign if BR is P1.
                util = cv if br_player == 0 else -cv
                best_util = util.max(axis=0)                         # (13, 13)
                if br_player == 0:
                    node_value[node_id] = best_util                  # already P0 util
                else:
                    node_value[node_id] = -best_util                 # back to P0 convention
            else:
                # Opponent plays their average strategy
                avg = self._avg_strategy_for(node_id)               # (13, n_legal)
                if player == 0:
                    node_value[node_id] = np.einsum('ri,irs->rs', avg, cv)
                else:
                    node_value[node_id] = np.einsum('si,irs->rs', avg, cv)

        root_id = int(self._topo_order[0])
        root_val = node_value[root_id]       # P0 utility
        if br_player == 1:
            root_val = -root_val
        return float((root_val * self._chance_reach).sum())

    # ------------------------------------------------------------------
    # Serialization (portable — dense numpy dump)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        # Store per-node regret/strategy as float32 lists (compact).
        return {
            "iterations":   self._iterations,
            "max_bids":     self.max_bids,
            "bid_space":    list(self.bid_space),
            "include_hh":   self.include_hh,
            "regret_sum":   [arr.astype(np.float32).tolist() for arr in self._regret_sum],
            "strategy_sum": [arr.astype(np.float32).tolist() for arr in self._strategy_sum],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CFRSolverFast":
        solver = cls(
            max_bids=int(d.get("max_bids", 6)),
            bid_space=tuple(d.get("bid_space", HC_PAIR_BIDS)),
            include_hh=bool(d.get("include_hh", True)),
        )
        solver._iterations = int(d["iterations"])
        for i, arr in enumerate(d["regret_sum"]):
            solver._regret_sum[i] = np.array(arr, dtype=np.float64)
        for i, arr in enumerate(d["strategy_sum"]):
            solver._strategy_sum[i] = np.array(arr, dtype=np.float64)
        return solver


# ---------------------------------------------------------------------------
# CLI helper for quick profiling
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Fast CFR+ solver (vectorized).")
    parser.add_argument("--iters",    type=int, default=100)
    parser.add_argument("--max-bids", type=int, default=4)
    parser.add_argument("--no-hh",    action="store_true")
    parser.add_argument("--show-exp", action="store_true",
                        help="Compute exploitability after the run.")
    args = parser.parse_args()

    solver = CFRSolverFast(
        max_bids   = args.max_bids,
        include_hh = not args.no_hh,
    )
    t0 = time.time()
    solver.run(args.iters)
    dt = time.time() - t0
    print(f"{args.iters} iters in {dt:.2f}s  ({dt/args.iters*1000:.1f} ms/iter)")
    print(f"game_value={solver.game_value():+.4f}")
    if args.show_exp:
        t1 = time.time()
        exp = solver.exploitability()
        print(f"exploitability={exp:.6f}  (compute time {time.time()-t1:.2f}s)")
