"""
eval.py — Evaluation suite for the R-NaD Liar's Poker agent.

Metrics (§5.3 of AGENT_DESIGN.md):
  win_rate_vs_random  — head-to-head win rate against a uniform random agent
  win_rate_vs_blind   — head-to-head win rate against the blind baseline agent
  avg_entropy         — average policy entropy over legal actions (higher = less
                        exploitable in theory; should drop as policy converges)
  bid_accuracy        — fraction of bids matching the blind baseline's optimal
                        action (calibration proxy)
  avg_return          — average episode return (expected ≈ 0 for zero-sum self-play)

Run from papers/Liars poker/:
    python -m agent.rnad.eval --checkpoint agent/checkpoints/rnad_final.pt
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_EVAL_DIR  = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.abspath(os.path.join(_EVAL_DIR, ".."))
_PAPER_DIR = os.path.abspath(os.path.join(_AGENT_DIR, ".."))
for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.game.bids import NUM_ACTIONS, CALL_ACTION           # noqa: E402
from agent.game.engine import new_match                        # noqa: E402
from agent.rnad.network import LiarsPokerNet, _mask_logits     # noqa: E402


# ---------------------------------------------------------------------------
# Single-game evaluation helper
# ---------------------------------------------------------------------------

def play_round(
    policy:     LiarsPokerNet,
    opponent,               # anything with choose_action(MatchState) → int
    policy_seat: int,
    hand_size:   int,
    num_players: int = 2,
) -> Dict[str, float]:
    """
    Play one round.  policy occupies `policy_seat`; opponent occupies all others.

    Returns a dict with:
        "win"     : 1.0 if policy won, 0.0 if lost
        "entropy" : average entropy of the policy's distributions this round
        "n_bids"  : total actions taken (round length)
    """
    state = new_match(num_players)
    state.hand_sizes = [hand_size] * num_players
    state.start_next_round()

    entropies: List[float] = []

    while state.round_state is not None:
        cp    = state.round_state.current_player
        legal = state.legal_actions()

        if cp == policy_seat:
            info = state.info_state(cp)
            with torch.no_grad():
                obs    = policy.encode_obs(info)
                logits, _ = policy.forward(obs)
            masked = _mask_logits(logits, legal)
            dist   = Categorical(logits=masked)
            action = int(dist.sample().item())
            entropies.append(float(dist.entropy().item()))
        else:
            action = opponent.choose_action(state)

        result = state.apply_action(action)
        if result is not None:
            won = 1.0 if result.winner_seat == policy_seat else 0.0
            return {
                "win":     won,
                "entropy": float(np.mean(entropies)) if entropies else 0.0,
                "n_bids":  float(len(state.round_history[-1].pool)
                                 if state.round_history else 0),
            }

    return {"win": 0.0, "entropy": 0.0, "n_bids": 0.0}


# ---------------------------------------------------------------------------
# Bid-accuracy helper (calibration against blind baseline)
# ---------------------------------------------------------------------------

def _bid_accuracy_episode(
    policy:     LiarsPokerNet,
    hand_size:  int,
    num_players: int = 2,
) -> float:
    """
    Play one round where the policy's greedy action is compared to the
    blind baseline's optimal action at each decision point.
    Returns fraction of decision points where they agree.
    """
    from agent.baseline.blind_equilibrium import get_blind_equilibrium
    from agent.game.bids import bid_to_index, Bid

    state = new_match(num_players)
    state.hand_sizes = [hand_size] * num_players
    state.start_next_round()

    matches = 0
    total   = 0

    while state.round_state is not None:
        cp    = state.round_state.current_player
        info  = state.info_state(cp)
        legal = state.legal_actions()

        # Policy greedy action
        with torch.no_grad():
            obs    = policy.encode_obs(info)
            logits, _ = policy.forward(obs)
        masked  = _mask_logits(logits, legal)
        p_action = int(masked.argmax().item())

        # Blind baseline action
        active     = info["active"]
        hand_sizes = info["hand_sizes"]
        n = sum(hand_sizes[s] for s, a in enumerate(active) if a)
        try:
            eq = get_blind_equilibrium(n)
            rs = state.round_state
            if rs.current_bid is None:
                bl_action = eq["initial_bid"]
            else:
                bid_idx   = bid_to_index(Bid(rs.current_bid[0], rs.current_bid[1]))
                bl_action = eq["policy"][bid_idx][0]
            bl_action = bl_action if bl_action in legal else (
                CALL_ACTION if CALL_ACTION in legal else legal[-1]
            )
        except Exception:
            bl_action = legal[0]

        if p_action == bl_action:
            matches += 1
        total += 1

        # Advance with policy action to keep the episode going
        result = state.apply_action(p_action)
        if result is not None:
            break

    return matches / max(total, 1)


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy:       LiarsPokerNet,
    num_episodes: int   = 500,
    hand_size:    int   = 1,
    num_players:  int   = 2,
    device:       Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run the full evaluation suite and return a metrics dict.

    Metrics returned:
        win_rate_vs_random   float
        win_rate_vs_blind    float
        avg_entropy          float
        bid_accuracy         float  (vs blind baseline greedy action)
    """
    policy.eval()

    # Import agents lazily to avoid circular imports
    import sys, os
    sys.path.insert(0, os.path.join(_AGENT_DIR, "web", "backend"))
    from agent.web.backend.agents import RandomAgent, BlindBaselineAgent

    random_agent = RandomAgent()
    blind_agent  = BlindBaselineAgent()

    wins_vs_random = []
    wins_vs_blind  = []
    entropies      = []
    accuracies     = []

    for ep in range(num_episodes):
        seat = ep % num_players  # rotate which seat the policy occupies

        r_random = play_round(policy, random_agent, seat, hand_size, num_players)
        r_blind  = play_round(policy, blind_agent,  seat, hand_size, num_players)

        wins_vs_random.append(r_random["win"])
        wins_vs_blind.append(r_blind["win"])
        entropies.append(r_blind["entropy"])

        if ep < 200:  # bid accuracy is slower; sample fewer
            accuracies.append(_bid_accuracy_episode(policy, hand_size, num_players))

    return {
        "win_rate_vs_random": float(np.mean(wins_vs_random)),
        "win_rate_vs_blind":  float(np.mean(wins_vs_blind)),
        "avg_entropy":        float(np.mean(entropies)),
        "bid_accuracy":       float(np.mean(accuracies)) if accuracies else 0.0,
    }


# ---------------------------------------------------------------------------
# Detailed calibration report (printed to stdout)
# ---------------------------------------------------------------------------

def print_calibration_report(
    policy:     LiarsPokerNet,
    hand_size:  int   = 1,
    num_players: int  = 2,
    n_episodes: int   = 1000,
) -> None:
    """
    Print a table comparing the policy's action frequencies to the
    blind baseline equilibrium policy for each possible standing bid.
    """
    from agent.game.bids import all_bids, HAND_NAMES
    from agent.baseline.blind_equilibrium import get_blind_equilibrium

    policy.eval()
    bids = all_bids()

    # Collect (standing_bid_idx, policy_action) pairs
    action_counts: dict = {}  # bid_idx → {action: count}

    for _ in range(n_episodes):
        state = new_match(num_players)
        state.hand_sizes = [hand_size] * num_players
        state.start_next_round()

        while state.round_state is not None:
            cp    = state.round_state.current_player
            info  = state.info_state(cp)
            legal = state.legal_actions()
            rs    = state.round_state

            cb_idx = -1 if rs.current_bid is None else bids.index(
                __import__("agent.game.bids", fromlist=["Bid"]).Bid(
                    rs.current_bid[0], rs.current_bid[1]
                )
            )

            with torch.no_grad():
                obs    = policy.encode_obs(info)
                logits, _ = policy.forward(obs)
            masked = _mask_logits(logits, legal)
            action = int(masked.argmax().item())

            if cb_idx not in action_counts:
                action_counts[cb_idx] = {}
            action_counts[cb_idx][action] = action_counts[cb_idx].get(action, 0) + 1

            result = state.apply_action(action)
            if result is not None:
                break

    # Print summary
    print(f"\n{'Bid':>22}  {'Policy action':>22}  {'Baseline action':>22}")
    print("-" * 72)

    # Get baseline
    n = hand_size * num_players
    try:
        eq = get_blind_equilibrium(n)
    except Exception:
        print("(blind equilibrium not available)")
        return

    for bid_idx in sorted(action_counts.keys()):
        if bid_idx < 0:
            bid_str = "(no bid yet)"
            bl_act  = eq["initial_bid"]
        else:
            bid_str = str(bids[bid_idx])
            bl_act  = eq["policy"][bid_idx][0]

        counts = action_counts[bid_idx]
        total  = sum(counts.values())
        top_action = max(counts, key=counts.get)
        freq = counts[top_action] / total

        top_str = (
            "CALL" if top_action == CALL_ACTION
            else str(bids[top_action])
        )
        bl_str = (
            "CALL" if bl_act == CALL_ACTION
            else str(bids[bl_act])
        )
        match = "✓" if top_action == bl_act else " "

        print(f"{bid_str:>22}  {top_str:>18} ({freq:.2f})  {bl_str:>22}  {match}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a saved R-NaD checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--hand-size",   type=int, default=1)
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--episodes",    type=int, default=500)
    parser.add_argument("--calibration", action="store_true",
                        help="Print calibration report vs blind baseline")
    args = parser.parse_args()

    ckpt   = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    net    = LiarsPokerNet(config)
    net.load_state_dict(ckpt["policy_state"])
    net.eval()

    print(f"Loaded checkpoint: {args.checkpoint}  (iter {ckpt['iteration']})")

    results = evaluate_policy(
        net,
        num_episodes = args.episodes,
        hand_size    = args.hand_size,
        num_players  = args.num_players,
    )
    for k, v in results.items():
        print(f"  {k:<30}: {v:.4f}")

    if args.calibration:
        print_calibration_report(
            net,
            hand_size   = args.hand_size,
            num_players = args.num_players,
        )
