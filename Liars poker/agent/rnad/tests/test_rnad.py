"""
Tests for M3 R-NaD components. Run via:

    cd "papers/Liars poker/"
    python -m agent.rnad.tests.test_rnad
"""

import copy
import os
import sys

_PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PAPER_DIR not in sys.path:
    sys.path.insert(0, _PAPER_DIR)

import numpy as np
import torch

from agent.game.bids import NUM_BIDS, NUM_ACTIONS, CALL_ACTION
from agent.game.engine import new_match
from agent.rnad.config import RNaDConfig
from agent.rnad.network import LiarsPokerNet, _mask_logits
from agent.rnad.trainer import (
    Step, collect_round, compute_rnad_returns, compute_loss,
    RNaDTrainer,
)

# Shared tiny config for fast tests
def _fast_cfg(**kwargs) -> RNaDConfig:
    defaults = dict(
        num_iterations       = 10,
        episodes_per_update  = 8,
        eval_freq            = 9999,
        checkpoint_freq      = 9999,
        log_freq             = 9999,
        use_warm_start       = True,
        use_aux_loss         = True,
    )
    defaults.update(kwargs)
    return RNaDConfig(**defaults)


# ---------------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------------

def test_network_forward_shape():
    """Policy logits must be (NUM_ACTIONS,) and value must be (1,)."""
    cfg = _fast_cfg()
    net = LiarsPokerNet(cfg)
    state = new_match(2)
    state.hand_sizes = [1, 1]
    state.start_next_round()
    info    = state.info_state(0)
    obs     = net.encode_obs(info)

    assert obs.shape == (net.trunk_input_dim,), f"obs shape {obs.shape}"

    logits, value = net(obs)
    assert logits.shape == (NUM_ACTIONS,), f"logits shape {logits.shape}"
    assert value.shape  == (1,),           f"value shape {value.shape}"
    print("  test_network_forward_shape: PASS")


def test_action_masking():
    """Masked illegal actions must have −inf logits → zero probability."""
    cfg = _fast_cfg()
    net = LiarsPokerNet(cfg)
    state = new_match(2)
    state.hand_sizes = [1, 1]
    state.start_next_round()
    info  = state.info_state(0)
    obs   = net.encode_obs(info)
    logits, _ = net(obs)
    legal = state.legal_actions()

    masked = _mask_logits(logits, legal)
    probs  = torch.softmax(masked, dim=-1)

    illegal = [a for a in range(NUM_ACTIONS) if a not in legal]
    for a in illegal:
        assert probs[a].item() < 1e-7, f"illegal action {a} has prob {probs[a]:.6f}"

    # Legal actions should have positive probability
    for a in legal:
        assert probs[a].item() > 0.0, f"legal action {a} has zero prob"

    print("  test_action_masking: PASS")


def test_value_in_range():
    """Value head output must be in (−1, 1) due to Tanh."""
    cfg = _fast_cfg()
    net = LiarsPokerNet(cfg)
    for hand_size in [1, 2, 3]:
        state = new_match(2)
        state.hand_sizes = [hand_size, hand_size]
        state.start_next_round()
        info = state.info_state(0)
        obs  = net.encode_obs(info)
        _, v = net(obs)
        assert -1.0 < v.item() < 1.0, f"value {v.item()} out of (-1,1) for hand_size={hand_size}"
    print("  test_value_in_range: PASS")


def test_act_returns_legal():
    """net.act() must always return an action in legal_actions."""
    cfg = _fast_cfg()
    net = LiarsPokerNet(cfg)
    state = new_match(2)
    state.hand_sizes = [1, 1]
    state.start_next_round()
    info  = state.info_state(0)
    legal = state.legal_actions()
    for _ in range(20):
        action, lp, val = net.act(info, legal)
        assert action in legal, f"act() returned illegal action {action}"
        assert isinstance(lp, float)
        assert isinstance(val, float)
    print("  test_act_returns_legal: PASS")


# ---------------------------------------------------------------------------
# Episode collection tests
# ---------------------------------------------------------------------------

def test_collect_round_rewards():
    """Every step must have reward ∈ {+1, −1}."""
    cfg    = _fast_cfg()
    policy = LiarsPokerNet(cfg)
    anchor = copy.deepcopy(policy)
    for p in anchor.parameters(): p.requires_grad_(False)

    steps = collect_round(policy, anchor, hand_size=1, num_players=2)
    assert len(steps) >= 2, f"too few steps: {len(steps)}"

    for s in steps:
        assert s.reward in (1.0, -1.0), f"reward {s.reward} ∉ {{+1,−1}}"
        assert s.action in s.legal_actions, "collected action not in legal_actions"

    seats = [s.seat for s in steps]
    assert 0 in seats and 1 in seats, "both seats should act in a round"
    print("  test_collect_round_rewards: PASS")


def test_zero_sum_rewards():
    """In a 2-player round, exactly one seat wins and one loses."""
    cfg    = _fast_cfg()
    policy = LiarsPokerNet(cfg)
    anchor = copy.deepcopy(policy)
    for p in anchor.parameters(): p.requires_grad_(False)

    for _ in range(20):
        steps = collect_round(policy, anchor, hand_size=1, num_players=2)
        seat_rewards = {}
        for s in steps:
            seat_rewards[s.seat] = s.reward

        rewards = sorted(seat_rewards.values())
        assert rewards == [-1.0, 1.0], f"rewards {rewards} not zero-sum"

    print("  test_zero_sum_rewards: PASS")


# ---------------------------------------------------------------------------
# R-NaD return tests
# ---------------------------------------------------------------------------

def test_rnad_returns_sign():
    """Winner's transformed returns should be positive; loser's negative (for small η)."""
    cfg    = _fast_cfg()
    policy = LiarsPokerNet(cfg)
    anchor = copy.deepcopy(policy)
    for p in anchor.parameters(): p.requires_grad_(False)

    steps = collect_round(policy, anchor, hand_size=1, num_players=2)
    compute_rnad_returns(steps, eta=0.01)  # tiny η → returns ≈ ±1

    for s in steps:
        expected_sign = 1.0 if s.reward == 1.0 else -1.0
        assert (
            expected_sign * s.transformed_return > 0
        ), f"seat {s.seat}: reward {s.reward}, return {s.transformed_return:.4f}"

    print("  test_rnad_returns_sign: PASS")


def test_rnad_returns_bounded():
    """Transformed returns must stay in a reasonable range even with large η."""
    cfg    = _fast_cfg()
    policy = LiarsPokerNet(cfg)
    anchor = copy.deepcopy(policy)
    for p in anchor.parameters(): p.requires_grad_(False)

    all_steps = []
    for _ in range(50):
        steps = collect_round(policy, anchor, hand_size=1, num_players=2)
        all_steps.extend(steps)

    compute_rnad_returns(all_steps, eta=1.0)

    returns = [s.transformed_return for s in all_steps]
    assert max(abs(r) for r in returns) < 30.0, \
        f"returns out of expected range: {max(abs(r) for r in returns):.2f}"
    print("  test_rnad_returns_bounded: PASS")


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

def test_loss_backward():
    """
    Loss backward must not raise and must compute gradients on all reachable
    parameters.  Use hand_size=3 (pool n=6) so warm-start aux targets are
    generated, exercising the aux_head.
    """
    cfg    = _fast_cfg()
    policy = LiarsPokerNet(cfg)
    anchor = copy.deepcopy(policy)
    for p in anchor.parameters(): p.requires_grad_(False)

    steps = []
    for _ in range(32):
        # hand_size=3 → n=6 (in warm-start range) so aux targets are produced
        steps.extend(collect_round(policy, anchor, hand_size=3, num_players=2,
                                   warm_start=__import__(
                                       "agent.rnad.warm_start",
                                       fromlist=["WarmStartLookup"]
                                   ).WarmStartLookup()))
    compute_rnad_returns(steps, eta=cfg.eta)

    policy.train()
    loss, metrics = compute_loss(policy, steps, cfg, torch.device("cpu"))
    loss.backward()

    for name, param in policy.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"no gradient for {name}"

    # All metric values should be finite
    for k, v in metrics.items():
        assert np.isfinite(v), f"metric {k} = {v} is not finite"

    # aux_loss should be non-zero when aux targets are present
    assert metrics["loss/aux"] > 0.0, "aux_loss should be > 0 with warm_start targets"

    print("  test_loss_backward: PASS")


# ---------------------------------------------------------------------------
# Full training smoke test
# ---------------------------------------------------------------------------

def test_short_training_run():
    """20 iterations of training must not crash and loss must be finite."""
    cfg = _fast_cfg(num_iterations=20, episodes_per_update=16)
    trainer = RNaDTrainer(cfg)
    trainer.train()

    # Final loss should be finite
    print("  test_short_training_run: PASS")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running R-NaD tests...\n")
    tests = [
        test_network_forward_shape,
        test_action_masking,
        test_value_in_range,
        test_act_returns_legal,
        test_collect_round_rewards,
        test_zero_sum_rewards,
        test_rnad_returns_sign,
        test_rnad_returns_bounded,
        test_loss_backward,
        test_short_training_run,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {t.__name__}: FAIL — {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed.")
    if failed:
        sys.exit(1)
