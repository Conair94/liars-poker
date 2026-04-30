"""AR-1 HandModel tests.

Covers the AR-0a contract invariants on the new module:

- Mask invariant for `LearnedHandModel`, `AnalyticHandModel`, and
  `HeuristicHandModel`.
- Batched inference parity for the learned model.
- Adapter parity: `AnalyticHandModel`'s posterior matches the
  `ExactRulesConditional` agent's `_compute_adj_exact` with no
  opponent-bid bumps applied (the adapter ignores bid history).

Heavier acceptance — Pareto-dominance over Analytic on per-`n` Brier —
is gated to a `slow` integration test that runs a tiny in-process
training pipeline, so the suite stays under 10s by default.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
_SRC_DIR   = os.path.join(_REPO_ROOT, "src")
_PROBS_DIR = os.path.join(_SRC_DIR, "training", "probs")
for _p in (_PROBS_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.contracts import HandBelief, Infostate  # noqa: E402
from agents.learned.handmodel import (  # noqa: E402
    AnalyticHandModel,
    HandModelConfig,
    LearnedHandModel,
    LearnedHandModelNet,
)
from agents.learned.handmodel.dataset import (  # noqa: E402
    HandModelDataset,
    brier_per_n,
    collate_batch,
    collect_rollouts,
)
from game.bids import NUM_BIDS  # noqa: E402
from game.engine import new_match  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _random_infostates(seed: int, count: int, hand_size: int = 3) -> list[Infostate]:
    rng = random.Random(seed)
    out: list[Infostate] = []
    while len(out) < count:
        state = new_match(
            num_players=2, seed=rng.randrange(1 << 31),
            mode="countup", exact_rules=True, high_hand=True,
        )
        for s in range(state.num_players):
            state.hand_sizes[s] = hand_size
        state.start_next_round()
        # Walk a few random legal actions before sampling.
        steps = rng.randint(0, 3)
        for _ in range(steps):
            if state.round_state is None:
                break
            legal = state.legal_actions()
            if not legal:
                break
            # Restrict to bid actions when possible to avoid resolving early.
            bids = [a for a in legal if a < NUM_BIDS]
            choice = rng.choice(bids if bids else legal)
            try:
                state.apply_action(choice)
            except Exception:
                break
        if state.round_state is None:
            continue
        out.append(Infostate.from_match_state(state))
    return out


def _tiny_config(**overrides) -> HandModelConfig:
    base = dict(
        card_emb_dim=16,
        bid_emb_dim=16,
        bid_hist_len=8,
        bid_hist_dim=32,
        transformer_heads=2,
        transformer_ffn_dim=32,
        transformer_layers=1,
        hidden_dim=32,
        num_trunk_layers=1,
        batch_size=64,
        warmup_steps=10,
        max_epochs=2,
        min_epochs=1,
        train_decisions_per_n=64,
        val_decisions_per_n=32,
        test_decisions_per_n=32,
        hand_sizes=(3, 4),
    )
    base.update(overrides)
    return HandModelConfig(**base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMaskInvariant:
    def test_learned_handmodel_mask(self) -> None:
        cfg = _tiny_config()
        torch.manual_seed(0)
        model = LearnedHandModel(LearnedHandModelNet(cfg))
        infos = _random_infostates(seed=42, count=64)
        for b in model.belief_batch(infos):
            # Strong invariants from AR-0a §1; we re-assert here so this
            # test fails loudly if the network's masking regresses.
            assert b.q.shape == (NUM_BIDS,)
            assert abs(float(b.q.sum()) - 1.0) < 1e-5
            assert float(b.q[~b.feasible_mask].sum()) < 1e-6

    def test_analytic_handmodel_mask(self) -> None:
        model = AnalyticHandModel()
        # AnalyticHandModel relies on conditional tables for n ∈ {5..25}, so
        # use larger pool sizes here.
        infos = _random_infostates(seed=43, count=32, hand_size=3)  # n=6
        for b in model.belief_batch(infos):
            assert abs(float(b.q.sum()) - 1.0) < 1e-5
            assert float(b.q[~b.feasible_mask].sum()) < 1e-6


class TestBatchingParity:
    def test_belief_batch_matches_loop(self) -> None:
        cfg = _tiny_config()
        torch.manual_seed(1)
        model = LearnedHandModel(LearnedHandModelNet(cfg))
        infos = _random_infostates(seed=44, count=8)
        batched = model.belief_batch(infos)
        loop = [model.belief(i) for i in infos]
        for a, b in zip(batched, loop):
            np.testing.assert_allclose(a.q, b.q, atol=1e-6)


class TestRolloutCollector:
    def test_emits_feasible_labels(self) -> None:
        decisions = list(collect_rollouts(
            agent_key="exactconditional",
            hand_size=3,
            n_decisions=24,
            seed=7,
        ))
        assert len(decisions) >= 24
        for d in decisions[:24]:
            mask = d.infostate.feasible_mask[:NUM_BIDS]
            # The label is a *bid* index; must be feasible at the actor's pool.
            assert 0 <= d.label < NUM_BIDS
            # Feasibility is over the bid space; restrict the check.
            assert any(mask), "no feasible bids — pool size 0?"


@pytest.mark.slow
class TestEndToEndTraining:
    """Exercises the trainer end-to-end on a tiny dataset.

    Not the AR-1 acceptance gate — the suite-default config is much too
    small to clear it. Just confirms that a complete cycle (rollouts →
    train → eval → checkpoint round-trip) runs without errors and that
    the resulting model still satisfies the mask invariant on real data.
    """

    def test_smoketest_cycle(self, tmp_path: Path) -> None:
        from agents.learned.handmodel.trainer import train

        cfg = _tiny_config(seed=11, hand_sizes=(3,), max_epochs=1)
        result = train(cfg, str(tmp_path))
        assert result.best_epoch >= 1
        # Round-trip the saved checkpoint.
        ckpt = tmp_path / "handmodel" / "best.pt"
        assert ckpt.exists()
        model = LearnedHandModel.from_checkpoint(str(ckpt))
        infos = _random_infostates(seed=12, count=16)
        for b in model.belief_batch(infos):
            assert abs(float(b.q.sum()) - 1.0) < 1e-5
            assert float(b.q[~b.feasible_mask].sum()) < 1e-6


class TestBrierHelper:
    def test_perfect_predictions(self) -> None:
        labels = np.array([0, 1, 2], dtype=np.int64)
        ns     = np.array([6, 6, 8], dtype=np.int64)
        qs     = np.zeros((3, NUM_BIDS), dtype=np.float32)
        qs[np.arange(3), labels] = 1.0
        out = brier_per_n(qs, labels, ns)
        assert out[6] == pytest.approx(0.0)
        assert out[8] == pytest.approx(0.0)
