"""
trainer.py — R-NaD training loop for card-based Liar's Poker.

Algorithm (§5.1 of AGENT_DESIGN.md):
  Outer iteration t:
    1. Collect B episodes under policy π_θ.
    2. For each decision point (s, a) by player i:
         r̃_t  =  r_t  −  η · (log π_θ(a|s) − log π_reg(a|s))
         G̃_t  =  r̃_t  +  γ · G̃_{t+1}     (backward cumulative sum)
    3. Policy gradient:   L_π  = −Σ log π_θ(a|s) · (G̃_t − V_θ(s))
       Value regression:  L_V  = Σ (V_θ(s) − G̃_t)²
       Entropy bonus:     L_H  = −Σ H(π_θ(·|s))
       Aux (optional):    L_aux = Σ CE(aux_pred, conditional_target)
    4. Update θ via Adam.
    5. Every anchor_update_freq iterations: π_reg ← π_θ.

Stage A curriculum: both seats frozen at hand_size = config.stage_a_hand_size.
Each episode is one round (bid → call → resolution).
Reward for each seat: +1 (winner), −1 (loser).

The same policy network is shared across both seats (symmetric self-play).

Run from papers/Liars poker/:
    python -m agent.rnad.trainer
"""

from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_RNAD_DIR  = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.abspath(os.path.join(_RNAD_DIR, ".."))
_PAPER_DIR = os.path.abspath(os.path.join(_AGENT_DIR, ".."))
for _p in (_PAPER_DIR, _AGENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.game.bids import NUM_ACTIONS, CALL_ACTION          # noqa: E402
from agent.game.engine import MatchState, new_match           # noqa: E402
from agent.rnad.config import RNaDConfig                      # noqa: E402
from agent.rnad.network import LiarsPokerNet, _mask_logits    # noqa: E402


# ---------------------------------------------------------------------------
# Trajectory step
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One decision point in an episode."""
    info:           dict          # raw info_state dict (stored for re-encoding)
    action:         int           # chosen action index
    log_prob_old:   float         # log π_θ(a|s) at collection time (no grad)
    log_prob_reg:   float         # log π_reg(a|s) at collection time (no grad)
    value_old:      float         # V_θ(s) at collection time (no grad)
    seat:           int           # which player took this step
    legal_actions:  List[int]     # legal actions at this step

    # Filled in after the episode ends
    reward: float          = 0.0
    transformed_return: float = 0.0   # G̃_t (R-NaD transformed, per-seat)

    # Optional: warm-start aux target for the auxiliary prediction head
    aux_target: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Episode collection (Stage A: single-round episodes)
# ---------------------------------------------------------------------------

def collect_round(
    policy:     LiarsPokerNet,
    anchor:     LiarsPokerNet,
    hand_size:  int,
    num_players: int = 2,
    warm_start=None,
) -> List[Step]:
    """
    Play one round of Liar's Poker with both seats using `policy`.

    Returns the list of Steps for that round (all seats interleaved).
    Rewards are set to +1 for the winner and −1 for the loser.

    Parameters
    ----------
    policy      : current policy network  (used for action sampling)
    anchor      : regularisation anchor   (frozen copy, used for log_prob_reg)
    hand_size   : cards dealt to each active player (Stage A: fixed)
    num_players : number of seats (2 for Stage A)
    warm_start  : optional WarmStartLookup (for aux target extraction)
    """
    state = new_match(num_players)
    # Override hand sizes so the "match" starts at the desired stage.
    state.hand_sizes = [hand_size] * num_players
    state.start_next_round()

    steps: List[Step] = []

    while state.round_state is not None:
        cp    = state.round_state.current_player
        info  = state.info_state(cp)
        legal = state.legal_actions()

        with torch.no_grad():
            obs_p = policy.encode_obs(info)
            obs_r = anchor.encode_obs(info)

            logits_p, value_t = policy.forward(obs_p)
            logits_r, _       = anchor.forward(obs_r)

        # Mask illegal actions
        logits_p = _mask_logits(logits_p, legal)
        logits_r = _mask_logits(logits_r, legal)

        dist_p = Categorical(logits=logits_p)
        dist_r = Categorical(logits=logits_r)
        action = int(dist_p.sample().item())

        log_prob_old = float(dist_p.log_prob(torch.tensor(action)).item())
        log_prob_reg = float(dist_r.log_prob(torch.tensor(action)).item())

        # Aux target (if requested and in range)
        aux_tgt = None
        if warm_start is not None:
            active     = info["active"]
            hand_sizes = info["hand_sizes"]
            n = sum(hand_sizes[s] for s, a in enumerate(active) if a)
            _, _, cond_key = warm_start.get_features(info["own_hand"], n)
            if cond_key is not None:
                aux_tgt = warm_start.get_aux_target(cond_key, n)

        steps.append(Step(
            info          = info,
            action        = action,
            log_prob_old  = log_prob_old,
            log_prob_reg  = log_prob_reg,
            value_old     = float(value_t.item()),
            seat          = cp,
            legal_actions = legal,
            aux_target    = aux_tgt,
        ))

        result = state.apply_action(action)
        if result is not None:
            # Round over: assign per-seat terminal rewards (+1 / -1)
            for step in steps:
                if step.seat == result.winner_seat:
                    step.reward = 1.0
                else:
                    step.reward = -1.0
            break

    return steps


# ---------------------------------------------------------------------------
# R-NaD return computation
# ---------------------------------------------------------------------------

def compute_rnad_returns(
    steps:   List[Step],
    eta:     float,
    gamma:   float = 1.0,
) -> None:
    """
    Compute R-NaD transformed returns in place, **per seat**.

    For player i at their k-th decision step:
        G̃_k^i  =  r^i  −  η · Σ_{j=k}^{K_i}  (log π_j − log π_reg_j)

    where K_i is the last step taken by player i in this episode,
    and r^i ∈ {+1, −1} is the terminal reward for player i.

    With γ = 1 (finite horizon, no discounting):
        G̃_k  =  r  −  η · running_kl_from_k_to_end
    """
    seats = {s.seat for s in steps}
    for seat in seats:
        seat_steps = [s for s in steps if s.seat == seat]
        if not seat_steps:
            continue

        terminal_reward = seat_steps[-1].reward  # ±1

        # Backward pass: accumulate KL penalty from last step to first
        kl_suffix = 0.0
        for step in reversed(seat_steps):
            kl_suffix += step.log_prob_old - step.log_prob_reg
            step.transformed_return = terminal_reward - eta * kl_suffix


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(
    policy:   LiarsPokerNet,
    steps:    List[Step],
    config:   RNaDConfig,
    device:   torch.device,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the combined R-NaD loss for a batch of Steps.

    Returns
    -------
    total_loss : scalar Tensor (has gradient)
    metrics    : dict of float scalar metrics for logging
    """
    policy_losses  = []
    value_losses   = []
    entropy_terms  = []
    aux_losses     = []

    for step in steps:
        # Re-encode observation with current network weights (gradients flow)
        obs = policy.encode_obs(step.info)

        if config.use_aux_loss and step.aux_target is not None:
            logits, value, aux_logits = policy.forward_with_aux(obs)
        else:
            logits, value = policy.forward(obs)
            aux_logits = None

        # --- masked distribution ---
        masked_logits = _mask_logits(logits, step.legal_actions)
        dist          = Categorical(logits=masked_logits)

        log_prob      = dist.log_prob(torch.tensor(step.action, device=device))
        entropy       = dist.entropy()

        # --- advantage (stop-gradient on return and baseline) ---
        G_tilde  = torch.tensor(step.transformed_return, dtype=torch.float32, device=device)
        baseline = torch.tensor(step.value_old,          dtype=torch.float32, device=device)
        advantage = (G_tilde - baseline).detach()   # stop-gradient on advantage

        # --- losses ---
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.mse_loss(value.squeeze(), G_tilde))
        entropy_terms.append(-entropy)             # maximise entropy → minimise -H

        if aux_logits is not None:
            tgt = torch.from_numpy(step.aux_target).to(device)
            # CE between network's predicted pool distribution and the tabulated
            # conditional probabilities (soft targets via KL).
            aux_losses.append(F.kl_div(
                F.log_softmax(aux_logits, dim=-1),
                tgt,
                reduction="sum",
            ))

    n = len(steps)
    policy_loss  = torch.stack(policy_losses).mean()
    value_loss   = torch.stack(value_losses).mean()
    entropy_loss = torch.stack(entropy_terms).mean()
    aux_loss     = torch.stack(aux_losses).mean() if aux_losses else torch.tensor(0.0, device=device)

    total = (
        policy_loss
        + config.value_loss_coef  * value_loss
        + config.entropy_coef     * entropy_loss
        + config.aux_loss_coef    * aux_loss
    )

    metrics = {
        "loss/policy":  float(policy_loss.detach()),
        "loss/value":   float(value_loss.detach()),
        "loss/entropy": float((-entropy_loss).detach()),   # log as entropy (positive)
        "loss/aux":     float(aux_loss.detach()),
        "loss/total":   float(total.detach()),
    }
    return total, metrics


# ---------------------------------------------------------------------------
# Main trainer class
# ---------------------------------------------------------------------------

class RNaDTrainer:
    """
    Orchestrates R-NaD self-play training.

    Usage
    -----
    config  = RNaDConfig(stage="A", stage_a_hand_size=1, num_players=2)
    trainer = RNaDTrainer(config)
    trainer.train()
    """

    def __init__(self, config: RNaDConfig, device: Optional[str] = None) -> None:
        self.config = config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Policy network (trained)
        self.policy = LiarsPokerNet(config).to(self.device)
        # Anchor network (frozen copy, updated periodically)
        self.anchor = copy.deepcopy(self.policy)
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.anchor.eval()

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.lr,
        )

        # Warm-start lookup (for aux targets during collection)
        if config.use_aux_loss:
            from agent.rnad.warm_start import WarmStartLookup
            self._warm_start = WarmStartLookup()
        else:
            self._warm_start = None

        # Checkpointing
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Training state
        self.iteration   = 0
        self.total_steps = 0
        self._metrics_buf: List[dict] = []

        print(f"[RNaD] Network: {self.policy.num_parameters():,} parameters")
        print(f"[RNaD] Device: {self.device}")
        print(f"[RNaD] Stage: {config.stage}, hand_size={config.stage_a_hand_size}, "
              f"N={config.num_players}")

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        t0 = time.time()

        for it in range(self.config.num_iterations):
            self.iteration = it

            # 1. Collect episodes
            steps = self._collect_batch()

            # 2. Compute R-NaD transformed returns (in place)
            compute_rnad_returns(steps, self.config.eta, self.config.gamma)

            # 3. Compute loss + update
            self.policy.train()
            loss, metrics = compute_loss(
                self.policy, steps, self.config, self.device
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.grad_clip
            )
            self.optimizer.step()

            self._metrics_buf.append(metrics)
            self.total_steps += len(steps)

            # 4. Update anchor
            if it % self.config.anchor_update_freq == 0 and it > 0:
                self._update_anchor()

            # 5. Logging
            if it % self.config.log_freq == 0:
                self._log(it, time.time() - t0)

            # 6. Evaluation
            if it % self.config.eval_freq == 0 and it > 0:
                from agent.rnad.eval import evaluate_policy
                results = evaluate_policy(
                    self.policy,
                    num_episodes=self.config.eval_episodes,
                    hand_size=self.config.stage_a_hand_size,
                    num_players=self.config.num_players,
                    device=self.device,
                )
                self._log_eval(it, results)

            # 7. Checkpointing
            if it % self.config.checkpoint_freq == 0 and it > 0:
                self.save_checkpoint(f"iter_{it:06d}")

        self.save_checkpoint("final")
        print(f"[RNaD] Training complete — {self.iteration + 1} iterations, "
              f"{self.total_steps:,} steps, "
              f"{(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------------------------

    def _collect_batch(self) -> List[Step]:
        """Collect config.episodes_per_update round episodes."""
        self.policy.eval()
        all_steps: List[Step] = []

        for _ in range(self.config.episodes_per_update):
            steps = collect_round(
                policy      = self.policy,
                anchor      = self.anchor,
                hand_size   = self.config.stage_a_hand_size,
                num_players = self.config.num_players,
                warm_start  = self._warm_start,
            )
            all_steps.extend(steps)

        return all_steps

    def _update_anchor(self) -> None:
        """Copy current policy weights into the anchor."""
        self.anchor.load_state_dict(copy.deepcopy(self.policy.state_dict()))
        self.anchor.eval()
        for p in self.anchor.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, it: int, elapsed: float) -> None:
        if not self._metrics_buf:
            return
        avg = {k: float(np.mean([m[k] for m in self._metrics_buf]))
               for k in self._metrics_buf[0]}
        self._metrics_buf.clear()

        print(
            f"[{it:6d}] "
            f"loss={avg['loss/total']:+.4f}  "
            f"π={avg['loss/policy']:+.4f}  "
            f"V={avg['loss/value']:.4f}  "
            f"H={avg['loss/entropy']:.4f}  "
            f"aux={avg['loss/aux']:.4f}  "
            f"steps={self.total_steps:,}  "
            f"t={elapsed:.0f}s"
        )

    def _log_eval(self, it: int, results: dict) -> None:
        print(
            f"[{it:6d}] EVAL  "
            f"vs_random={results.get('win_rate_vs_random', 0):.3f}  "
            f"vs_blind={results.get('win_rate_vs_blind', 0):.3f}  "
            f"avg_entropy={results.get('avg_entropy', 0):.3f}"
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str) -> str:
        path = os.path.join(self.config.checkpoint_dir, f"rnad_{tag}.pt")
        torch.save({
            "iteration":      self.iteration,
            "total_steps":    self.total_steps,
            "policy_state":   self.policy.state_dict(),
            "anchor_state":   self.anchor.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config":         self.config,
        }, path)
        print(f"[RNaD] Checkpoint saved → {path}")
        return path

    @classmethod
    def load_checkpoint(
        cls,
        path:   str,
        device: Optional[str] = None,
    ) -> "RNaDTrainer":
        ckpt    = torch.load(path, map_location="cpu")
        config  = ckpt["config"]
        trainer = cls(config, device=device)
        trainer.policy.load_state_dict(ckpt["policy_state"])
        trainer.anchor.load_state_dict(ckpt["anchor_state"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer.iteration   = ckpt["iteration"]
        trainer.total_steps = ckpt["total_steps"]
        print(f"[RNaD] Loaded checkpoint from {path} (iter {trainer.iteration})")
        return trainer


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="R-NaD trainer for Liar's Poker")
    parser.add_argument("--hand-size",   type=int,   default=1)
    parser.add_argument("--num-players", type=int,   default=2)
    parser.add_argument("--eta",         type=float, default=0.2)
    parser.add_argument("--iterations",  type=int,   default=20_000)
    parser.add_argument("--batch-size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--resume",      type=str,   default=None)
    args = parser.parse_args()

    if args.resume:
        trainer = RNaDTrainer.load_checkpoint(args.resume)
    else:
        cfg = RNaDConfig(
            stage                = "A",
            stage_a_hand_size    = args.hand_size,
            num_players          = args.num_players,
            eta                  = args.eta,
            num_iterations       = args.iterations,
            episodes_per_update  = args.batch_size,
            lr                   = args.lr,
            use_warm_start       = not args.no_warm_start,
        )
        trainer = RNaDTrainer(cfg)

    trainer.train()
