# Liar's Poker Agent (Stage 2)

Reinforcement-learning agent for card-based Liar's Poker. See
[`../AGENT_DESIGN.md`](../AGENT_DESIGN.md) for the full design document and
[`AGENT_CATALOG.md`](AGENT_CATALOG.md) for agent descriptions.

## Layout

- `game/` — game engine (`engine.py`, `bids.py`) + tests
- `baseline/` — CFR Nash solver and blind-equilibrium baselines (M2)
  - `cfr_1v1.py` — reference CFR+ solver (slow, correctness baseline)
  - `cfr_1v1_fast.py` — vectorized CFR+ solver (75× speedup, production use)
  - `cfr_1v1_overnight.py` — overnight training script (`--solver fast/slow`)
- `rnad/` — R-NaD trainer, network, evaluation (M3–M4)
  - `trainer.py` — RNaDTrainer with MPS auto-detect, batched loss, exact_rules
  - `config.py` — RNaDConfig (includes `exact_rules`, `high_hand` flags)
  - `warm_start.py` — WarmStartLookup (LRU-cached features, exact conditional tables)
- `search/` — test-time compute / search wrappers (M5, planned)
- `web/` — FastAPI backend + HTMX frontend for human play (M6)
- `data/` — MC caches and warm-start tables
- `checkpoints/` — trained model checkpoints (gitignored except `.gitkeep`)

## Data Files

| File | Description | Script |
|------|-------------|--------|
| `hand_rank_probs_matrix.json` | Marginal at-least probabilities (52-card, 3M samples) | auto-built by `warm_start.py` |
| `extended_conditional_probs_ranked.json` | Standard conditional at-least tables (74 conditions, 1M/cond) | `compute_extended_conditional_probs.py` |
| `extended_conditional_exact_probs.json` | Exact-rules conditional tables (70 conditions, 10k/cond) | `compute_extended_conditional_exact_probs.py` |
| `exact_rules_probs.json` | Exact-rules marginal probabilities (10k samples/n) | `compute_exact_rules_probs.py` |
| `blind_equilibrium.json` | Blind Nash equilibrium for n=2..10 | `agent/baseline/blind_equilibrium.py` |
| `five_kings_probs.json` | 53-card marginal probabilities for Five-Kings mode | `compute_five_kings_probs.py` |
| `benchmark_results.json` | Head-to-head win rates across all agent pairs | `agent/benchmark.py` |
| `cfr_1v1_run/` | CFR+ checkpoint directories | `cfr_1v1_overnight.py` |

## Running Scripts

All scripts run from `Liars poker/` (not the repo root):

```bash
cd "Liars poker/"

# Standard conditional MC (at-least rules, 74 conditions, 1M/cond — ~2.5h, 8 cores)
python -m agent.data.compute_extended_conditional_probs

# Exact-rules conditional MC (70 conditions, 10k/cond — ~4-6h, 9 workers)
python -m agent.data.compute_extended_conditional_exact_probs

# CFR+ Nash solver — production run (max_bids=4, 50k iters, ~5.5h)
python -m agent.baseline.cfr_1v1_overnight \
    --name cfr_plus_mb4_hh \
    --max-bids 4 --batch 100 --total-iters 50000 \
    --solver fast

# R-NaD Stage A (hand-size=1, 5k iters, ~30 min on MPS)
python -m agent.rnad.trainer --stage A --hand-size 1 --iterations 5000 --batch-size 128

# Benchmark (all agent pairs, 100 games each)
python -m agent.benchmark --games 100 --seed 42

# Web UI (http://localhost:8000)
python -m agent.web.backend.app
```

## Milestone Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| M0 | Literature survey | Complete |
| M1 | Game engine | Complete |
| M2 | Blind baseline equilibrium | Complete |
| M2b | Warm-start lookup (marginal + conditional) | Complete |
| M2c | CFR+ 1v1 Nash solver (vectorized) | Infrastructure complete; production 50k-iter run pending |
| M3 | R-NaD trainer (MPS, batched loss, exact_rules) | Infrastructure complete |
| M5 | ExactRulesConditionalAgent (5-fix version) | Complete (2026-04-22) |
| M6a | Web UI with High Hand button | Complete (2026-04-22) |
| M4 | Full-game multi-round agent | Not started |
| M6b | Full web UI (stats, history, trained agent) | Not started |
