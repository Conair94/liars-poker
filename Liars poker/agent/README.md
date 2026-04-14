# Liar's Poker Agent (Stage 2)

Reinforcement-learning agent for card-based Liar's Poker. See
[`../AGENT_DESIGN.md`](../AGENT_DESIGN.md) for the full design document.

## Layout

- `game/` — OpenSpiel game implementation (`liars_poker_cards.py`) + tests
- `baseline/` — blind-variant equilibrium baselines (M2)
- `rnad/` — R-NaD trainer, network, evaluation (M3–M4)
- `search/` — test-time compute / search wrappers (M5)
- `web/` — FastAPI backend + frontend for human play (M6)
- `data/` — MC caches and warm-start tables (extended conditional probabilities, etc.)
- `checkpoints/` — trained model checkpoints (gitignored except `.gitkeep`)

## Running scripts

All Stage 2 scripts are run from the project root (`papers/Liars poker/`):

```bash
cd "papers/Liars poker/"
python -m agent.data.compute_extended_conditional_probs  # overnight MC
```
