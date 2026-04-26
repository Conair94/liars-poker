# src/

Python source packages for the Liar's Poker research project.

```
src/
├── game/          — pure-Python game engine (no ML dependencies)
├── agents/        — agent implementations + registry
│   ├── heuristic/ — closed-form heuristic agents (CFR, blind equilibrium)
│   ├── learned/   — trained neural-network agents
│   │   └── rnad/  — R-NaD self-play trainer + eval
│   └── search/    — placeholder for test-time compute agents (M5)
└── training/      — training utilities and probability scripts
    └── probs/     — probability table generators; output → data/probs/
```

---

## game/

`bids.py` — bid space (110 bids: HC 2 through SF A, plus CALL and HH actions).  
`engine.py` — deterministic match engine; `new_match()` → `MatchState`, `step()`.

Both modules are dependency-free (numpy optional); they are imported by agents,
training scripts, and the web backend.

---

## agents/

`registry.py` — `AGENT_REGISTRY` dict mapping display name → `AgentConfig`.
All agents must be registered here; never use if/else chains keyed on name.

**heuristic/**

| Module | Agent | Description |
|--------|-------|-------------|
| `blind_equilibrium.py` | `BlindBaselineAgent` | N=2 backward-induction Nash; ignores private cards |
| `cfr_1v1.py` | `CFR1v1Agent` | Reference CFR solver (exact game tree, slow) |
| `cfr_1v1_fast.py` | `CFR1v1FastAgent` | Vectorised CFR (10–100× faster, same solution) |
| `cfr_1v1_overnight.py` | — | Training script for overnight CFR runs |

**learned/rnad/**

| Module | Purpose |
|--------|---------|
| `config.py` | `RNaDConfig` dataclass — all hyperparameters |
| `network.py` | `LiarsPokerNet` — DeepSet card encoder + 4-layer MLP (418K params) |
| `trainer.py` | R-NaD self-play training loop (Stage A fixed-size + Stage B full-match) |
| `eval.py` | Evaluation vs random / blind agents; win-rate and bid-accuracy tracking |
| `warm_start.py` | `WarmStartLookup` — probability feature vectors for the network |

`cfr_nash.py` (in `agents/`) — `CFRNashAgent`; loads a trained CFR checkpoint
from `data/runs/cfr_1v1/` and samples from the average strategy.

---

## training/

`benchmark.py` — head-to-head benchmark runner; results written to
`data/runs/benchmark/`.

**probs/** — probability table generators. Run order:

```bash
python -m training.probs.generate_prob_tables          # marginal → data/probs/
python -m training.probs.compute_conditional_probs     # conditional → data/probs/
python -m training.probs.compute_exact_rules_probs     # exact-rules → data/probs/
```

All scripts cache results to `data/probs/*.json` and skip recomputation on
subsequent runs unless the cache is absent.

---

## Running the benchmark

```bash
# From repo root
python -m training.benchmark
```

See `docs-internal/design/` for architectural decisions and milestone tracking.
