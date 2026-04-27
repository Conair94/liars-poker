# Reflection Summary — `20260427-p5_3_smoke-d9033c`

- Records: **510**
- Buckets (agent × opponent × ruleset): **12**
- Load time: 0.02s

## Implemented Rules

### Infeasible-bid tripwire
_Should be 0 post-2026-04-24 feasibility-filter fix._

_No flags._

### Stale-bid repetition (opening bids)
_>70% of openings reuse the same bid (≥10 openings sampled)._

_No flags._

### Missed-call (P5-#3)
_P(call)<0.30 when standing bid is Pair-or-stronger._

_No flags._

### Rank-leak (P5-#3)
_Opening-bid concentration > 60% in one hand-rank bucket while < 10% in others (≥30 openings sampled)._

_No flags._


## Deferred Rules (need EU per choice — needs value model)

| rule | status |
|---|---|
| Low-EU choice (chosen.eu below kth-best by margin) | deferred: needs eu |
