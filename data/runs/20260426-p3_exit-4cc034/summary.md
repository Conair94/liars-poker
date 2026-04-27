# Reflection Summary — `20260426-p3_exit-4cc034`

- Records: **467,153**
- Buckets (agent × opponent × ruleset): **90**
- Load time: 20.58s

## Implemented Rules

### Infeasible-bid tripwire
_Should be 0 post-2026-04-24 feasibility-filter fix._

_No flags._

### Stale-bid repetition (opening bids)
_>70% of openings reuse the same bid (≥10 openings sampled)._

_No flags._


## Deferred Rules (need per-choice p/eu — arrives in P5)

| rule | status |
|---|---|
| Missed-call (P(call)<0.3 when posterior says no bid) | deferred: p+posterior |
| Low-EU choice (chosen.eu below 3rd-best by margin) | deferred: needs eu |
| Rank-leak entropy (bid dist. conditional on hand[0]) | deferred: needs p |
