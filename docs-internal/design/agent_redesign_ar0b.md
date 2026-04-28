# AR-0b Sub-design — Sweep driver, bench sweep, comparator

- **Status:** Draft
- **Date:** 2026-04-28
- **Owner:** main
- **Parent plan:** [agent_redesign_plan.md](agent_redesign_plan.md)
- **Parent design:** [agent_redesign.md](agent_redesign.md) §9
- **Predecessor:** AR-0a (contracts + checkpoints landed earlier this
  session)

This sub-design fixes the parallel-research pipeline: sweep config
schema, run-id format, atomic-manifest semantics, parquet column set,
and resume rules. AR-0b's deliverable is a working harness over the
existing heuristic ladder (no learning yet); AR-1+ will plug into it
unchanged.

## Goals

- A single sweep driver that takes one YAML and runs N independent
  sub-runs with crash isolation, atomic resume, and configurable
  parallelism.
- A post-sweep `bench_sweep` step that pairwise-benchmarks every
  finished run from the sweep against a configurable opponent set and
  writes a long-format Parquet table.
- A `compare` reporter that, given one or more sweep ids, produces a
  markdown summary table. Plotting (Pareto / calibration) is
  out-of-scope for AR-0b — it gets added in AR-1 once HandModel is
  the first artifact worth plotting.
- A smoketest config that exercises the harness on the existing
  heuristic ladder and reproduces the 2026-04-24 win-rate numbers
  within MC noise — proves the pipeline before any new agent exists.

## Non-goals

- Distributed execution, GPU scheduling, or queueing. Local subprocess
  pool only. Per-process device is configurable (`--device`); the
  driver does not place GPUs.
- Hyperparameter optimization (Bayesian, evolutionary, etc.). The
  sweep is a config product; smarter search is future work.
- Plot generation. Markdown table only in this phase.

## 1. Run identity and directory layout

Every sub-run owns a directory under `data/runs/<run_id>/`:

```
<run_id>/
├── config.yaml         # resolved (post-override) config
├── manifest.json       # written atomically; tracks last_completed_iter
└── (component-specific outputs land here)
```

`run_id` format: `<phase>-<UTC_TIMESTAMP>-<slug>-<git8>`, e.g.
`ar1-20260428T123456Z-handmodel-d128-89e3443`. The slug is the
short-name of the sweep cell (deterministic from the config dict).
Collisions: a duplicate run_id is an error unless `--resume` is given.

A *sweep* lives at `data/sweeps/<sweep_id>/`:

```
<sweep_id>/
├── sweep.yaml          # the input config, copied verbatim
├── index.json          # sweep_cell → run_id map; one entry per cell
├── runs/               # symlinks back to data/runs/<run_id>/ for tooling
└── bench/              # populated by bench_sweep
    └── results.parquet
```

`sweep_id` format: `<phase>-<UTC_TIMESTAMP>-<sweep_name>-<git8>`. Same
collision rule as `run_id`.

## 2. Sweep config schema

YAML with the following top-level keys:

```yaml
sweep_name: heuristic_ladder_smoketest
phase:      ar0b
production: false              # if true, refuses to start with dirty git tree
random_seed: 0                 # base seed; per-cell seed = base + cell_index
device: cpu

# How to enumerate cells. Two forms:
#   axis-product (default):
axes:
  - key: handmodel
    values: [analytic, learned-foo, learned-bar]
  - key: callpolicy
    values: [analytic, distilled-X]
#   OR an explicit list (mutually exclusive with `axes`):
# cells:
#   - {handmodel: analytic, callpolicy: analytic}
#   - {handmodel: learned-foo, callpolicy: distilled-X}

# What command runs for each cell. Templated with cell values.
runner:
  command: python -m training.bench_only
  args:
    handmodel:  "{handmodel}"
    callpolicy: "{callpolicy}"
    bidpolicy:  analytic
  fixed:
    games_per_pair: 200
```

Two required CLI flags on `python -m training.sweep`:

| Flag | Purpose |
| --- | --- |
| `--max-parallel N` | Subprocess pool size; default 1 |
| `--resume` | Re-invoke completed cells as no-ops; resume failed/incomplete from last manifest write |

## 3. Atomic manifest semantics

`manifest.json` is the source-of-truth for "is this run done?":

```json
{
  "run_id":             "ar0b-20260428T...-blind-vs-cond-89e3443",
  "sweep_id":           "ar0b-20260428T...-heuristic_smoketest-89e3443",
  "config_hash":        "sha256(config.yaml)",
  "git_sha":            "89e3443",
  "started_at":         "2026-04-28T12:34:56Z",
  "last_completed_iter": 0,
  "completed":          true,
  "exit_code":          0,
  "components": {
    "handmodel":  null,
    "callpolicy": null,
    "bidpolicy":  null
  }
}
```

Write rule: **always** write `manifest.json.tmp` and `os.replace` to
the canonical name. Never write in place.

A run is "completed" iff `manifest.json` exists and `completed: true`.
On `--resume`:

- For every cell in the sweep config:
  - If `data/runs/<run_id>/manifest.json` shows `completed: true`,
    skip.
  - Else delete the partial output dir and re-run from scratch.
    (AR-0b does not restart mid-iter — that's an AR-3 concern when
    runs are 30+ minutes; smoketest runs are seconds.)

`bench_sweep` writes its own manifest under
`<sweep_id>/bench/manifest.json` with the same shape.

## 4. The runner contract

Each cell runs `<runner.command> --config <resolved_yaml> --run-id <run_id>`.
The runner's only requirement: write a final `manifest.json` with
`completed: true` to `data/runs/<run_id>/`.

For AR-0b's smoketest the runner is a new `training.bench_only` entry
point that:

1. Reads its config (handmodel/callpolicy/bidpolicy keys = registry
   agent keys for now; in AR-1+ these will become composite component
   refs).
2. Builds a 2-player agent matchup (one cell = one matchup) and plays
   `games_per_pair` games via the existing
   [src/training/benchmark.py](../../src/training/benchmark.py)
   `run_match` function.
3. Records win-rate + Wilson 95% CI to `data/runs/<run_id>/results.json`.
4. Writes `manifest.json` with `completed: true`.

This isn't the *full* agent runner — it's the smoketest's runner.
AR-1's HandModel-pretrain runner will be a different command, but
honor the same contract (final `manifest.json` with completion flag).

## 5. bench_sweep schema

Output: `data/sweeps/<sweep_id>/bench/results.parquet`. Long format:

| column | dtype | meaning |
| --- | --- | --- |
| run_id | string | which run produced this row |
| cell | json string | sweep cell (the axis values for this run) |
| opponent | string | name of opponent (registry key or another run_id) |
| metric | string | one of `winrate`, `lbr`, `subgame` |
| value | float64 | metric value |
| ci_low, ci_high | float64 | Wilson 95% CI for `winrate`; NaN otherwise |
| n_games | int64 | sample size for `winrate`; -1 otherwise |
| games_per_pair | int64 | from sweep config |
| computed_at | string | ISO UTC timestamp |

Parquet via pandas (already a project dep). pyarrow is optional —
if unavailable, fall back to CSV with the same columns. We do **not**
add pyarrow as a hard dep for AR-0b; the smoketest works on either.

## 6. Comparator (markdown only)

`python -m training.compare <sweep_id> [<sweep_id>...]` reads each
sweep's `bench/results.parquet` and emits
`data/sweeps/<sweep_id>/report/summary.md`:

```markdown
# Sweep <sweep_id>

## Win-rate matrix (200 games per pair)

| run_id | opponent | win_rate | 95% CI |
| --- | --- | --- | --- |
| ... | ... | 0.85 | [0.79, 0.90] |

## Best by metric

- `winrate` vs heuristic ladder: <run_id> at 0.87
- `lbr` (lower is better): N/A (no LBR run yet)
```

That is the full v1 of `compare`. Plotting / Pareto is a separate
delivery in AR-1 once we have multi-axis Pareto candidates.

## 7. Reproducibility floor (parent §9.6)

The sweep driver enforces:

- `random_seed` is set in the config (rejected with a clear error if
  missing).
- Every sub-run gets `seed = base_seed + cell_index`. Deterministic
  per cell.
- `git_sha` is captured into both the sweep manifest and every run's
  `config.yaml`.
- If `production: true` in sweep config, refuses to start unless `git
  status --porcelain` is empty. (Default is `false` so smoketests
  don't need a clean tree.)
- Output dirs are content-addressable: a duplicate `run_id` errors
  unless `--resume` is given.

## 8. Smoketest

`configs/sweeps/heuristic_ladder_smoketest.yaml` declares 4 cells —
all-pairs of `{exactconditional, exact_mixed, exact_opp_model,
exact_adaptive}` taken two at a time, deduplicated. 50 games per pair
(small enough to run in <1 minute, large enough that win rate is in
the right ballpark for `exact_opp_model` vs. `exact_adaptive`).

Acceptance: smoketest sweep produces a `results.parquet` whose
`exactconditional` row has win-rate within ±0.10 of 0.50 against the
ladder peer agents (we do not require matching the historical 85.8%
because that was vs. *all* lower-rung agents, not vs. peers; the
smoketest exists to prove the harness, not to benchmark the agents).

## 9. Tests

- `tests/training/test_sweep.py`:
  - Cell enumeration from axis-product produces the right tuples.
  - `--resume` is a no-op on a fully-completed sweep.
  - Mid-run kill (simulated by writing a half-baked manifest, then
    re-invoking) re-runs that one cell.
  - Refuses to start with no `random_seed`.
- `tests/training/test_bench_sweep.py`:
  - On a fixed-seed pair of `RandomAgent` vs. itself, produces a
    `results.parquet` with one row per `(run_id, opponent, metric)`
    triple, win-rate near 0.5 ±0.15 at 100 games.
- `tests/training/test_compare.py`:
  - Markdown output contains expected headings and the right number
    of rows.

All AR-0b tests are <5 seconds.

## 10. Out-of-scope clarifications

- **No LBR/subgame in the smoketest's bench_sweep.** Those metrics
  are computationally non-trivial; AR-0b only exercises the win-rate
  path. AR-1's bench_sweep configs add the metrics columns once an
  agent exists where the metric is informative.
- **No GPU scheduling.** `--device` is plumbed through to the runner
  and that's it.
- **No live progress UI.** Sweep driver prints one line per cell
  start/finish to stderr; full progress lives in each cell's
  manifest.
