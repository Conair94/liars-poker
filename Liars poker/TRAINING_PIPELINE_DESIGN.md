# Training Pipeline Refactor — Design Document

**Status:** Draft for review (rev. 2)
**Date:** 2026-04-24
**Author:** Connor (with Claude)
**Supersedes:** portions of `AGENT_DESIGN.md`, `TRAINING_OPTIMIZATION_PLAN.md`, `IMPLEMENTATION_PLAN.md` once merged.
**Overarching research goal:** state-of-the-art reinforcement learning and AI-in-games research, with Liar's Poker as the current testbed.

## Changelog

- **rev. 2 (2026-04-24):** Decision made to retire the public web demo and focus on local development. §8 rewritten as "Frontend retirement." Q1 in §13 resolved and removed. New §14 "Brainstorming — Paths Forward" added with research-oriented suggestions.

This document is the first of three required artifacts before any refactoring begins (design doc → plan/checklist → testing checklist → implementation, per `New-features.md` §2). It defines *what* the refactored pipeline must do and the I/O contract between its pieces. It does **not** prescribe code; that is the next doc.

---

## 1. Motivation

The project has outgrown its original structure. Symptoms, in order of pain:

1. **Double-edit tax (being resolved by retiring the web demo).** Every agent change currently must be made twice — once in `Liars poker/agent/web/backend/agents.py`, once in `docs/index.html`. The chosen fix (see §8) is to retire the public web demo entirely until agents are strong enough to warrant re-hosting.
2. **Markdown sprawl.** Root and `Liars poker/` contain many long Markdown files (`AGENT_DESIGN.md`, `AGENT_CATALOG.md`, `LITERATURE_SURVEY.md`, `IMPLEMENTATION_PLAN.md`, `TRAINING_OPTIMIZATION_PLAN.md`, several READMEs). Every new session burns context re-reading overlapping docs.
3. **Benchmarks report win-rates only.** We can tell an agent *lost* but not *where* it made a bad decision. Debugging requires re-running with manual instrumentation.
4. **Nash / CFR agents are weak** at current hand sizes and will be obsolete at larger sizes without a ground-up redesign. The memory index notes `cfr_plus` sitting at ~49% win-rate.
5. **No separation between agent code and training infrastructure.** Config, solver, data, checkpoints, and agent logic live entangled under `Liars poker/agent/`, making modular development hard.

## 2. Goals & Non-Goals

### Goals
- **G1.** Single source of truth for agent behavior — one Python code path drives all training, benchmarking, and local interactive play. (The web demo is retired per §8; a future re-hosting effort will consume this same code path.)
- **G2.** Decision-level logging: every action records state, options considered, probabilities, choice, and outcome.
- **G3.** Automated reflection: a tool that scans decision logs and flags likely mistakes without human review.
- **G4.** File layout where a fresh session can orient in ≤200 lines of top-level docs.
- **G5.** Modular agent architecture — hand model, bid policy, and call policy as independent, swappable components.
- **G6.** Reproducible runs: `config + seed → identical output`.
- **G7.** A foundation that supports a state-of-the-art CFR/Nash/RL redesign in later sessions.

### Non-Goals (for this refactor)
- Not rewriting the paper's LaTeX or results.
- Not retraining existing checkpoints — they should migrate forward via a loader shim.
- Not building the new Nash/CFR agent itself — only the infrastructure that will later host it.
- Not changing the game rules or hand evaluator.

## 3. Requirements

### 3.1 Functional
- **F1.** A single CLI entry point (e.g. `python -m training run <config>`) that trains, checkpoints, benchmarks, and emits a summary.
- **F2.** Every game played during benchmarking writes a `decisions.jsonl` record per turn (schema in §6).
- **F3.** A `reflect` subcommand reads `decisions.jsonl` and produces a flaw report — a ranked list of suspicious decisions per agent.
- **F4.** Agents are defined by a composition of three interfaces (§7). The ladder heuristics, CFR, and R-NaD agents must all fit this.
- **F5.** The web demo consumes the same agent definitions as the benchmark (approach chosen in §8).
- **F6.** A `run_id` namespaces every output under `data/runs/<run_id>/` — no more files in ad-hoc locations.
- **F7.** Existing checkpoints under `Liars poker/agent/checkpoints/` load without retraining.

### 3.2 Non-Functional
- **N1.** Top-level orientation: repo root has ≤3 Markdown files; `src/` has a single `README.md` that lists subpackages in ≤100 lines.
- **N2.** Decision log format is append-only JSONL, greppable, loadable into pandas in one line.
- **N3.** Reflection is fast enough to run inline after a 500-game benchmark (< ~30 s).
- **N4.** No net regression on existing benchmark win-rates after the refactor (§12 acceptance).
- **N5.** All new modules typed (`from __future__ import annotations`, `py.typed` marker) — preferred for catching the kind of silent-default bugs listed in memory (e.g. `CFRSolver param name`, `eval.py Ruleset Params`).

## 4. Proposed Repository Layout

```
liars-poker/
├── README.md                    # repo entry — links to docs/design/
├── pyproject.toml               # single Python project root (replaces scattered __init__ only)
├── src/
│   ├── game/                    # rules, state, hand evaluator — no agent imports
│   ├── agents/
│   │   ├── core/                # Agent, HandModel, BidPolicy, CallPolicy, registry
│   │   ├── heuristic/           # blind, conditional, exact-rules ladder
│   │   ├── search/              # CFR, CFR+, future DeepStack
│   │   └── learned/             # R-NaD, future Deep CFR / NFSP
│   ├── training/
│   │   ├── configs/             # one file per canonical run (YAML or TOML — Q5)
│   │   ├── runners/             # solvers and trainers
│   │   ├── benchmark.py         # tournament runner, writes decisions.jsonl
│   │   ├── reflect.py           # scans decisions.jsonl → flaw report
│   │   └── probs/               # probability-table generators
│   └── web/                     # whichever frontend strategy wins (§8)
├── data/
│   ├── probs/                   # precomputed probability tables (moved from agent/data)
│   ├── checkpoints/             # model weights
│   └── runs/<run_id>/
│       ├── config.yaml
│       ├── metrics.json         # summary stats
│       ├── decisions.jsonl      # per-turn records
│       └── summary.md           # human-readable report + reflection highlights
├── paper/                       # LaTeX sources (moved from "Liars poker/")
├── docs/                        # GH Pages site (unchanged until §8 resolved)
└── docs-internal/
    └── design/                  # this doc, literature survey, ADRs
```

**Why this shape.** Flattens the `Liars poker/agent/web/backend/...` nesting. Paper and code cleanly separated. Every run's artifacts live in *one* folder, not sprawled across `agent/data/` and `agent/checkpoints/`.

## 5. Configuration & Run IDs

- **Config format:** one file per canonical experiment under `src/training/configs/`. Candidates: YAML, TOML, or Python dataclasses. (Open question Q5.)
- **Run ID:** `YYYYMMDD-<agent_or_experiment>-<short_hash>` where the hash is of the resolved config. Deterministic: same config → same ID → idempotent overwrite.
- **Seeds:** every config must specify a seed; benchmark fans it out deterministically across games.

## 6. Decision Log Schema

One JSON object per line, per turn, per agent action.

```json
{
  "run_id": "20260501-exact_cond-ab12cd",
  "game_id": 42,
  "turn": 7,
  "agent": "ExactRulesConditional",
  "agent_seat": 0,
  "opponent": "CFRPlusMB4",
  "ruleset": {"exact_rules": true, "high_hand": true},
  "state": {
    "hand": ["AS","KS","QS"],
    "cards_on_table": 6,
    "standing_bid": "three_of_a_kind:J",
    "bid_history": ["pair:7","two_pair:9_5","three_of_a_kind:J"]
  },
  "choices": [
    {"action": "call",            "p": 0.31, "eu": 0.12},
    {"action": "bid:flush:S",     "p": 0.22, "eu": 0.08},
    {"action": "bid:3oK:Q",       "p": 0.18, "eu": 0.05},
    {"action": "bid:straight:9",  "p": 0.00, "eu": null, "feasible": false}
  ],
  "chosen": "bid:flush:S",
  "reasoning_tag": "softmax_sample",
  "outcome": {"challenged": true, "bid_existed": false, "point_delta": -1}
}
```

- `choices` is the *full* action space the agent considered, with its own probability assignment and (optionally) expected utility.
- `reasoning_tag` is a short enum-like string the agent itself writes (e.g. `min_viable`, `max_prob`, `softmax_sample`, `forced_escalation`) so reflect.py can cluster by decision mode without heuristic inference.
- Infeasible actions appear with `feasible: false` so reflection can confirm the filter is actually running.

## 7. Modular Agent Interfaces

```python
class HandModel(Protocol):
    """Posterior belief over opponent hands / remaining deck given history."""
    def update(self, observation: Observation) -> None: ...
    def posterior(self) -> HandDistribution: ...

class BidPolicy(Protocol):
    def propose(self, state: GameState, hand_model: HandModel) -> list[ScoredAction]: ...

class CallPolicy(Protocol):
    def decide(self, state: GameState, hand_model: HandModel) -> ScoredAction: ...

class Agent:
    hand_model: HandModel
    bid_policy: BidPolicy
    call_policy: CallPolicy
    def act(self, state) -> Decision: ...  # also emits the decision-log record
```

This lets us compose, e.g., *(exact-rules hand model) + (CFR bid policy) + (learned call policy)* and benchmark it against the sum of its parts. Each interface has its own test suite and can be trained/evaluated in isolation.

Migration path for existing agents:
- **Heuristic ladder** → shared `ExactRulesHandModel` + per-agent `BidPolicy`/`CallPolicy`.
- **CFR / CFR+** → policy is already a table; wrap it as `BidPolicy` + `CallPolicy` sharing a trivial hand model.
- **R-NaD** → network exposes `BidPolicy` + `CallPolicy`; hand model is implicit in the network's hidden state but still exposed via `posterior()` for logging.

## 8. Frontend Retirement (decided)

**Decision (2026-04-24):** The public web demo is retired. All hosting work stops. Agents will be developed and evaluated locally until they are strong enough to be worth re-deploying.

Rationale:

- The demo's value is demonstrating *finished* agents. Current agents are research artifacts, not finished work — exposing them to the public optimizes for the wrong audience.
- The double-edit tax (Python ↔ JS) has directly caused bugs and consumed session budget that should go toward research.
- Every frontend consolidation option considered (Pyodide, JSON-export, transpile, serverless, hybrid) adds infra work and ongoing maintenance that buys nothing for the research goal.
- A later re-hosting effort will be cheaper than maintaining it now, because the refactored pipeline (§4) will give us a single Python code path to target.

### 8.1 Retirement Checklist (do not execute yet — belongs in the next plan doc)

1. **Undeploy GitHub Pages.** In the repo's GitHub Settings → Pages, set source to "None" (or delete the Pages deployment). This takes the live site offline.
2. **Archive the frontend code.** Move `docs/` and `Liars poker/agent/web/frontend/` to `archive/web-2026-04/` (keep in-tree for reference; do not delete history).
3. **Retire the backend.** Move `Liars poker/agent/web/backend/` to the same archive location. The `agents.py` JS-mirror file stops receiving updates.
4. **README update.** Remove the "Play online" link from `README.md` and add a note that the project is in local-dev-only mode.
5. **CI/action cleanup.** Remove any GitHub Actions workflow that builds or deploys Pages.
6. **Record the decision.** Add an ADR-style note under `docs-internal/design/` so a future session understands *why* the web demo was removed.

### 8.2 Future Re-Hosting

When agents are competitive enough to merit public play, revisit §8 of rev. 1 of this doc. The refactored single-source Python pipeline will make **Option B** (JSON-exported policies) or **Option A** (Pyodide) cheap to adopt. Do not pre-build for that future now.

## 9. Automated Reflection (v1)

`reflect.py` ingests `decisions.jsonl` and emits `summary.md` with a flaw report. Initial rule set:

| Rule | Flag when… |
|---|---|
| Infeasible bid leak | `chosen` has `feasible: false` in `choices` *(should be impossible post-filter; tripwire)* |
| Missed-call | `p(call) < 0.3` but posterior says `P(standing_bid_exists) < 0.1` |
| Low-EU choice | `chosen.eu` is below the 3rd-best action's EU by > margin |
| Stale bid repetition | agent proposes a ladder-equal bid repeatedly across similar states |
| Rank-leak | bid distribution conditional on `hand[0]` rank has entropy below threshold (see memory: Mixed Strategy Insight) |

Output is aggregated by (agent, opponent, ruleset) and sorted by flag rate. Per-flag sample decisions are linked back to `decisions.jsonl` line numbers.

## 10. Nash / CFR Redesign — Scope Fence

The new training pipeline is a *prerequisite* for the Nash/CFR redesign, not the redesign itself. What this refactor commits to providing:

- A `search/` package with a clean `BidPolicy`/`CallPolicy` surface.
- Probability-table plumbing that scales to larger hand sizes without assuming the current 3-card layout.
- A decision-logging surface that solver-based agents can emit through.

Actual algorithm choice (Deep CFR, DeepStack continual resolving, NFSP, updated R-NaD) belongs in a *separate* design doc after a literature-survey refresh (New-features.md §3.5). Question Q3 in §13 asks for the target hand sizes that redesign must handle, since that determines which algorithms are viable.

## 11. Rollout Plan (spans multiple sessions)

1. **This doc** — reviewed and open questions answered.
2. **Plan/checklist doc** (next session) — concrete file moves, interface stubs, migration order.
3. **Testing checklist doc** — golden-path and edge-case tests for each new module.
4. **Implementation, phased:**
   - P1: `src/` skeleton + game package move (no behavior change).
   - P2: **Frontend retirement** (§8.1) — undeploy GH Pages, archive `docs/` and `Liars poker/agent/web/`, remove deploy workflows, update README. Do this early so subsequent agent edits never touch JS again.
   - P3: decision logging + reflect.py v1 (wrap existing agents; no refactor).
   - P4: modular agent refactor — heuristics first, then CFR, then R-NaD.
   - P5: Nash/CFR redesign (separate design doc).
5. **Markdown consolidation** — fold `AGENT_DESIGN.md`, `AGENT_CATALOG.md`, `TRAINING_OPTIMIZATION_PLAN.md`, `IMPLEMENTATION_PLAN.md` into `docs-internal/design/` with a single index.

## 12. Acceptance Criteria

The refactor is done when:
- **A1.** All existing agents pass a benchmark producing identical win-rates (±2pp at 500 games) to the pre-refactor numbers in memory.
- **A2.** A new agent can be added by writing one file under `src/agents/` and one line in the registry — no JS edit needed (assuming chosen frontend option).
- **A3.** `python -m training reflect <run_id>` produces a flaw report for every legacy agent in under 30 s on a 500-game run.
- **A4.** Repo root has ≤3 `.md` files; `src/` has a single `README.md`.
- **A5.** Every run output lives under `data/runs/<run_id>/` and nowhere else.
- **A6.** A fresh Claude Code session can reach productive work from `README.md` without loading more than 200 lines of docs.

## 13. Open Questions (please answer before next session)

1. **Project goal balance.** Is the priority (a) shipping strong agents for the paper, (b) building reusable research infra, or (c) both, with infra as means to end? This shifts how aggressively to invest in modular interfaces vs. just making current agents stronger.
2. **Target hand sizes** for the forthcoming Nash/CFR redesign — 5, 7, 9, arbitrary? Drives algorithm choice.
3. **Heuristic ladder future.** Freeze at current behavior (maintenance-free going forward), or keep iterating?
4. **Config format.** YAML, TOML, or Python dataclasses? (Dataclasses are most native; YAML is most human-friendly; TOML is a middle ground.)
5. **Paper location.** Keep LaTeX under `paper/` in this repo, or split into a separate repo?
6. **Reflection cadence.** Run automatically after every benchmark, or as an on-demand `reflect` subcommand?
7. **Run-ID convention.** Accept `YYYYMMDD-<name>-<hash>`, or prefer something else (e.g. `wandb`-style ulids)?
8. **Checkpoint migration.** Should the loader auto-detect old checkpoint paths from `Liars poker/agent/checkpoints/` and rewrite them into `data/checkpoints/`, or do a one-shot manual move?
9. **Test infra.** Pytest lives in a few scattered folders today — consolidate under `tests/` mirroring `src/`, or per-package `tests/` dirs?
10. **OpenSpiel bridge.** Willing to adopt OpenSpiel as a reference/infra dependency (unlocks free Deep CFR, NFSP, PSRO, MCCFR baselines — see §14.4), or keep the game engine fully in-house?

Resolved:

- ~~**Frontend hosting.**~~ (2026-04-24) Retire the public web demo; focus on local development. See §8.

## 14. Brainstorming — Paths Forward for SOTA Research

The overarching goal is state-of-the-art RL and AI-in-games research. The refactor described above is table-stakes for getting there — this section is a deliberately speculative menu of *where to point the pipeline next*. These are my suggestions as a collaborator, not commitments. Each item is labeled with a rough cost/value estimate so you can prune. Items interact; I've noted the key dependencies.

### 14.1 Exploitability as the North-Star Metric

**What:** For 2-player zero-sum games, the gold-standard benchmark is *exploitability* — the value a best-response opponent can extract against a fixed strategy. A Nash equilibrium has exploitability 0. Win-rate in round-robins is a weak proxy and can reward agents that merely beat weak opponents.

**How:** Implement a best-response solver that, given an agent's policy, computes the maximum-value counter-policy via backward induction over the public tree. Exploitability = ½(BR vs. P1 + BR vs. P2) − game value. Use it as (a) the headline benchmark alongside win-rate, (b) an evaluation signal during R-NaD / Deep CFR training.

**Why it's the right north star:** every major imperfect-information-game paper (CFR, DeepStack, Libratus, Pluribus, R-NaD) measures it. Without it we cannot credibly claim convergence.

**Cost:** medium — requires an efficient traverser of the public game tree, which we have the beginnings of in the CFR+ solver. **Value:** high, and unlocks the next items.

### 14.2 Small-Game Oracle Curriculum

**What:** Solve exact Nash for tiny variants (e.g. 2-card hands, deck subsets) using a canonical CFR+/LP solver. Use the resulting policies as ground truth to (a) validate learned agents converge to the right thing, (b) provide a curriculum where a learned agent walks from tiny → full-size games.

**Why it works:** Kuhn poker and Leduc poker play exactly this role in the literature. They are cheap enough to solve exactly, rich enough to exhibit bluffing, and serve as standard unit tests for new algorithms. A "Kuhn-Liar's-Poker" variant would be an original contribution.

**Cost:** medium — need a reduced-game factory and a reference LP/CFR solver. **Value:** high — buys correctness validation that neither win-rate nor exploitability alone provides.

### 14.3 Hand Abstraction / Bucketing

**What:** Classical poker-AI scaling trick (Libratus, Pluribus, DeepStack). Partition the private-hand space into equivalence classes (buckets) using a similarity metric — for Liar's Poker likely based on hand-rank distribution and pattern potential. Train the agent on the abstracted game, then "translate" the policy back at play time.

**Why we need it:** at hand size 5+ the raw information-set count explodes. Tabular CFR is already stretched. Bucketing is the standard way to get CFR to scale.

**Cost:** medium–high (designing a good abstraction is research in itself). **Value:** the main scaling lever for search-based agents.

### 14.4 OpenSpiel Integration

**What:** OpenSpiel (DeepMind) provides reference implementations of Deep CFR, NFSP, PSRO, MCCFR, R-NaD, exploitability solvers, and more — all working on any game that exposes their interface. CLAUDE.md already cites `openspiel`.

**How:** write a thin adapter so our `game/` package registers as an OpenSpiel game. Two-way: we get their algorithm zoo; they (and future researchers) get Liar's Poker as a testbed.

**Cost:** low–medium (the adapter is mechanical). **Value:** very high — order-of-magnitude reduction in baseline implementation work, plus instant credibility for the paper.

**Tradeoff:** creates a dependency. Framed as question Q10 above.

### 14.5 ReBeL / DeepStack-Style Continual Resolving

**What:** Instead of solving the whole game offline (CFR) or learning blindly from self-play (R-NaD alone), *resolve subgames at play time* using a learned value network at the leaves. This is the Libratus / ReBeL (Brown, Bakhtin, Lerer, Gong 2020) architecture and is still state of the art for large imperfect-information games.

**Why it fits Liar's Poker:** relatively short horizons, clean public-state structure (bid history is public), belief updates are analytically tractable. This is the most direct path to a genuinely strong agent.

**Cost:** high — this is a full research project, not a week of work. **Value:** this is the path to "state of the art." §10's Nash/CFR-redesign scope fence was written with ReBeL in mind.

### 14.6 PSRO / Population-Based Training

**What:** Maintain a *population* of agents. Each iteration, train a new agent as a best response to a mixture over the population; add it to the population; re-solve the meta-game. Converges to Nash under standard conditions and produces a diverse strategy zoo as a side effect.

**Why it's attractive here:** solves the problem of "agent X looks good because it only plays agent Y" in our current round-robin. Also gives the paper a natural way to present strategy diversity.

**Cost:** medium once infra is in place. **Value:** high for both research and demo purposes.

### 14.7 Engine Throughput — JAX Rewrite of the Hot Path

**What:** The game engine and hand evaluator are a bottleneck for any RL method that needs millions of rollouts. A JAX rewrite with `vmap` + `jit` over the evaluator would likely yield 100×+ speedup on GPU, and is the cheapest lever for more training throughput.

**Why now:** memory already records that "CPU beats MPS" for R-NaD because of sequential collection. A JAX rewrite fixes that at the root — batched rollout on accelerator.

**Cost:** medium (1–2 focused sessions). **Value:** multiplies every subsequent RL experiment.

### 14.8 Human-Play Dataset & Behavioral Cloning

**What:** Collect logs from real-human play sessions (yourself, friends, a terminal client, or a Mechanical Turk–style study). Use as (a) a supervised warm-start for learned agents, (b) a qualitative benchmark ("how human-like is this agent's bluffing distribution?"), (c) a novel contribution for the paper.

**Why it's research-grade:** almost no published work on imperfect-information games has real-human baselines beyond poker. Liar's Poker is niche enough that even a small dataset would be genuinely new.

**Cost:** low–medium (collection is the long pole; analysis is cheap). **Value:** directly strengthens the paper's narrative.

### 14.9 Experiment Tracking & Reproducibility Stack

**What:** Adopt Hydra for configs and Weights & Biases (or MLflow) for experiment tracking from day 1 of the refactor. Every run gets: resolved config, git SHA, seed, metrics time-series, artifacts. Free plans cover academic use.

**Why:** research without systematic tracking silently wastes 20–50% of effort on re-runs and forgotten results. This is cheap insurance.

**Cost:** low. **Value:** compounds over every subsequent experiment.

### 14.10 Theory + Empirics Pairing

**What:** Liar's Poker has clean theoretical structure that most RL-in-games papers don't get to exploit: the bid ladder is a totally ordered action space, belief updates from bids have tractable structure under exact rules, and "cheap talk" analysis from mechanism design is directly applicable. Pair empirical RL results with theoretical characterizations (e.g. "the learned policy matches a characterized pooling equilibrium in this regime").

**Why:** papers that combine strong empirics with even modest theory punch far above their weight at venues like NeurIPS, AAAI, and EC. Pure-empirical papers at the same venues are saturated.

**Cost:** the highest of any item here — requires sustained theory work. **Value:** potentially the difference between a technical report and a publishable-at-top-venue paper.

### 14.11 Process Improvements (meta)

Smaller, compounding suggestions for how we work, orthogonal to the above:

- **Auto-generate paper tables from `data/runs/`.** Decouple "run experiment" from "update paper" so numbers never drift. Already hinted at by the CLAUDE.md JSON-cache discipline.
- **Agent-card README per agent.** When each agent lives in its own file under `src/agents/`, include a short front-matter block (algorithm summary, hyperparams, known failure modes, last benchmarked date). Cheap, and future-you / reviewers will thank you.
- **ADR log under `docs-internal/design/adr/`.** One short markdown per irreversible decision (frontend retirement, OpenSpiel adoption, etc.). Replaces the "why did we decide this?" context-burn.
- **Reproducibility harness in CI.** A GitHub Action that runs a tiny version of the benchmark on every PR and diffs win-rates against a frozen baseline. Catches regressions like the pre-retirement JS/Python drift.
- **Nightly "reflection" cron.** Once §9's `reflect.py` exists, schedule it nightly against the current head; surface any newly-introduced flaw classes before they compound.

### 14.12 Suggested Ordering

If I were sequencing the research program around the refactor:

1. Refactor (P1) + retire frontend (P2) + decision logging (P3). *Prerequisite for everything.*
2. Exploitability solver (14.1) + experiment tracking (14.9). *No more flying blind.*
3. OpenSpiel adapter (14.4) — if Q10 is yes. *Buys the algorithm zoo.*
4. Small-game oracle curriculum (14.2). *Correctness floor.*
5. JAX engine rewrite (14.7). *Throughput unlock.*
6. Pick **one** of: ReBeL-style continual resolving (14.5), PSRO (14.6), hand abstraction (14.3). All three is a PhD; one is a paper.
7. Human-play dataset (14.8) and/or theory pairing (14.10) — as the writing phase approaches.

This ordering is a suggestion, not a plan. Please push back on any item you disagree with — items you veto can be dropped from the next plan doc.

---

*Next artifact:* once Q1–Q10 are answered, produce **`TRAINING_PIPELINE_PLAN.md`** — the concrete checklist of file moves, interface stubs, and migration ordering.
