# ADR-002: OpenSpiel adoption

- **Status:** Accepted
- **Date:** 2026-04-24
- **Supersedes:** —

## Context

The project currently has a hand-rolled game engine, hand-rolled CFR/CFR+ solver, hand-rolled R-NaD trainer, and no exploitability metric. This means:

- We cannot cheaply check whether our solver actually converges to Nash, only whether one of our agents beats another. Win-rate is a noisy, opponent-dependent signal.
- We have no oracle for small-game variants (e.g. a Kuhn-sized reduction) that could validate the solver's correctness independently.
- Future work (PSRO, ReBeL-style continual resolving — `TRAINING_PIPELINE_DESIGN.md` §14.5–14.6) duplicates infrastructure that is already mature in OpenSpiel (Lanctot et al. 2019, arXiv:1908.09453).

Options considered:

- **A.** Continue building bespoke infrastructure. Rejected — high engineering cost, no validation oracle, ecosystem isolation.
- **B.** Rewrite the engine on top of OpenSpiel and discard the in-house solver. Rejected — the in-house engine and CFR+ solver are already validated and tuned for this game; rewriting incurs risk for marginal gain.
- **C.** Adopt OpenSpiel as an interop layer: keep the in-house engine as the source of truth for rules, expose Liar's Poker via an OpenSpiel adapter so we get exploitability, tabular-policy tooling, and small-game oracles for free. **Selected.**

## Decision

Adopt OpenSpiel as a peer dependency via an adapter layer (`src/interop/openspiel_adapter.py`). The in-house engine remains the canonical rules implementation. The adapter exposes the canonical engine to OpenSpiel's APIs so we can use:

- `pyspiel.exploitability` as a primary correctness metric for any tabular- or callable-policy agent.
- A reduced "Kuhn-style" Liar's Poker variant solved exactly with OpenSpiel's CFR+ as a regression oracle for our solver.
- OpenSpiel's PSRO / NFSP / continual-resolving implementations as off-the-shelf research targets later.

Round-trip equivalence (legal-action sets, terminal rewards) between the two engines is enforced by tests on every PR.

## Consequences

**Positive:**

- Free exploitability metric — every benchmark run reports it alongside win-rate.
- Independent oracle for solver validation — if our CFR+ disagrees with OpenSpiel's CFR+ on a Kuhn-sized variant, the bug is in our solver.
- Cheap path to PSRO and ReBeL-style methods later.
- Aligns with literature, which makes the paper easier to write and review.

**Negative:**

- Adds a non-trivial C++ build dependency (`open_spiel`) to anyone running the project. Mitigated by making it an optional extra in `pyproject.toml` (`pip install -e ".[openspiel]"`).
- Adapter must be kept in sync as the in-house engine evolves. Mitigated by the round-trip equivalence test running on every PR.
- One more abstraction layer between agents and the game; we accept this for the metric leverage it buys.

## Execution

Adapter implementation is scheduled in P4 of `TRAINING_PIPELINE_PLAN.md`. The canonical OpenSpiel game ID and information-state encoding are pinned by ADR-005 once that phase lands.
