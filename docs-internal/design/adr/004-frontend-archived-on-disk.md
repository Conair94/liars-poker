# ADR-004: Frontend archived on disk

- **Status:** Accepted
- **Date:** 2026-04-25
- **Supersedes:** —
- **Related:** ADR-001 (decided the retirement)

## Context

ADR-001 (2026-04-24) decided to retire the public web demo. This ADR records the **execution** of that decision: where the code lives now, what the public-facing surface looks like, and what is required to bring it back.

ADR-001 captures the *why*; this one captures the *where* and *how-to-undo*.

## Decision

The frontend is archived on-disk under `archive/web-2026-04/`, preserving the original layout:

- `archive/web-2026-04/docs/` — the compiled JS client that GitHub Pages was serving (originally at repo-root `docs/`).
- `archive/web-2026-04/web/` — the FastAPI backend + frontend sources (originally at `Liars poker/agent/web/`).

The archive is read-only by convention (`archive/README.md`). No active code under `src/`, `tests/`, or `paper/` imports from it.

The public deployment is taken offline:

- GitHub Pages source set to `None` in repo settings (user action, 2026-04-25).
- No GitHub Actions workflow currently deploys Pages — the previous deployment was the built-in "deploy from branch" mode, so disabling it in settings is sufficient. (Repo has no `.github/workflows/` directory at the time of archiving; nothing to delete.)
- `https://<user>.github.io/liars-poker/` returns 404 once GitHub propagates the change.

## Consequences

**Positive:**

- Source-of-truth for the JS client is preserved (commit history + on-disk copy) without any risk of silent drift, since nothing imports it.
- A future "Stage 2 web demo" effort starts from a known artifact rather than a bare slate.

**Negative:**

- Resurrection requires rewriting the JS client against the modular `Agent` interface defined in `TRAINING_PIPELINE_DESIGN.md` §7. The archived JS will have drifted further the longer it sits.
- Anyone with a bookmark to the public URL gets a 404. We accept this — there is no traffic of consequence.

## Reversal

To restore the public demo:

1. Copy (do not move) `archive/web-2026-04/docs/` back to repo-root `docs/`.
2. Copy `archive/web-2026-04/web/` back to a live location (likely `src/web/`, not `Liars poker/...`, since the legacy directory is removed in P6).
3. Re-implement the registry and ruleset wiring against the current `src/agents/registry.py` interface — the archived `web/backend/agents.py` predates the registry extraction in P1.8.
4. Re-enable GitHub Pages in repo settings.

Step 3 is the bulk of the work; steps 1, 2, and 4 are mechanical.
