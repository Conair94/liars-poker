# Archive

Everything under `archive/` is **frozen and read-only**. Subdirectories preserve historical artifacts that have been removed from the active codebase but are kept on-disk for reference, attribution, or possible future resurrection.

## Conventions

- One subdirectory per archived component, named `<component>-YYYY-MM/` (the date is when the component was retired, not when it was created).
- Each subdirectory should retain its original internal layout — do **not** restructure on archive.
- No code under `archive/` is imported, tested, deployed, or otherwise wired into the live system.
- Resurrecting an archived component is a deliberate act: copy out, modernize against current interfaces, and re-add to the active tree. Do not edit in place.

## Current contents

- `web-2026-04/` — public GitHub Pages JS demo + FastAPI backend retired per ADR-001 / ADR-004 on 2026-04-25. Contains the original `docs/` (compiled JS client) and `Liars poker/agent/web/` (FastAPI app + frontend sources).
