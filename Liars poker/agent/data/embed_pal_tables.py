"""
embed_pal_tables.py — Embed EXACT_RULES_PAL and FIVE_KINGS_PAL into docs/index.html.

Reads exact_rules_probs.json and five_kings_probs.json from agent/data/, then
replaces the placeholder `const EXACT_RULES_PAL = {};` and `const FIVE_KINGS_PAL = {};`
lines in docs/index.html with the full probability arrays.

Usage (from repo root or Liars poker/ dir):
    python -m agent.data.embed_pal_tables
    python -m agent.data.embed_pal_tables --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE       = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR  = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO_ROOT  = os.path.abspath(os.path.join(PAPER_DIR, ".."))

EXACT_RULES_FILE = os.path.join(HERE, "exact_rules_probs.json")
FIVE_KINGS_FILE  = os.path.join(HERE, "five_kings_probs.json")
INDEX_HTML       = os.path.join(REPO_ROOT, "docs", "index.html")


def _round_vec(vec: list, decimals: int = 5) -> list:
    return [round(v, decimals) for v in vec]


def _build_pal_js(json_path: str, key: str = "at_least") -> str:
    """Build a JS object literal string from a probs JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    lines = []
    for n_str, entry in sorted(data.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1):
        if not n_str.isdigit():
            continue
        vec = entry[key]
        rounded = _round_vec(vec)
        lines.append(f'  "{n_str}":{json.dumps(rounded)}')

    return "{\n" + ",\n".join(lines) + "\n}"


def embed(dry_run: bool = False) -> None:
    missing = []
    if not os.path.exists(EXACT_RULES_FILE):
        missing.append(EXACT_RULES_FILE)
    if not os.path.exists(FIVE_KINGS_FILE):
        missing.append(FIVE_KINGS_FILE)
    if missing:
        print(f"[embed_pal_tables] Missing files — run the compute scripts first:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    with open(INDEX_HTML) as f:
        html = f.read()

    # Build replacement strings
    exact_js = _build_pal_js(EXACT_RULES_FILE, key="exact")   # exact_prob[i], NOT at_least
    fk_js    = _build_pal_js(FIVE_KINGS_FILE,  key="at_least")

    # Replace placeholder or existing const declarations
    exact_pattern = r'const EXACT_RULES_PAL = \{[^;]*\};'
    fk_pattern    = r'const FIVE_KINGS_PAL = \{[^;]*\};'

    exact_comment = (
        "// EXACT_RULES_PAL[n][i] = P(pool contains 5-card exact match >= bid_i | n)\n"
        "// Populated at build time from exact_rules_probs.json. Empty = agents fall back to BlindBaselineAgent."
    )
    fk_comment = (
        "// FIVE_KINGS_PAL[n][i] = P(pool_best >= bid_i | n, 53-card deck)\n"
        "// Populated at build time from five_kings_probs.json. Empty = agents fall back to BlindBaselineAgent."
    )

    # Remove existing comments if present (we'll re-add them)
    html = re.sub(r'// EXACT_RULES_PAL\[n\]\[i\][^\n]*\n// Populated[^\n]*\n', '', html)
    html = re.sub(r'// FIVE_KINGS_PAL\[n\]\[i\][^\n]*\n// Populated[^\n]*\n', '', html)

    new_exact = f"{exact_comment}\nconst EXACT_RULES_PAL = {exact_js};"
    new_fk    = f"{fk_comment}\nconst FIVE_KINGS_PAL = {fk_js};"

    html_new = re.sub(exact_pattern, new_exact, html, flags=re.DOTALL)
    if html_new == html:
        print("[embed_pal_tables] WARNING: EXACT_RULES_PAL pattern not found — nothing replaced.")
    else:
        print(f"[embed_pal_tables] Embedded EXACT_RULES_PAL ({len(exact_js)} chars)")
    html = html_new

    html_new = re.sub(fk_pattern, new_fk, html, flags=re.DOTALL)
    if html_new == html:
        print("[embed_pal_tables] WARNING: FIVE_KINGS_PAL pattern not found — nothing replaced.")
    else:
        print(f"[embed_pal_tables] Embedded FIVE_KINGS_PAL ({len(fk_js)} chars)")
    html = html_new

    if dry_run:
        print("[embed_pal_tables] Dry run — no file written.")
    else:
        with open(INDEX_HTML, "w") as f:
            f.write(html)
        print(f"[embed_pal_tables] Written → {INDEX_HTML}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed PAL tables into docs/index.html")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit without writing")
    args = parser.parse_args()
    embed(dry_run=args.dry_run)
