"""
Automated reflection v1 (TRAINING_PIPELINE_DESIGN §9).

Loads ``data/runs/<run_id>/decisions.jsonl``, runs the v1 rule set, and
writes ``summary.md`` to the same directory.

Rules implemented:
  - Infeasible-bid tripwire (v1)
  - Stale-bid-repetition cluster (v1)
  - Missed-call rule (v2 — P5-#3): low P(call) when a strong response is
    available
  - Rank-leak entropy check (v2 — P5-#3): opening-bid distribution differs
    too sharply across the agent's first private rank

Rules still deferred (require ``eu`` per choice — needs a value model):
  - Low-EU choice rule (chosen action's eu below the kth-best by margin)

Usage:
    python -m training.reflect <run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Path setup (mirrors benchmark.py so the module runs as a script)
_HERE      = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _iter_records(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------


def rule_infeasible_bid(records: list[dict]) -> dict[tuple, int]:
    """Tripwire: chosen action marked feasible=False in its own choices list.

    Post the 2026-04-24 feasibility-filter fix this should be 0.
    """
    flags: dict[tuple, int] = Counter()
    for r in records:
        chosen = r["chosen"]
        for c in r["choices"]:
            if c["action"] == chosen and c.get("feasible", True) is False:
                key = _group_key(r)
                flags[key] += 1
                break
    return flags


def rule_stale_bid_repetition(records: list[dict]) -> dict[tuple, int]:
    """Flag agents that repeatedly open a fresh round with the same bid.

    A v1 proxy for §9 'stale-bid repetition': within an (agent, opponent,
    ruleset) bucket, count the modal opening bid's share. If one opening
    bid is chosen on >70% of opening turns AND there were ≥10 such turns,
    flag the bucket once per repeated opening beyond the first.
    """
    by_bucket: dict[tuple, list[str]] = defaultdict(list)
    for r in records:
        if r["state"]["standing_bid"] is None and r["chosen"].startswith("bid:"):
            by_bucket[_group_key(r)].append(r["chosen"])

    flags: dict[tuple, int] = Counter()
    for bucket, openings in by_bucket.items():
        if len(openings) < 10:
            continue
        modal, count = Counter(openings).most_common(1)[0]
        if count / len(openings) > 0.70:
            flags[bucket] = count - 1  # first occurrence isn't stale
    return flags


def rule_missed_call(records: list[dict], threshold: float = 0.30) -> dict[tuple, int]:
    """Flag turns where P(call) is unexpectedly low when the standing bid is
    very strong.

    Heuristic: if the standing bid is a Pair-or-stronger and yet the agent's
    distribution puts < `threshold` mass on CALL, count it. This catches
    agents that bid up into untenable territory rather than calling. Without
    a hand-model posterior we cannot tell whether the bid is genuinely
    unsupported, so this is a *coarse* version of the design-doc rule.
    """
    flags: dict[tuple, int] = Counter()
    for r in records:
        sb = r["state"].get("standing_bid")
        if sb is None or not sb.startswith("bid:"):
            continue
        # `bid:<HAND_ABBREV>:<RANK>` — anything beyond HC counts as "strong".
        try:
            hand_abbrev = sb.split(":", 2)[1]
        except IndexError:
            continue
        if hand_abbrev == "HC":
            continue
        # Find P(call) in the choices list.
        call_prob = None
        for c in r["choices"]:
            if c["action"] == "call":
                call_prob = c.get("p")
                break
        if call_prob is None:
            continue  # agent does not expose `p`; skip silently
        if call_prob < threshold:
            flags[_group_key(r)] += 1
    return flags


def rule_rank_leak(records: list[dict],
                   min_openings: int = 30,
                   leak_threshold: float = 0.6) -> dict[tuple, int]:
    """Flag agents whose opening-bid distribution leaks information about
    the agent's strongest private card.

    For each (agent, opponent, ruleset) bucket, partition the agent's
    opening turns by ``hand[0]`` rank (the high card of the agent's sorted
    hand). Compute, for each rank-bucket, the modal opening-bid share. If
    in any rank-bucket >`leak_threshold` mass concentrates on a single bid
    that is rare (<10%) in OTHER rank-buckets, flag the bucket.

    Without bid probabilities this is a behavioural proxy: deterministic
    rank-conditioned bidding is the most blatant rank-leak failure.
    """
    by_bucket: dict[tuple, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["state"].get("standing_bid") is not None:
            continue
        if not r["chosen"].startswith("bid:"):
            continue
        hand = r["state"].get("hand") or []
        if not hand:
            continue
        # The hand list is in card-rank order (`<R><S>` strings); take the
        # high-card rank as the leak signal.
        hi_rank = hand[-1][0]
        by_bucket[_group_key(r)][hi_rank].append(r["chosen"])

    flags: dict[tuple, int] = Counter()
    for bucket, by_rank in by_bucket.items():
        total = sum(len(v) for v in by_rank.values())
        if total < min_openings:
            continue
        for rank, openings in by_rank.items():
            if len(openings) < 5:
                continue
            modal, count = Counter(openings).most_common(1)[0]
            share_in_rank = count / len(openings)
            if share_in_rank < leak_threshold:
                continue
            # How rare is `modal` in OTHER rank buckets?
            other = [o for k, lst in by_rank.items() if k != rank for o in lst]
            if not other:
                continue
            other_share = other.count(modal) / len(other)
            if other_share < 0.10:
                flags[bucket] += len(openings)
    return flags


def _group_key(r: dict) -> tuple:
    rs = r["ruleset"]
    return (
        r["agent"],
        r["opponent"],
        f"exact={rs.get('exact_rules')},hh={rs.get('high_hand')},mode={rs.get('mode')}",
    )


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _format_table(title: str, flags: dict[tuple, int], total_by_bucket: dict[tuple, int],
                  note: str = "") -> str:
    rows = sorted(
        flags.items(),
        key=lambda kv: (-kv[1] / max(total_by_bucket.get(kv[0], 1), 1), kv[0]),
    )
    out = [f"### {title}"]
    if note:
        out.append(f"_{note}_")
    if not rows:
        out.append("\n_No flags._\n")
        return "\n".join(out)
    out.append("")
    out.append("| agent | opponent | ruleset | flags | turns | rate |")
    out.append("|---|---|---|---:|---:|---:|")
    for (agent, opp, ruleset), n in rows:
        total = total_by_bucket.get((agent, opp, ruleset), 0) or 1
        rate = n / total
        out.append(f"| {agent} | {opp} | {ruleset} | {n} | {total} | {rate:.2%} |")
    out.append("")
    return "\n".join(out)


def _bucket_totals(records: list[dict]) -> dict[tuple, int]:
    totals: dict[tuple, int] = Counter()
    for r in records:
        totals[_group_key(r)] += 1
    return totals


def write_summary(records: list[dict], out_path: str, run_id: str,
                  load_seconds: float) -> None:
    totals = _bucket_totals(records)
    infeasible = rule_infeasible_bid(records)
    stale = rule_stale_bid_repetition(records)
    missed = rule_missed_call(records)
    leak = rule_rank_leak(records)

    parts = [
        f"# Reflection Summary — `{run_id}`",
        "",
        f"- Records: **{len(records):,}**",
        f"- Buckets (agent × opponent × ruleset): **{len(totals)}**",
        f"- Load time: {load_seconds:.2f}s",
        "",
        "## Implemented Rules",
        "",
        _format_table("Infeasible-bid tripwire", infeasible, totals,
                      note="Should be 0 post-2026-04-24 feasibility-filter fix."),
        _format_table("Stale-bid repetition (opening bids)", stale, totals,
                      note=">70% of openings reuse the same bid (≥10 openings sampled)."),
        _format_table("Missed-call (P5-#3)", missed, totals,
                      note="P(call)<0.30 when standing bid is Pair-or-stronger."),
        _format_table("Rank-leak (P5-#3)", leak, totals,
                      note="Opening-bid concentration > 60% in one hand-rank bucket "
                           "while < 10% in others (≥30 openings sampled)."),
        "",
        "## Deferred Rules (need EU per choice — needs value model)",
        "",
        "| rule | status |",
        "|---|---|",
        "| Low-EU choice (chosen.eu below kth-best by margin) | deferred: needs eu |",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_run_dir(run_id: str) -> str:
    direct = os.path.join(_REPO_ROOT, "data", "runs", run_id)
    if os.path.isdir(direct):
        return direct
    raise FileNotFoundError(f"run dir not found: {direct}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reflect on a benchmark run's decision log.")
    ap.add_argument("run_id", help="run id, matching data/runs/<run_id>/")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_id)
    decisions = os.path.join(run_dir, "decisions.jsonl")
    if not os.path.exists(decisions):
        raise FileNotFoundError(f"missing decisions log: {decisions}")

    t0 = time.time()
    records = list(_iter_records(decisions))
    load_seconds = time.time() - t0

    out_path = os.path.join(run_dir, "summary.md")
    write_summary(records, out_path, args.run_id, load_seconds)
    elapsed = time.time() - t0
    print(f"Wrote {out_path}  ({len(records):,} records, {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
