"""
AR-0b compare reporter — markdown summary from sweep bench results.

Usage:
    python -m training.compare <sweep_id> [<sweep_id> ...]

Reads data/sweeps/<sweep_id>/bench/results.parquet (or .csv fallback)
and writes data/sweeps/<sweep_id>/report/summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_SWEEPS = _REPO_ROOT / "data" / "sweeps"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_results(bench_dir: Path) -> list[dict]:
    parquet_path = bench_dir / "results.parquet"
    csv_path = bench_dir / "results.csv"
    try:
        import pandas as pd  # type: ignore[import]
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"no results file in {bench_dir}")
        return df.to_dict(orient="records")
    except ImportError:
        # pandas not available — parse CSV by hand
        if not csv_path.exists():
            raise FileNotFoundError(f"no results.csv in {bench_dir}")
        rows = []
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
            for line in f:
                vals = line.strip().split(",")
                rows.append(dict(zip(header, vals)))
        return rows


def _format_ci(ci_low, ci_high) -> str:
    try:
        return f"[{float(ci_low):.3f}, {float(ci_high):.3f}]"
    except (TypeError, ValueError):
        return "N/A"


def generate_report(sweep_id: str) -> str:
    sweep_dir = _DATA_SWEEPS / sweep_id
    bench_dir = sweep_dir / "bench"
    rows = _load_results(bench_dir)

    winrate_rows = [r for r in rows if str(r.get("metric", "")) == "winrate"]

    lines: list[str] = []
    lines.append(f"# Sweep {sweep_id}")
    lines.append("")
    lines.append(f"_Generated {_utc_now()}_")
    lines.append("")

    if winrate_rows:
        games_per_pair = winrate_rows[0].get("games_per_pair", "?")
        lines.append(f"## Win-rate matrix ({games_per_pair} games per pair)")
        lines.append("")
        lines.append("| run_id | opponent | win_rate | 95% CI |")
        lines.append("| --- | --- | --- | --- |")

        def sort_key(r):
            try:
                return -float(r.get("value", 0))
            except (TypeError, ValueError):
                return 0.0

        for row in sorted(winrate_rows, key=sort_key):
            run_id = row.get("run_id", "")
            opponent = row.get("opponent", "")
            try:
                win_rate = float(row.get("value", 0))
            except (TypeError, ValueError):
                win_rate = float("nan")
            ci = _format_ci(row.get("ci_low"), row.get("ci_high"))
            lines.append(f"| {run_id} | {opponent} | {win_rate:.4f} | {ci} |")
        lines.append("")

        # Best by metric
        lines.append("## Best by metric")
        lines.append("")
        if winrate_rows:
            best = max(winrate_rows, key=lambda r: float(r.get("value", 0)) if r.get("value") else 0)
            best_run = best.get("run_id", "?")
            best_val = float(best.get("value", 0))
            lines.append(f"- `winrate` vs heuristic ladder: {best_run} at {best_val:.4f}")
        lines.append("- `lbr` (lower is better): N/A (no LBR run yet)")
        lines.append("")
    else:
        lines.append("_No win-rate data found in bench results._")
        lines.append("")

    return "\n".join(lines)


def write_report(sweep_id: str) -> Path:
    report_dir = _DATA_SWEEPS / sweep_id / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "summary.md"
    content = generate_report(sweep_id)
    out.write_text(content)
    print(f"[compare] wrote {out}", flush=True)
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(description="AR-0b sweep comparator")
    parser.add_argument("sweep_ids", nargs="+", help="one or more sweep_id values")
    args = parser.parse_args()
    for sweep_id in args.sweep_ids:
        write_report(sweep_id)


if __name__ == "__main__":
    _main()
