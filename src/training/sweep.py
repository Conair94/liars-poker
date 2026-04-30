"""
AR-0b sweep driver.

Usage:
    python -m training.sweep configs/sweeps/heuristic_ladder_smoketest.yaml \\
        [--max-parallel 4] [--resume] [--dry-run]

Each cell in the sweep gets its own data/runs/<run_id>/ directory.
The sweep itself lives at data/sweeps/<sweep_id>/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_RUNS = _REPO_ROOT / "data" / "runs"
_DATA_SWEEPS = _REPO_ROOT / "data" / "sweeps"


# ---------------------------------------------------------------------------
# git helpers

def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "00000000"


def _git_is_clean() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.strip() == b""
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Run / sweep id helpers

def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _cell_slug(cell: dict[str, Any]) -> str:
    """Deterministic short name for a sweep cell dict."""
    parts = []
    for k in sorted(cell):
        v = str(cell[k]).replace(" ", "_").replace("/", "-")
        parts.append(f"{k[0]}{v}")
    slug = "-".join(parts)
    return slug[:40]


def make_run_id(phase: str, slug: str, sha: str) -> str:
    return f"{phase}-{_utc_now()}-{slug}-{sha}"


def make_sweep_id(phase: str, sweep_name: str, sha: str) -> str:
    return f"{phase}-{_utc_now()}-{sweep_name}-{sha}"


# ---------------------------------------------------------------------------
# Config loading and cell enumeration

def load_sweep_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _validate_sweep_config(cfg)
    return cfg


def _validate_sweep_config(cfg: dict) -> None:
    required = ["sweep_name", "phase", "runner"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"sweep config missing required key: {k!r}")
    if "random_seed" not in cfg:
        raise ValueError(
            "sweep config must specify 'random_seed' (reproducibility requirement)"
        )
    has_axes = "axes" in cfg
    has_cells = "cells" in cfg
    if has_axes and has_cells:
        raise ValueError("sweep config may not specify both 'axes' and 'cells'")
    if not has_axes and not has_cells:
        raise ValueError("sweep config must specify either 'axes' or 'cells'")


def enumerate_cells(cfg: dict) -> list[dict[str, Any]]:
    if "cells" in cfg:
        return list(cfg["cells"])
    axes = cfg["axes"]
    keys = [ax["key"] for ax in axes]
    value_lists = [ax["values"] for ax in axes]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


# ---------------------------------------------------------------------------
# Atomic manifest write

def _write_manifest(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _manifest_completed(run_dir: Path) -> bool:
    m = run_dir / "manifest.json"
    if not m.exists():
        return False
    try:
        data = json.loads(m.read_text())
        return bool(data.get("completed", False))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Resolved config: interpolate cell values into runner args

def _resolve_cell_config(cfg: dict, cell: dict, cell_index: int) -> dict:
    runner = cfg["runner"]
    args = dict(runner.get("args", {}))
    fixed = dict(runner.get("fixed", {}))
    # Template-substitute cell values into args
    resolved_args = {}
    for k, v in args.items():
        if isinstance(v, str):
            for ck, cv in cell.items():
                v = v.replace("{" + ck + "}", str(cv))
        resolved_args[k] = v
    return {
        "sweep_name": cfg["sweep_name"],
        "phase": cfg["phase"],
        "random_seed": cfg["random_seed"] + cell_index,
        "device": cfg.get("device", "cpu"),
        "cell": cell,
        "cell_index": cell_index,
        "runner": {
            "command": runner["command"],
            "args": resolved_args,
            "fixed": fixed,
        },
    }


def _write_resolved_config(run_dir: Path, resolved: dict) -> Path:
    out = run_dir / "config.yaml"
    with open(out, "w") as f:
        yaml.dump(resolved, f, default_flow_style=False, sort_keys=True)
    return out


# ---------------------------------------------------------------------------
# Subprocess runner

def _run_cell(
    run_id: str,
    sweep_id: str,
    resolved_cfg: dict,
    config_path: str,
    sha: str,
) -> dict:
    """Execute one cell's runner as a subprocess. Returns a result dict."""
    run_dir = _DATA_RUNS / run_id
    runner = resolved_cfg["runner"]
    cmd_base = runner["command"].split()
    args_flags = []
    for k, v in runner["args"].items():
        args_flags += [f"--{k}", str(v)]
    for k, v in runner["fixed"].items():
        args_flags += [f"--{k}", str(v)]
    cmd = (
        cmd_base
        + ["--config", config_path, "--run-id", run_id]
        + args_flags
    )
    print(f"[sweep] start  {run_id}", flush=True)
    # Rewrite "python -m foo" → "[sys.executable] -m foo" so the subprocess
    # uses the same interpreter that the sweep driver runs under.
    if cmd[0] in ("python", "python3"):
        exec_cmd = [sys.executable] + cmd[1:]
    else:
        exec_cmd = cmd
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO_ROOT / "src"))
    proc = subprocess.run(exec_cmd, capture_output=False, cwd=str(_REPO_ROOT), env=env)
    exit_code = proc.returncode
    completed = _manifest_completed(run_dir)
    print(
        f"[sweep] finish {run_id}  exit={exit_code}  completed={completed}",
        flush=True,
    )
    return {"run_id": run_id, "exit_code": exit_code, "completed": completed}


# ---------------------------------------------------------------------------
# Sweep orchestration

def run_sweep(config_path: str, max_parallel: int = 1, resume: bool = False,
              dry_run: bool = False) -> None:
    cfg = load_sweep_config(config_path)

    if cfg.get("production", False) and not _git_is_clean():
        raise RuntimeError(
            "sweep config has production=true but git tree is dirty; "
            "commit or stash changes first"
        )

    sha = _git_sha()
    sweep_id = make_sweep_id(cfg["phase"], cfg["sweep_name"], sha)
    cells = enumerate_cells(cfg)

    sweep_dir = _DATA_SWEEPS / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    bench_dir = sweep_dir / "bench"
    bench_dir.mkdir(exist_ok=True)
    runs_link_dir = sweep_dir / "runs"
    runs_link_dir.mkdir(exist_ok=True)

    # Copy sweep config verbatim
    import shutil
    shutil.copy2(config_path, sweep_dir / "sweep.yaml")

    index: dict[str, str] = {}  # cell_key → run_id
    run_jobs: list[tuple[str, str, dict, str]] = []  # (run_id, sweep_id, resolved, config_path)

    for i, cell in enumerate(cells):
        slug = _cell_slug(cell)
        run_id = make_run_id(cfg["phase"], slug, sha)
        index[json.dumps(cell, sort_keys=True)] = run_id
        run_dir = _DATA_RUNS / run_id

        if resume and _manifest_completed(run_dir):
            print(f"[sweep] skip   {run_id} (already completed)", flush=True)
            continue

        if resume and run_dir.exists():
            import shutil as _sh
            _sh.rmtree(run_dir)
            print(f"[sweep] reset  {run_id} (incomplete, will re-run)", flush=True)

        resolved = _resolve_cell_config(cfg, cell, i)
        resolved["run_id"] = run_id
        resolved["sweep_id"] = sweep_id
        resolved["git_sha"] = sha

        if not dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)

            # Symlink into sweep/runs/
            link = runs_link_dir / run_id
            if not link.exists():
                try:
                    link.symlink_to(run_dir.resolve())
                except OSError:
                    pass  # non-fatal on systems that disallow symlinks

            config_out = _write_resolved_config(run_dir, resolved)

            # Write initial manifest (incomplete)
            _write_manifest(
                run_dir / "manifest.json",
                {
                    "run_id": run_id,
                    "sweep_id": sweep_id,
                    "git_sha": sha,
                    "started_at": _utc_now(),
                    "last_completed_iter": 0,
                    "completed": False,
                    "exit_code": None,
                    "components": {"handmodel": None, "callpolicy": None, "bidpolicy": None},
                },
            )
            run_jobs.append((run_id, sweep_id, resolved, str(config_out)))

    # Write index.json
    _write_manifest(sweep_dir / "index.json", index)

    if dry_run:
        print(f"[sweep] dry-run: {len(cells)} cells enumerated, none executed")
        return

    if not run_jobs:
        print("[sweep] all cells already completed", flush=True)
        return

    if max_parallel == 1:
        for run_id, sweep_id_inner, resolved, cfg_path in run_jobs:
            _run_cell(run_id, sweep_id_inner, resolved, cfg_path, sha)
    else:
        with ProcessPoolExecutor(max_workers=max_parallel) as pool:
            futures = {
                pool.submit(_run_cell, run_id, sweep_id_inner, resolved, cfg_path, sha): run_id
                for run_id, sweep_id_inner, resolved, cfg_path in run_jobs
            }
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    run_id = futures[fut]
                    print(f"[sweep] ERROR  {run_id}: {exc}", file=sys.stderr, flush=True)

    print(f"[sweep] done   sweep_id={sweep_id}", flush=True)


# ---------------------------------------------------------------------------
# Entry point

def _main() -> None:
    parser = argparse.ArgumentParser(description="AR-0b sweep driver")
    parser.add_argument("config", help="path to sweep YAML config")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="skip completed cells; re-run failed/incomplete from scratch")
    parser.add_argument("--dry-run", action="store_true",
                        help="enumerate cells without running them")
    args = parser.parse_args()
    run_sweep(args.config, max_parallel=args.max_parallel,
              resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    _main()
