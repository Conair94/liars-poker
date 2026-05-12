"""AR-2 Phase 7 — sweep-summary elbow detector unit tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC  = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from training.ar2_sweep_summary import (  # noqa: E402
    compute_slopes,
    select_elbow,
    summarise_sweep,
)


def test_compute_slopes_basic():
    # Simple monotone improvement.
    kl = {
        4:  {1000: 1.0, 5000: 0.5, 10000: 0.4, 50000: 0.39},
        6:  {1000: 1.0, 5000: 0.6, 10000: 0.5, 50000: 0.49},
        8:  {1000: 1.0, 5000: 0.7, 10000: 0.6, 50000: 0.59},
        10: {1000: 1.0, 5000: 0.8, 10000: 0.7, 50000: 0.69},
    }
    slopes = compute_slopes(kl)
    # slope at N=5000 requires N=2500 which isn't in the data → None.
    assert slopes[4][5000] is None
    # slope at N=50000 requires N=25000 which isn't in the data → None.
    assert slopes[4][50000] is None
    # slope at N=10000 requires N=5000 (present): (0.5 - 0.4) / 0.5 = 0.2
    assert abs(slopes[4][10000] - 0.2) < 1e-9


def test_select_elbow_plateau_at_10k():
    # Plateau (<5% improvement) at N=10k for every n.
    kl = {
        4:  {5000: 0.10, 10000: 0.099, 50000: 0.098},
        6:  {5000: 0.15, 10000: 0.148, 50000: 0.147},
        8:  {5000: 0.20, 10000: 0.198, 50000: 0.197},
        10: {5000: 0.25, 10000: 0.247, 50000: 0.246},
    }
    chosen, reason, _ = select_elbow(kl)
    assert chosen == 10000
    assert reason == "elbow_at_N=10000"


def test_select_elbow_fallback_monotone():
    # Every doubling produces a big improvement; no elbow.
    kl = {
        4:  {1000: 1.0, 5000: 0.5, 10000: 0.25, 50000: 0.125},
        6:  {1000: 1.0, 5000: 0.5, 10000: 0.25, 50000: 0.125},
        8:  {1000: 1.0, 5000: 0.5, 10000: 0.25, 50000: 0.125},
        10: {1000: 1.0, 5000: 0.5, 10000: 0.25, 50000: 0.125},
    }
    chosen, reason, _ = select_elbow(kl)
    assert chosen == 10000
    assert reason == "fallback_per_design_3.3"


def test_select_elbow_smallest_computable_at_5k():
    # Plateau begins at N=5k. slope_n(5k) is computable iff N=2500 is in the
    # data — it isn't, so the smallest computable elbow point is N=10k.
    kl = {
        4:  {1000: 1.0, 5000: 0.99, 10000: 0.985, 50000: 0.98},
        6:  {1000: 1.0, 5000: 0.99, 10000: 0.985, 50000: 0.98},
        8:  {1000: 1.0, 5000: 0.99, 10000: 0.985, 50000: 0.98},
        10: {1000: 1.0, 5000: 0.99, 10000: 0.985, 50000: 0.98},
    }
    chosen, reason, _ = select_elbow(kl)
    assert chosen == 10000
    assert reason == "elbow_at_N=10000"


def _write_cell(data_root: Path, run_id: str, N: int, kl_per_n: dict) -> None:
    run_dir = data_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ar2_summary.json").write_text(json.dumps({
        "N": N, "seed": 0, "run_id": run_id,
        "callpolicy_val_bce_best": 0.5,
        "bidpolicy_val_kl_best": sum(kl_per_n.values()) / len(kl_per_n),
        "bidpolicy_val_kl_per_n": {str(k): v for k, v in kl_per_n.items()},
        "callpolicy_ckpt": str(run_dir / "callpolicy" / "best.pt"),
        "bidpolicy_ckpt":  str(run_dir / "bidpolicy"  / "best.pt"),
    }))


def test_summarise_sweep_end_to_end(tmp_path):
    sweep_id = "ar2-test-sweep"
    data_root = tmp_path
    sweep_dir = data_root / "sweeps" / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Plateau at N=10000.
    cells_kl = {
        1000:  {4: 1.0,  6: 1.0,  8: 1.0,  10: 1.0},
        5000:  {4: 0.5,  6: 0.6,  8: 0.7,  10: 0.8},
        10000: {4: 0.49, 6: 0.59, 8: 0.69, 10: 0.79},
        50000: {4: 0.485, 6: 0.585, 8: 0.685, 10: 0.785},
    }
    index = {}
    for N, kl_per_n in cells_kl.items():
        run_id = f"ar2-cell-N{N}"
        _write_cell(data_root, run_id, N, kl_per_n)
        index[json.dumps({"N": N}, sort_keys=True)] = run_id
    (sweep_dir / "index.json").write_text(json.dumps(index))

    elbow = summarise_sweep(sweep_id, data_root=str(data_root))
    assert elbow["chosen_N"] == 10000
    assert elbow["reason"] == "elbow_at_N=10000"
    # Artefacts.
    assert (sweep_dir / "ar2_kl_curve.json").exists()
    assert (sweep_dir / "elbow.json").exists()
    # Plot is optional (depends on matplotlib).
    assert elbow["callpolicy_ckpt"] is not None
    assert elbow["bidpolicy_ckpt"] is not None
