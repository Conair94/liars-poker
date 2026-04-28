"""AR-0a tests for the canonical checkpoint schema."""

from __future__ import annotations

import pytest
import torch

from agents.checkpoints import SCHEMA_VERSION, load_component, save_component


def test_save_and_load_round_trip(tmp_path):
    net = torch.nn.Linear(3, 4)
    path = str(tmp_path / "small.pt")
    save_component(
        path,
        component="handmodel",
        config={"hidden_dim": 64, "depth": 2},
        state_dict=net.state_dict(),
        iter=1234,
        parent_run="ar1-2026-04-28T00:00:00-pilot-89e3443",
    )
    blob = load_component(path)
    assert blob["component"] == "handmodel"
    assert blob["config"]["hidden_dim"] == 64
    assert blob["iter"] == 1234
    assert blob["parent_run"].startswith("ar1-")
    assert blob["schema_version"] == SCHEMA_VERSION
    assert "git_sha" in blob and isinstance(blob["git_sha"], str)
    # state_dict keys parity
    assert set(blob["state_dict"].keys()) == set(net.state_dict().keys())


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_component(str(tmp_path / "no-such-file.pt"))


def test_save_rejects_unknown_component(tmp_path):
    path = str(tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="unknown component"):
        save_component(
            path,
            component="not_a_real_component",  # type: ignore[arg-type]
            config={},
            state_dict={},
            iter=0,
        )


def test_save_atomic_write(tmp_path):
    """A successful save leaves no .tmp file behind."""
    path = str(tmp_path / "atomic.pt")
    save_component(path, component="callpolicy", config={}, state_dict={}, iter=0)
    assert not (tmp_path / "atomic.pt.tmp").exists()
