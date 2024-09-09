"""🧪 Tests for `skald.Logger`."""

# ruff: noqa: S101, D103, ANN001, ANN201, SLF001, PLR2004
from __future__ import annotations

from typing import TYPE_CHECKING

from skald import combine_runs
from skald.utils import get_skald_runs, is_skald_run

if TYPE_CHECKING:
    from pathlib import Path


def test_is_skald_run(skald_dir: Path, other_dir: Path):
    """❓ Tests the skald run check."""
    assert is_skald_run(skald_dir)
    assert not is_skald_run(other_dir)


def test_get_skald_runs(skald_multirun_dir):
    """🔎 Tests the get skald runs function."""
    skald_runs = get_skald_runs(skald_multirun_dir)
    assert len(skald_runs) == 2


def test_combine_runs(skald_multirun_dir):
    """📂 Tests skald.combine_runs."""
    metrics = combine_runs(
        skald_multirun_dir,
        include_params=["learning_rate", "train.num_epochs", "not_present"],
    )
    assert len(metrics) == 9
    assert "learning_rate" in metrics.columns
    assert "train.num_epochs" in metrics.columns
    assert "sizes" not in metrics.columns
