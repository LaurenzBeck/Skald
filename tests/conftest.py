"""⚗️📄 Global fixtures for all tests.

Since they are in a `conftest.py` file,
they are available without explicit imports in all tests.
"""

# ruff: noqa: S101, D103, ANN001, ANN201
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from skald import Logger

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def metrics() -> list[dict]:
    """📈 A list of metrics."""
    return [
        {"name": "metric1", "value": 1, "step": 1},
        {"name": "metric2", "value": 0.5, "step": 1},
        {"name": "metric1", "value": 1, "step": 2},
        {"name": "metric2", "value": 1.0, "step": 2},
    ]


@pytest.fixture()
def multikey_metrics() -> list[dict]:
    """🗝️🔑 A list of metricswith multiple keys."""
    return [
        {"name": "loss", "value": 1.1, "step": 1, "stage": "train"},
        {"name": "loss", "value": 1.1, "step": 2, "stage": "train"},
        {"name": "loss", "value": 1.3, "step": 1, "stage": "test"},
        {"name": "loss", "value": 1.3, "step": 2, "stage": "test"},
        {"name": "avg_loss", "value": 1.2, "step": 1},
    ]


@pytest.fixture()
def params() -> dict:
    """⚙️ parameters."""
    return {
        "learning_rate": 0.9,
        "sizes": [0.8, 0.1, 0.1],
        "train": {
            "gradient_clipping": True,
            "num_epochs": 100,
        },
    }


@pytest.fixture()
def params_modified() -> dict:
    """⚙️ modified parameters."""
    return {
        "learning_rate": 1.0,
        "sizes": [1.0, 1.0, 1.0],
        "train": {
            "gradient_clipping": True,
            "num_epochs": 100,
        },
    }


@pytest.fixture()
def skald_dir(tmp_path_factory, metrics: list[dict], params: dict) -> Path:
    """📂 a directory containing a skald run."""
    run_dir = tmp_path_factory.mktemp("skald_dir")
    with Logger(run_dir, "test") as logger:
        for metric in metrics:
            name = metric.pop("name")
            value = metric.pop("value")
            ids = metric
            logger.log_metric(name, value, **ids)
        logger.log_params(params)
        return logger.run_dir


@pytest.fixture()
def skald_multirun_dir(
    tmp_path_factory,
    metrics: list[dict],
    multikey_metrics: list[dict],
    params: dict,
    params_modified,
) -> Path:
    """📂 a directory containing skald runs."""
    run_dir = tmp_path_factory.mktemp("reports")
    with Logger(run_dir, "1") as logger:
        for metric in metrics:
            name = metric.pop("name")
            value = metric.pop("value")
            ids = metric
            logger.log_metric(name, value, **ids)
        logger.log_params(params)
    with Logger(run_dir, "2") as logger:
        for metric in multikey_metrics:
            name = metric.pop("name")
            value = metric.pop("value")
            ids = metric
            logger.log_metric(name, value, **ids)
        logger.log_params(params_modified)

    return run_dir


@pytest.fixture()
def other_dir(tmp_path_factory) -> Path:
    """📂 a directory containing arbitrary files."""
    directory = tmp_path_factory.mktemp("other_dir")

    (directory / "subdirectory").mkdir()
    with (directory / "a.txt").open("w") as f:
        f.write("hello there!")

    # a file with skald naming for confusion
    with (directory / "metrics.csv").open("w") as f:
        f.write("totally a csv file")

    return directory
