"""⚗️📄 Global fixtures for all tests.

Since they are in a `conftest.py` file,
they are available without explicit imports in all tests.
"""

# ruff: noqa: S101, D103, ANN001, ANN201
from __future__ import annotations

import pytest


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
