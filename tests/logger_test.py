"""🧪 Tests for `skald.Logger`."""

# ruff: noqa: S101, D103, ANN001, ANN201, SLF001, PLR2004
from __future__ import annotations

from typing import TYPE_CHECKING

import beartype
import matplotlib.pyplot as plt
import polars as pl
import pytest
import yaml

import skald
from skald.utils import flatten_dict

if TYPE_CHECKING:
    from pathlib import Path

    from skald.enums import MetricsFileFormat


def test_init_logger(tmp_path: Path):
    """📄 Tests creating a logger."""
    logger = skald.Logger(tmp_path, "test")

    assert logger.run_name == "test"

    assert (tmp_path / "test").is_dir()
    assert (tmp_path / "test" / "console.log").is_file()
    assert (tmp_path / "test" / "params.yaml").is_file()
    assert (tmp_path / "test" / "metrics.parquet").is_file()
    assert (tmp_path / "test" / "artifacts").is_dir()


def test_metric_logging_exceptions(tmp_path: Path):
    """🔥 Tests the logging of invalid metrics."""
    logger = skald.Logger(tmp_path, "test")

    with pytest.raises(beartype.roar.BeartypeCallHintParamViolation):
        logger.log_metric(1, 42)


@pytest.mark.parametrize("metrics_file_format", ["csv", "parquet"])
def test_metric_logging(
    tmp_path: Path,
    metrics: list[dict],
    metrics_file_format: MetricsFileFormat,
):
    """📈 Tests the logging of metrics."""
    logger = skald.Logger(tmp_path, "test", metrics_file_format=metrics_file_format)

    # initialize an metric with id=None to test a previous error with `pl.concat`
    logger.log_scalar("name", 42, step=None)

    for metric in metrics:
        name = metric.pop("name")
        value = metric.pop("value")
        ids = metric
        logger.log_metric(name, value, **ids)
        logger.log_scalar(name, value, **ids)

    # test the behaviour of a log call with a new id
    logger.log_scalar("name", 42, additional_id="😫")

    logger.log_metrics({"loss": 1.23, "accuracy": 0.9}, step=1)
    logger.log_scalars({"loss": 1.23, "accuracy": 0.9}, step=2)
    logger.log_dict({"loss": 1.23, "accuracy": 0.9}, step=3)

    match metrics_file_format:
        case "csv":
            saved_metrics = pl.read_csv(logger.metrics_file)
        case "parquet":
            saved_metrics = pl.read_parquet(logger.metrics_file)

    assert logger._metrics.equals(saved_metrics)


@pytest.mark.parametrize("metrics_file_format", ["csv", "parquet"])
def test_multikey_metric_logging(
    tmp_path: Path,
    multikey_metrics,
    metrics_file_format: MetricsFileFormat,
):
    """🔑🗝️ Tests the logging of metrics with multiple keys/ids."""
    with skald.Logger(
        tmp_path,
        "test",
        metrics_file_format=metrics_file_format,
        persistence_strategy="lazy",
    ) as logger:
        for metric in multikey_metrics:
            name = metric.pop("name")
            value = metric.pop("value")
            ids = metric
            logger.log_metric(name, value, **ids)
            logger.log_scalar(name, value, **ids)

        metrics = logger._metrics
        metrics_file = logger.metrics_file

    match metrics_file_format:
        case "csv":
            saved_metrics = pl.read_csv(metrics_file)
        case "parquet":
            saved_metrics = pl.read_parquet(metrics_file)

    assert metrics.equals(saved_metrics)


def test_params_logging(tmp_path: Path, params: dict) -> None:
    """⚙️ Tests the logging of parameters."""
    logger = skald.Logger(tmp_path, "test")

    flattened_params = flatten_dict(params) | logger._params

    learning_rate = params.pop("learning_rate")

    logger.log_param("learning_rate", learning_rate)
    logger.log_params(params)

    saved_params = yaml.safe_load(logger.params_file.read_text())

    assert flattened_params == saved_params


def test_figure_logging(tmp_path: Path) -> None:
    """🖼️ Tests the logging of matplotlib figures."""
    logger = skald.Logger(tmp_path, "test")

    fig = plt.figure()

    plt.plot([1, 2, 3])

    logger.log_figure("fig_1", fig)

    assert (logger.artifacts_dir / "fig_1.png").is_file()

    logger.log_figure("fig_2")

    assert (logger.artifacts_dir / "fig_2.png").is_file()


def test_log(tmp_path: Path, params: dict) -> None:
    """✨ Tests the type and argument based `Logger.log`."""
    logger = skald.Logger(tmp_path, "test")

    # str, float, **ids -> `log_metric`
    logger.log("loss", 0.9, stage="train")
    assert "loss" not in logger._params
    metrics = logger._metrics
    assert len(metrics) == 1
    assert metrics[0]["name"][0] == "loss"
    assert metrics[0]["value"][0] == 0.9
    assert metrics[0]["stage"][0] == "train"

    # dict, **ids -> `log_metrics`
    logger.log({"accuracy": 0.8}, stage="train")
    assert "accuracy" not in logger._params
    metrics = logger._metrics
    assert len(metrics) == 2
    assert metrics[1]["name"][0] == "accuracy"
    assert metrics[1]["value"][0] == 0.8
    assert metrics[1]["stage"][0] == "train"

    # dict -> `log_params`
    logger.log(params)
    # * exclude the Skáld run params
    assert {
        k: v for k, v in logger._params.items() if not k.startswith("skald.")
    } == flatten_dict(params)
    assert len(logger._metrics) == 2

    # str, Figure -> `log_figure`
    fig = plt.figure()
    plt.plot([1, 2, 3])
    logger.log("fig_1", fig)
    assert (logger.artifacts_dir / "fig_1.png").is_file()

    # str -> `info`
    message = "this will be printed"
    logger.log(message)
    assert message in (logger.run_dir / "console.log").open("r").read()
