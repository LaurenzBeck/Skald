"""🧪 Tests for `skald.Logger`."""
# ruff: noqa: S101, D103, ANN001, ANN201

import pytest
from pathlib import Path
import beartype
import polars as pl
import yaml

import skald
from skald.enums import MetricsFileFormat
from skald.utils import flatten_dict


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
    tmp_path: Path, metrics: list[dict], metrics_file_format: MetricsFileFormat
):
    """📈 Tests the logging of metrics."""
    logger = skald.Logger(tmp_path, "test", metrics_file_format=metrics_file_format)

    for metric in metrics:
        name = metric.pop("name")
        value = metric.pop("value")
        ids = metric
        logger.log_metric(name, value, **ids)
        logger.log_scalar(name, value, **ids)

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
    tmp_path: Path, multikey_metrics, metrics_file_format: MetricsFileFormat
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
    "⚙️ Tests the logging of parameters."
    logger = skald.Logger(tmp_path, "test")

    flattened_params = flatten_dict(params) | logger._params

    learning_rate = params.pop("learning_rate")

    logger.log_param("learning_rate", learning_rate)
    logger.log_params(params)

    saved_params = yaml.safe_load(logger.params_file.read_text())

    assert flattened_params == saved_params
