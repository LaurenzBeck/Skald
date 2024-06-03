"""# Skáld

📟 a simple and efficient experiment logger for Python 🐍
"""

from datetime import datetime
from typing import Any, Self
import yaml
from pathlib import Path
from beartype import beartype
from loguru import logger
import polars as pl
from polars import DataFrame

from skald.enums import (
    PersistenceStrategy,
    PersistenceStrategyLit,
    MetricsFileFormat,
    MetricsFileFormatLit,
)
from skald.utils import flatten_dict


@beartype
class Logger:
    # 🔎 Logger and Run Details
    dir: Path
    run_name: str
    run_dir: Path
    timestamp: datetime
    metrics_file: Path
    metrics_file_format: MetricsFileFormat
    params_file: Path
    artifacts_dir: Path
    persistence_strategy: PersistenceStrategy

    # 🔢 Internals
    _metrics: DataFrame
    _params: dict[str, Any]

    def __init__(
        self,
        dir: Path = Path("."),
        run_name: str | None = None,
        metrics_file: Path = Path("metrics"),
        params_file: Path = Path("params.yaml"),
        artifacts_dir: Path = Path("artifacts"),
        persistence_strategy: PersistenceStrategy | PersistenceStrategyLit = "eager",
        metrics_file_format: MetricsFileFormat | MetricsFileFormatLit = "parquet",
    ) -> None:
        self.timestamp = datetime.now()
        self.run_name = run_name if run_name else str(self.timestamp)
        self.run_dir = dir / self.run_name

        self.run_dir.mkdir()

        logger.info(f"📂 logging to {self.run_dir}")

        self.persistence_strategy = persistence_strategy

        # logging run meta data in run_dir / params_file
        self._params = {
            "skald.timestamp": str(self.timestamp),
            "skald.run_name": self.run_name,
            "skald.version": "v1",
        }
        self.params_file = self.run_dir / params_file
        self._save_params()

        self._metrics = DataFrame()
        self.metrics_file = self.run_dir / metrics_file.with_suffix(
            f".{metrics_file_format}"
        )
        self.metrics_file_format = metrics_file_format
        self._save_metrics()

        self.artifacts_dir = self.run_dir / artifacts_dir
        self.artifacts_dir.mkdir()

    def log(self):
        raise NotImplementedError

    def get_metrics_with_params(self) -> DataFrame:
        """🐻‍❄️ returns the metrics with params in additional columns.

        Returns:
            DataFrame: tidy metrics + params
        """
        return self._metrics.with_columns(
            [pl.lit(param).alias(name) for param, name in self._params.items()]
        )

    # 📈 Metrics

    def log_metric(self, name: str, value: int | float, **ids: dict) -> None:
        """📈 logs a metric with identifiers given as keyword arguments.

        If `self.persistence_strategy` is `"eager"`, the metric will be saved to disk.

        Args:
            name (str): name of the metric
            value (int | float): value of the metric
            **ids: arbitrary keyword arguments used as metric identifier.
        """
        if isinstance(value, int):
            value = float(value)

        new_entry = DataFrame({"name": name, "value": value} | ids)
        self._metrics = pl.concat([self._metrics, new_entry], how="diagonal")

        match self.persistence_strategy:
            case PersistenceStrategy.EAGER:
                self._save_metrics()

    def log_scalar(self, name: str, value: int | float, **ids: dict) -> None:
        """📈 logs a scalar (metric) with identifiers given as keyword arguments.

        If `self.persistence_strategy` is `"eager"`, the metric will be saved to disk.

        Args:
            name (str): name of the metric
            value (int | float): value of the metric
            **ids: arbitrary keyword arguments used as metric identifier.
        """
        self.log_metric(name, value, **ids)

    def log_metrics(self, metrics: dict[str, int | float], **ids) -> None:
        """📈 logs a dictionary of metrics with identifiers given as keyword arguments.

        If `self.persistence_strategy` is `"eager"`, the metrics will be saved to disk.

        Examples:
            An example of how to log two metrics stored in a dictionary.

            >>> logger.log_metrics({"loss": 1.23, "accuracy": 0.9}, stage="test", step=1)


        Args:
            metrics (dict[str, int | float]): single-level dictionary of metrics.
            **ids: arbitrary keyword arguments used as metric identifier.
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, **ids)

    def log_scalars(self, metrics: dict[str, int | float], **ids) -> None:
        """📈 logs a dictionary of scalars (metrics) with identifiers given as keyword arguments.

        If `self.persistence_strategy` is `"eager"`, the metrics will be saved to disk.

        Examples:
            An example of how to log two metrics stored in a dictionary.

            >>> logger.log_metrics({"loss": 1.23, "accuracy": 0.9}, stage="test", step=1)


        Args:
            metrics (dict[str, int | float]): single-level dictionary of metrics.
            **ids: arbitrary keyword arguments used as metric identifier.
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, **ids)

    def log_dict(self, metrics: dict[str, int | float], **ids) -> None:
        """📈 logs a dictionary of metrics with identifiers given as keyword arguments.

        If `self.persistence_strategy` is `"eager"`, the metrics will be saved to disk.

        Examples:
            An example of how to log two metrics stored in a dictionary.

            >>> logger.log_metrics({"loss": 1.23, "accuracy": 0.9}, stage="test", step=1)


        Args:
            metrics (dict[str, int | float]): single-level dictionary of metrics.
            **ids: arbitrary keyword arguments used as metric identifier.
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, **ids)

    # ⚙️ Parameters

    def log_param(self, name: str, value: Any) -> None:
        """⚙️ logs a single parameter.

        If the parameter exists, it will be overwritten.

        If `self.persistence_strategy` is `"eager"`, the parameter will be saved to disk.

        Args:
            name (str): name of the parameter.
            value (Any): value of the parameter.
                parameters are stored as yaml files, so any yaml-compatible value is allowed.
        """
        self._params[name] = value

        match self.persistence_strategy:
            case PersistenceStrategy.EAGER:
                self._save_params()

    def log_arg(self, name: str, value: Any) -> None:
        """⚙️ logs a single argument (parameter).

        If `self.persistence_strategy` is `"eager"`, the parameter will be saved to disk.

        Args:
            name (str): name of the parameter.
            value (Any): value of the parameter.
                parameters are stored as yaml files, so any yaml-compatible value is allowed.
        """
        self.log_param(name, value)

    def log_params(self, params: dict, separator: str = ".") -> None:
        """⚙️ logs a dictionary of parameters.

        To facilitate the consumption and visualization, the parameters dictionary will be flattened.

        If `self.persistence_strategy` is `"eager"`, the parameters will be saved to disk.

        Args:
            params (dict): (possibly nested) dictionary of parameters.
            separator (str, optional): keys are concatenated using this separator. Defaults to '.'.
        """
        params = flatten_dict(params, sep=separator)

        for param, value in params.items():
            self.log_param(param, value)

    def log_args(self, params: dict, separator: str = ".") -> None:
        """⚙️ logs a dictionary of arguments (parameters).

        To facilitate the consumption and visualization, the parameters dictionary will be flattened.

        If `self.persistence_strategy` is `"eager"`, the parameters will be saved to disk.

        Args:
            params (dict): (possibly nested) dictionary of parameters.
            separator (str, optional): keys are concatenated using this separator. Defaults to '.'.
        """
        self.log_params(params, separator)

    # 🖼️ Artifacts

    def log_figure(self):
        raise NotImplementedError

    def log_image(self):
        raise NotImplementedError

    # 💾 Saving to Disk

    def save(self) -> None:
        """💾 saves metrics and parameters to disk."""
        match self.persistence_strategy:
            case PersistenceStrategy.EAGER:
                logger.info(
                    "There is no need to call `save` when using `PersistenceStrategy.EAGER`."
                )

        self._save_metrics()
        self._save_params()

    def _save_metrics(self) -> None:
        """💾 saves metrics to disk."""
        match self.metrics_file_format:
            case "csv":
                self._metrics.write_csv(self.metrics_file)
            case "parquet":
                self._metrics.write_parquet(self.metrics_file)

    def _save_params(self) -> None:
        """💾 saves parameters to disk."""
        with (self.params_file).open("w") as f:
            yaml.dump(self._params, f)

    def __enter__(self) -> Self:
        """📲 called when entering a context manager."""
        return self

    def __exit__(self, exc_type_, exc_value_, traceback_) -> None:
        """🚪 called when leaving a context manager.

        The parameters and metrics will be saved.
        """
        self.save()
