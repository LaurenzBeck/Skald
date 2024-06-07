"""# Skáld

📟 a simple and efficient experiment logger for Python 🐍
"""

from datetime import datetime
from typing import Any, Self
import yaml
from pathlib import Path
from beartype import beartype
from loguru import logger
from loguru._logger import Logger as LoguruLogger
import polars as pl
from polars import DataFrame
import sys
import contextlib

from skald.enums import (
    PersistenceStrategy,
    PersistenceStrategyLit,
    MetricsFileFormat,
    MetricsFileFormatLit,
)
from skald.utils import flatten_dict
from skald import tui


@beartype
class Logger:
    """📃 an experiment logger for storing console logs, metrics, parameters and artifacts.

    This class integrates an optional [textualize](https://textual.textualize.io/) TUI app for viewing the logs.

    ???+ info

        Skáld also provides a CLI tool (`skald`) to view logs later.
        simply call `$skald <path_to_log_dir>` or `$skald --help` for help.

    Attributes:
        dir (Path): directory, where the log "run" directory will be created.
        run_name (str): name of the experiment/run. Can be defined by the user.
            If not defined, the current timestamp will be used.
        run_dir (Path): path to the run directory. Will be `dir/{run_name}`.
        timestamp (datetime): timestamp of the creation of the logger instance.
        metrics_file (Path): path to the metrics file.
        metrics_file_format (MetricsFileFormat): file format of the metrics.
            Currently, csv and parquet are supported. Defaults to "parquet".
        params_file (Path): path to the parameters file.
        artifacts_dir (Path): path to the artifacts directory.
        persistence_strategy (PersistenceStrategy): when to save the metrics to disk.
            Used for performance optimizations. Defaults to "eager".
        tui (bool): Whether to start the experiment viewer terminal user interface.
    """

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
    tui: bool

    # 🔢 Internals
    _logger: LoguruLogger
    _metrics: DataFrame
    _params: dict[str, Any]

    def __init__(
        self,
        dir: Path = Path("."),
        run_name: str | None = None,
        persistence_strategy: PersistenceStrategy | PersistenceStrategyLit = "eager",
        metrics_file_format: MetricsFileFormat | MetricsFileFormatLit = "parquet",
        tui: bool = False,
    ):
        """🏃‍♂️ creates a logger instance and prepares a Skáld run.

        Each run will have the following structure:

        - `dir/{run_name}/`
            - `console.log`
            - `metrics.{metrics_file_format}`
            - `params.yaml`
            - `artifacts/`

        Args:
            dir (Path, optional): path to the directory under which the run directory will be created. Defaults to Path(".").
            run_name (str | None, optional): name of the run. If None, use the current time as `run_name`. Defaults to None.
            persistence_strategy (PersistenceStrategy | PersistenceStrategyLit, optional): when to save the metrics to disk.
                Used for performance optimizations. Defaults to "eager".
            metrics_file_format (MetricsFileFormat | MetricsFileFormatLit, optional): file format of the metrics. Defaults to "parquet".
            tui (bool, optional): Whether to start the experiment viewer terminal user interface. Defaults to false.
        """
        self.timestamp = datetime.now()
        self.run_name = (
            run_name if run_name else self.timestamp.strftime("%Y%m%d-%H%M%S")
        )
        self.run_dir = dir / self.run_name

        self.run_dir.mkdir()

        # 📃 console logging with loguru
        # * we also redirect stdout to the log file, which requires a trick from:
        # * https://loguru.readthedocs.io/en/stable/resources/recipes.html#capturing-standard-stdout-stderr-and-warnings
        self._logger = logger

        class _StreamToLogger:
            def __init__(self, _logger):
                self._logger = _logger

            def write(self, buffer):
                self._logger.opt(depth=1, raw=True).info(buffer)

            def flush(self):
                pass

        self._logger.remove()
        self._logger.add(sys.__stdout__)
        self._logger.add(f"{self.run_dir/'console.log'}")
        self._redirect_stdout = contextlib.redirect_stdout(
            _StreamToLogger(self._logger)
        )

        self.info(f"📂 logging to {self.run_dir}")

        self.persistence_strategy = persistence_strategy

        # logging run meta data in run_dir / params_file
        self._params = {
            "skald.timestamp": str(self.timestamp),
            "skald.run_name": self.run_name,
            "skald.version": "v1",
        }
        self.params_file = self.run_dir / Path("params.yaml")
        self._save_params()

        self._metrics = DataFrame()
        self.metrics_file = self.run_dir / Path("metrics").with_suffix(
            f".{metrics_file_format}"
        )
        self.metrics_file_format = metrics_file_format
        self._save_metrics()

        self.artifacts_dir = self.run_dir / Path("artifacts")
        self.artifacts_dir.mkdir()

        self.tui = tui

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
                self.info(
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
        self._redirect_stdout.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """🚪 called when leaving a context manager.

        The parameters and metrics will be saved.
        """
        self.save()
        self._redirect_stdout.__exit__(exc_type, exc_value, traceback)
        # * because the tui is not yet fully featured (metrics are not updating),
        # * we only start the tui after the log has finished to inspect the logs.
        if self.tui:
            tui.view_experiment(self.run_dir)

    # 📃 Console Logging

    def debug(self, message: str, *args, **kwargs) -> None:
        """🔎 logs a debug message using [loguru](https://loguru.readthedocs.io/en/stable/)."""
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """ℹ️ logs an info message using [loguru](https://loguru.readthedocs.io/en/stable/)."""
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """❗ logs a warning message using [loguru](https://loguru.readthedocs.io/en/stable/)."""
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """🔥 logs n error message using [loguru](https://loguru.readthedocs.io/en/stable/)."""
        self._logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """🤯 logs a critical message using [loguru](https://loguru.readthedocs.io/en/stable/)."""
        self._logger.critical(message, *args, **kwargs)

    def success(self, message: str, *args, **kwargs) -> None:
        """✅ logs a success message using [loguru](https://loguru.readthedocs.io/en/stable/)."""
        self._logger.success(message, *args, **kwargs)
