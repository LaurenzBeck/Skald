"""# ⏱️ Benchmarks."""

import os
import shutil
from functools import partial
from pathlib import Path
from random import random
from time import time
from typing import Callable

import pkg_resources
import toml
import typer
from polars import DataFrame

from skald import Logger
from skald.enums import MetricsFileFormatLit


def timeit(task: Callable, number: int) -> float:  # noqa: D103
    times = []
    for _ in range(number):
        t_start = time()
        task()
        times.append(time() - t_start)
    return sum(times) / number


def read_version_from_pyproject_toml(file_path: str = "pyproject.toml") -> str:
    """Reads the version field from pyproject.toml.

    Args:
        file_path (str, optional): Path to pyproject.toml file. Defaults to 'pyproject.toml'.

    Returns:
        str: The version string from pyproject.toml.
    """  # noqa: E501
    with open(file_path) as f:  # noqa: PTH123
        return toml.load(f).get("tool", {}).get("poetry", {}).get("version")


# Tasks


def empty_run(dir: Path, file_format: MetricsFileFormatLit) -> None:  # noqa: D103
    with Logger(
        dir,
        run_name=f"{file_format}0-{random()}",  # noqa: S311
        metrics_file_format=file_format,
    ):
        pass


def run(dir: Path, file_format: MetricsFileFormatLit, num_samples: int) -> None:  # noqa: D103
    with Logger(
        dir,
        run_name=f"{file_format}{num_samples}-{random()}",  # noqa: S311
        metrics_file_format=file_format,
    ) as logger:
        for _ in range(num_samples):
            logger.log_metric("metric", random())  # noqa: S311


def benchmark(num_iterations: int = 1) -> None:  # noqa: D103
    version = pkg_resources.get_distribution("skald").version
    # version = read_version_from_pyproject_toml("pyproject.toml")  # noqa: ERA001
    data = []
    temp_dir = Path("./tmp/benchmarks")
    shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)  # noqa: PTH102
    for file_format in ["csv", "parquet"]:
        task = partial(empty_run, temp_dir, file_format)
        data.append(
            {
                "version": version,
                "format": file_format,
                "num_iterations": num_iterations,
                "num_samples": 0,
                "runtime": timeit(task, number=num_iterations) / num_iterations,
            },
        )
        for num_samples in [10, 100, 1000, 10000]:
            task = partial(run, temp_dir, file_format, num_samples)
            data.append(
                {
                    "version": version,
                    "format": file_format,
                    "num_iterations": num_iterations,
                    "num_samples": num_samples,
                    "runtime": timeit(task, number=num_iterations) / num_iterations,
                },
            )

    df = DataFrame(data)  # noqa: PD901
    df.write_csv(temp_dir / "benchmarks.csv")
    print(df)  # noqa: T201


if __name__ == "__main__":
    typer.run(benchmark)
