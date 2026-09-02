"""# ⏱️ Benchmarks."""

from __future__ import annotations

import os
import shutil
import tomllib
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from random import random
from time import perf_counter
from typing import TYPE_CHECKING

import polars as pl
import psutil
import typer

from skald import Logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from skald.enums import MetricsFileFormatLit

ENTRY_COUNTS = (100, 1_000, 10_000, 100_000)


def project_version() -> str:
    """Read the package version from the source metadata."""
    with Path("pyproject.toml").open("rb") as pyproject:
        metadata = tomllib.load(pyproject)
    return metadata["project"]["version"]


def run_once(task: Callable, results) -> None:  # noqa: ANN001
    """Run one timed workload and return its runtime to the parent process."""
    start = perf_counter()
    task()
    results.put(perf_counter() - start)


def measure(task: Callable, number: int) -> tuple[float, float]:
    """Measure mean runtime and peak RSS in isolated processes."""
    context = get_context("spawn")
    runtimes = []
    peak_rss = 0

    for _ in range(number):
        results = context.Queue()
        benchmark_process = context.Process(target=run_once, args=(task, results))
        benchmark_process.start()
        child_process = psutil.Process(benchmark_process.pid)

        while benchmark_process.is_alive():
            try:
                peak_rss = max(peak_rss, child_process.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            benchmark_process.join(0.001)

        benchmark_process.join()
        if benchmark_process.exitcode != 0:
            msg = f"benchmark process exited with code {benchmark_process.exitcode}"
            raise RuntimeError(msg)
        runtimes.append(results.get(timeout=1))
        results.close()

    return sum(runtimes) / number, peak_rss / 1024**2


def run(
    dir: Path,
    file_format: MetricsFileFormatLit,
    num_samples: int,
    method: str,
) -> None:
    """Log and persist one lazy metric workload."""
    with Logger(
        dir,
        run_name=f"{file_format}{num_samples}-{random()}",  # noqa: S311
        metrics_file_format=file_format,
        persistence_strategy="lazy",
    ) as logger:
        match method:
            case "scalar":
                for step in range(num_samples):
                    logger.log_metric("metric", random(), step=step)  # noqa: S311
            case "dictionary":
                metrics = {
                    f"metric_{index}": random()  # noqa: S311
                    for index in range(100)
                }
                for step in range(num_samples // len(metrics)):
                    logger.log_metrics(metrics, step=step)


def benchmark(num_iterations: int = 3) -> None:  # noqa: D103
    skald_version = project_version()
    data = []
    temp_dir = Path("./tmp/benchmarks")
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.mkdir(temp_dir)  # noqa: PTH102
    for file_format in ["csv", "parquet"]:
        for method in ["scalar", "dictionary"]:
            for num_samples in ENTRY_COUNTS:
                task = partial(run, temp_dir, file_format, num_samples, method)
                runtime, peak_rss = measure(task, number=num_iterations)
                data.append(
                    {
                        "version": skald_version,
                        "format": file_format,
                        "persistence_strategy": "lazy",
                        "method": method,
                        "num_iterations": num_iterations,
                        "num_samples": num_samples,
                        "runtime": runtime,
                        "peak_rss_mib": peak_rss,
                    },
                )

    output_file = Path(__file__).with_name("benchmarks.csv")
    previous_results = pl.read_csv(
        output_file,
        schema_overrides={"peak_rss_mib": pl.Float64},
    )
    previous_results = previous_results.filter(pl.col("version") != skald_version)
    results = pl.concat(
        [previous_results, pl.DataFrame(data)],
        how="diagonal_relaxed",
    ).select(
        "version",
        "format",
        "persistence_strategy",
        "method",
        "num_iterations",
        "num_samples",
        "runtime",
        "peak_rss_mib",
    )
    results.write_csv(output_file)
    print(results.filter(pl.col("version") == skald_version))  # noqa: T201


if __name__ == "__main__":
    typer.run(benchmark)
