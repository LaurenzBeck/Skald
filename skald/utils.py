"""🛠️ utility functions for `skald`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


def flatten_dict(d: dict, sep: str = ".") -> dict:
    """📖 flattens a nested dictionary by concatenating keys with <sep>.

    Code from https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/

    Args:
        d (dict): dictionary
        sep (str, optional): keys are concatenated using this separator. Defaults to '.'

    Returns:
        dict: flattened dictionary
    """
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    return flat_dict


def is_skald_run(directory: Path) -> bool:
    """❓ checks whether a `directory` is (likely) a skald run.

    Args:
        directory (Path): 📂

    Returns:
        bool: true if `directory` is (likely) a skald run.
    """
    return (
        directory.is_dir()
        and (directory / "console.log").is_file()
        and (directory / "params.yaml").is_file()
        and (
            (directory / "metrics.parquet").is_file()
            or (directory / "metrics.csv").is_file()
        )
        and (directory / "artifacts").is_dir()
    )


def get_skald_runs(directory: Path) -> list[Path]:
    """🔎📂 recursively traverses a `directory` and returns all skald runs.

    Args:
        directory (Path): 📂

    Returns:
        list[Path]: list of paths to skald runs.
    """
    return [d for d in directory.rglob("**/") if is_skald_run(d)]
