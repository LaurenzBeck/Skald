"""🧑‍💻 command line interfaces."""

from pathlib import Path
from typing import Annotated, List, Optional

import typer
from loguru import logger

from skald import combine_runs, tui

app = typer.Typer(help="📟 a simple and efficient experiment logger for Python 🐍")


@app.command()
def view(
    experiment_dir: Path,
    update_interval: Annotated[Optional[float], typer.Option()] = None,
) -> None:
    """🔎⚗️ view a Skáld experiment in `experiment_dir` with a [textualize](https://textual.textualize.io/) app.

    Args:
        experiment_dir (Path): directory containing logs of a Skáld run
        update_interval (float | None, optional): time in seconds to update the widgets
            by reading the files again. Defaults to None
    """  # noqa: E501
    tui_app = tui.ExperimentViewerApp(experiment_dir, update_interval)
    tui_app.run()


@app.command()
def combine(
    directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="📂 directory to search for skald runs.",
        ),
    ],
    output_file_path: Annotated[
        Path,
        typer.Argument(
            file_okay=True,
            writable=True,
            help="📄 file path where the output will be saved to.",
        ),
    ],
    include_params: Annotated[
        Optional[List[str]],
        typer.Option(
            "--param",
            "-p",
            help="⚙️ list of parameters to include in the result for each run.",
        ),
    ] = None,
) -> None:
    """🔎📂 crawls a `directory` recursively for runs and saves the result`.

    Every parameter in `include_params` will be read from each run (if present) and
    added to the saved dataframe as additional columns.
    """
    results = combine_runs(directory, include_params)

    match output_file_path.suffix:
        case ".csv":
            results.write_csv(output_file_path)
        case ".parquet":
            results.write_parquet(output_file_path)

    logger.success(f"💾 saved result to {output_file_path}.")


def main() -> None:
    """🚪 typer entry point."""
    app()


if __name__ == "__main__":
    main()
