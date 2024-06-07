"""🧑‍💻 terminal user interfaces"""

from pathlib import Path
from typing import Optional

import polars as pl
import typer
from typing_extensions import Annotated
from textual.app import App, ComposeResult
from textual.widgets import (
    Header,
    TabbedContent,
    TabPane,
    RichLog,
    Pretty,
    DirectoryTree,
    Static,
)
from textual_plotext import PlotextPlot
import yaml
from aiofile import async_open

from textual.containers import ScrollableContainer


class LogsWidget(RichLog):
    """📃 a widget, that periodically reads `logs_file` and displays the result."""

    def __init__(self, logs_file: Path, update_interval: float | None = None):
        self.logs_file = logs_file
        self.update_interval = update_interval
        super().__init__()

    async def on_mount(self) -> None:
        await self.read_logs()
        if self.update_interval:
            self.update_timer = self.set_interval(self.update_interval, self.read_logs)

    async def read_logs(self) -> None:
        async with async_open(self.logs_file, "r") as f:
            logs = await f.read()

        self.clear()
        self.write(logs)


class MetricsWidget(PlotextPlot):
    """📉 a static widget, that plots a metric using [plottext](https://github.com/Textualize/textual-plotext)."""

    def __init__(self, name: str, values: list[float], ids: list, id: str):
        self.metric_name = name
        self.metric_values = values
        self.metric_ids = ids
        super().__init__(id=id)

    def on_mount(self) -> None:
        self.plt.plot(self.metric_values, self.metric_ids)
        self.plt.title(self.metric_name)


class StaticMetricsListWidget(Static):
    """🔎📈 a static widget, that reads a `metrics_file` and plots each metric."""

    def __init__(self, metrics_file: Path):
        self.metrics_file = metrics_file
        super().__init__()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        self.read_metrics()

        plots = self.get_plots()

        yield ScrollableContainer(*plots, id="plots")

    def get_plots(self) -> list[MetricsWidget]:
        plots = []
        # parse metrics and decide what to plot
        for (name,), df in self.metrics.groupby(["name"]):
            # get values
            values = df["value"].to_list()
            # get ids
            # * in this first prototype, we just arange the values
            ids = range(len(values))

            plots.append(MetricsWidget(name, values, ids, id=f"{name}-plot"))

        return plots

    async def read_metrics_async(self) -> None:
        async with async_open(self.metrics_file, "rb") as f:
            metrics = await f.read()
        match self.metrics_file.suffix:
            case ".csv":
                self.metrics = pl.read_csv(metrics)
            case ".parquet":
                self.metrics = pl.read_parquet(metrics)

    def read_metrics(self) -> None:
        match self.metrics_file.suffix:
            case ".csv":
                self.metrics = pl.read_csv(self.metrics_file)
            case ".parquet":
                self.metrics = pl.read_parquet(self.metrics_file)


class ParametersWidget(Pretty):
    """⚙️ a widget, that periodically reads `params_file` and displays the result."""

    def __init__(self, params_file: Path, update_interval: float | None = None):
        self.params_file = params_file
        self.update_interval = update_interval
        super().__init__(None)

    async def on_mount(self) -> None:
        await self.read_params()
        if self.update_interval:
            self.update_timer = self.set_interval(
                self.update_interval, self.read_params
            )

    async def read_params(self) -> None:
        async with async_open(self.params_file, "r") as f:
            file_content = await f.read()
        params = yaml.safe_load(file_content)
        self.update(params)


class ArtifactsWidget(DirectoryTree):
    """📂 a simple directory tree widget that updates periodically."""

    def __init__(self, artifacts_dir: Path, update_interval: float | None = None):
        self.artifacts_dir = artifacts_dir
        self.update_interval = update_interval
        super().__init__(artifacts_dir)

    def on_mount(self) -> None:
        if self.update_interval:
            self.update_timer = self.set_interval(self.update_interval, self.reload)


class ExperimentViewerApp(App):
    """🔎⚗️ a Textual app to view Skáld experiments."""

    TITLE = "🎶 Skáld"
    SUB_TITLE = "an experiment viewer"

    BINDINGS = [
        ("l", "switch_tab('tab-1')", "📃 view logs"),
        ("m", "switch_tab('tab-2')", "📈 view metrics"),
        ("p", "switch_tab('tab-3')", "⚙️ view parameters"),
        ("a", "switch_tab('tab-4')", "📂 view artifacts"),
        ("d", "toggle_dark", "🌒 toggle dark mode"),
    ]

    CSS_PATH = "tui.tcss"

    def __init__(self, experiment_dir: Path, update_interval: float | None = None):
        """Creates an experiment viewer for a Skáld experiment in `experiment_dir`.

        Args:
            experiment_dir (Path): directory containing logs of a Skáld run
            update_interval (float | None, optional): time in seconds to update the widgets
                by reading the files again. Defaults to None
        """
        self.experiment_dir = experiment_dir
        self.update_interval = update_interval
        super().__init__()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        with TabbedContent():
            with TabPane("📃 logs"):
                yield LogsWidget(
                    self.experiment_dir / "console.log", self.update_interval
                )
            with TabPane("📈 metrics"):
                yield StaticMetricsListWidget(
                    next(self.experiment_dir.glob("metrics.*"))
                )
            with TabPane("⚙️ params"):
                yield ParametersWidget(
                    self.experiment_dir / "params.yaml", self.update_interval
                )
            with TabPane("📂 artifacts"):
                yield ArtifactsWidget(
                    self.experiment_dir / "artifacts", self.update_interval
                )
        # yield Footer() #* the footer seems buggy in dark mode -> duplicated footers, missing lines and scrolling issues...

    def action_switch_tab(self, tab: str) -> None:
        """Change the active tab.

        Args:
            tab (str): new active tab
        """
        self.query_one(TabbedContent).active = tab

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark


def view_experiment(
    experiment_dir: Path,
    update_interval: Annotated[Optional[float], typer.Option()] = None,
):
    """🔎⚗️ view a Skáld experiment in `experiment_dir` with a [textualize](https://textual.textualize.io/) app.

    Args:
        experiment_dir (Path): directory containing logs of a Skáld run
        update_interval (float | None, optional): time in seconds to update the widgets
            by reading the files again. Defaults to None
    """
    app = ExperimentViewerApp(experiment_dir, update_interval)
    app.run()


def main():
    typer.run(view_experiment)


if __name__ == "__main__":
    main()
