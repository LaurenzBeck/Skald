<p align="center">
    <img src="https://github.com/LaurenzBeck/Skald/blob/main/docs/images/fantasy_fjord_art.jpg?raw=true"/></a>
</p>

<h1 align="center">
    Skáld
</h1>

<p align="center">
📟 a simple and efficient experiment logger for Python 🐍
</p>

<p align="center">
    <a href="https://pypi.org/project/skald/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/skald?label=%F0%9F%93%A6%20PyPi"></a>
    <a href="https://www.repostatus.org/#wip"><img src="https://www.repostatus.org/badges/latest/wip.svg" alt="Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public." /></a>
    <a href="https://github.com/LaurenzBeck/Skald/actions/workflows/python-package.yaml"><img alt="🐍 Python package" src="https://github.com/LaurenzBeck/Skald/actions/workflows/python-package.yaml/badge.svg"></a>
</p>

<p align="center">
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11|3.12-yellow?logo=python"></a>
    <a href="https://python-poetry.org/"><img alt="Poetry" src="https://img.shields.io/badge/Poetry-1.8.2-blue?logo=Poetry"></a>
</p>

<p align="center">
    <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
    <a href="https://github.com/beartype/beartype"><img alt="Beartype" src="https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg"></a>
</p>

> Skáld - An Old Norse word for a poet, usually applied to a Norwegian or Icelandic court poet or bard of the period from the 9th century to the 13th. Skaldic verse is marked by its elaborate patterns of metre, rhyme, and alliteration, and by its use of kennings.

---

## 📃 Table of Contents

- [💡 Motivation](#-motivation)
- [👀 Concepts](#-installation)
  - [📂 Logging Structure](#-logging-structure)
- [📦 Installation](#-installation)
- [🚀 Usage](#-installation)

---

## 💡 Motivation

During my PhD, I tried different Experiment/Metrics loggers including:

- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [MLFlow](https://mlflow.org/#core-concepts)
- [Weights and Biases](https://wandb.ai/site)
- [DVCLive](https://dvc.org/doc/dvclive)
- other logging solutions from deep learning and continual learning frameworks (mostly stdout and csv loggers)

While those are quite mature logging solutions, which often offer beautiful dashboards, I was looking for something light-weight, local and file-based as DVCLive, but with a more ergonomic and [tidy](https://vita.had.co.nz/papers/tidy-data.pdf) [^1] structure of the logs for simpler consumption and analysis.

The latest of my workflows before Skáld involved using a mixture of DVCLive and a custom logger from [FACIL](https://github.com/mmasana/FACIL/tree/master) [^2] in combination with log crawling CLI scripts and analysis and visualizations performed in jupyter notebooks 🤨

Another problem I faced with those solutions is that all of them offered a single "step" identifier for each logged metric, which is not sufficient for deep learning use-cases outside the conventional epoch-based training loops.

Because I like building python packages and I felt the need to tidy my experiment logs, I created `skald` as a small side project. I hope that some people find some enjoyment in using the package too ❤️

## 👀 Concepts

Skáld is an **experiment logger**, that offers a standardized logging structure and interface to log different aspects of an experiment including:

- **parameters**|**arguments** - meta information and variables, that don't change during the experiment
- **metrics**|**scalars** - single-valued, numeric variables that change during the experiment
- **artifacts** - additional result files like images or plots

Each metric has a unique **name** and a user-defined set of **id variables**, that identify the value of the metric at a certain step.

While a stateful version of Skáld is planned, that updates an id/step variables through some manual call (often in a Callback). The first version is very explicit and requires every id to be logged in each call.

### 📂 Logging Structure

Logs of metrics will be represented by tidy dataframes that are stored as readable `metrics.csv` or more space efficient `metrics.parquet` files.

To save space, parameters will not be included in these dataframes, but in a separate file (`params.yaml`) by default.

Artifacts will be stored in a separate sub-directory (`artifacts/` by default).

Skáld has its own [loguru](https://loguru.readthedocs.io/en/stable/index.html) logger instance and exposes the logging functions.
The logs will stored in a `console.log` file.
If you use the logger as a context manager, *stdout* (~ `print` statements) will also be saved in `console.log`.

## 📦 Installation

The package can be installed with:

```sh
pip install skald
```

> 🧑‍💻 to install a development environment (which you need if you want to work on the package, instead of just using the package), `cd` into the project's root directory and call:

```bash
poetry install --sync --compile
```

## 🚀 Usage

The API of Skáld is very similar to DVCLive and other loggers.
It also exposes [loguru's](https://loguru.readthedocs.io/en/stable/index.html) logging api, so you can also use a Skáld logger as your terminal logger.

A basic example:

```python
from skald import Logger

# get experiment parameters from CLI arguments, parameter files ...
params: dict = { ... }

# instanciate a logger with a certain run name
with Logger("test-run-1") as logger:
    logger.log_params(params)
    # experiment logic
    metric: float = evaluate(model)
    logger.log_metric("accuracy", metric)
    logger.success("finished experiment!")
```

To launch the experiment viewer TUI after the run is completed, use `tui=True` in the constructor of the `Logger`.

You can also launch the experiment viewer manually to inspect your logs:

```sh
$skald <experiment_run_dir>
```

---

## 🖼️ ©️ Banner Artwork Attribution

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/3.0/88x31.png" /></a><br />The art in the banner of this README is licensed under a [Creative Commons Attribution-NonCommercial-No Derivatives Works 3.0 License](https://creativecommons.org/licenses/by-nc-nd/3.0/). It was made by [th3dutchzombi3](https://www.deviantart.com/th3dutchzombi3). Check out his beautiful artwork ❤️

---

## 📄 References

[^1]: H. Wickham, “Tidy Data,” Journal of Statistical Software, vol. 59, pp. 1–23, Sep. 2014, doi: 10.18637/jss.v059.i10.
[^2]: M. Masana, X. Liu, B. Twardowski, M. Menta, A. D. Bagdanov, and J. van de Weijer, “Class-incremental learning: survey and performance evaluation on image classification.” arXiv, Oct. 11, 2022. doi: 10.48550/arXiv.2010.15277.
