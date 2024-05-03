"""# Skáld

📟 a simple and efficient experiment logger for Python 🐍
"""

from pathlib import Path
import beartype
from loguru import logger


@beartype
class Logger:
    dir: Path
    meta_file: Path
    metrics_file: Path
    params_file: Path | None
    artifacts_dir: Path

    def __init__(
        self,
        dir: Path = Path("."),
        meta_file: Path = Path("meta.yaml"),
        metrics_file: Path = Path("metrics.parquet"),
        params_file: Path | None = None,
        artifacts_dir: Path = Path("artifacts"),
    ) -> None:
        logger.info(f"📂 logging to {dir}")
