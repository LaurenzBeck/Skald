"""🔢 all enumerations for `skald`.

For every enumeration, a Literal with the same values as the enumeration is created.
In every function that excepts a custom `skald` enum as an argument, use the type
<enum_name> | <enum_name>Lit to have strong type checking, string autocompletion and
good documentation generation.
"""

from enum import Enum
from typing import Literal, TypeAlias


class PersistenceStrategy(str, Enum):
    """💾 strategy which defines when metrics and parameters are persisted in a file."""

    EAGER = "eager"
    """Persists the metrics and parameters immediately after every `log` call."""
    LAZY = "lazy"
    """Only persists metrics and parameters when a user calls `save`
    or the context manager is exited."""


PersistenceStrategyLit: TypeAlias = Literal[
    "eager",
    "lazy",
]


class MetricsFileFormat(str, Enum):
    """📄 the file format the metrics will be stored as."""

    CSV = "csv"
    """Human readable, but the logs take more space to store."""
    PARQUET = "parquet"
    """Efficient, binary format."""


MetricsFileFormatLit: TypeAlias = Literal[
    "csv",
    "parquet",
]
