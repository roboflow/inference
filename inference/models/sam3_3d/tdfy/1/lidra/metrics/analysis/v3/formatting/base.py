"""Abstract base class for formatters."""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
import pandas as pd

ConfigT = TypeVar("ConfigT")


class Formatter(ABC, Generic[ConfigT]):
    """Abstract base class for all formatters.

    Formatters are responsible for converting processed results into
    human-readable output formats (CSV, Rich tables, etc).
    """

    @abstractmethod
    def format(self, results: pd.DataFrame, config: ConfigT) -> Any:
        """Format results according to configuration.

        Args:
            results: The results to format (typically a DataFrame)
            config: Configuration object with formatting options

        Returns:
            Formatted output (str for CSV, Table for Rich, etc.)
        """
        pass
