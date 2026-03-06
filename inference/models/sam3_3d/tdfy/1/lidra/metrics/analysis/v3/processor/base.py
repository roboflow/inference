"""Base processor class defining the analysis pipeline structure."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any, Optional
from dataclasses import dataclass
import pandas as pd
import os
from loguru import logger

from .logging import json_logging


ConfigT = TypeVar("ConfigT")
DataT = TypeVar("DataT")
ResultsT = TypeVar("ResultsT")


@dataclass
class BaseConfig:
    """Base configuration for all processors."""

    # Input/Output
    input_file: str
    output_directory: Optional[str] = None  # If specified, always saves as CSV
    console_format: str = "rich"  # "rich" or "csv" - only used for console output

    # Output options
    verbose: bool = False  # Print processing statistics
    quiet: bool = False  # Suppress all console output (files are still saved)
    format_precision: int = 6  # Number of decimal places for formatting
    extract_data_log_file: str = (
        "extract_data.json"  # Filename for JSON logs during data extraction
    )


class Processor(ABC, Generic[ConfigT, DataT, ResultsT]):
    """Base processor defining the analysis pipeline structure.

    Template method pattern: the pipeline is fixed but individual
    steps can be customized by subclasses.
    """

    def run(self, config: ConfigT) -> None:
        """Main processing pipeline - template method.

        This method defines the skeleton of the algorithm and
        delegates specific steps to subclasses.
        """
        # Display header unless quiet
        if not (hasattr(config, "quiet") and config.quiet):
            self._display_header()

        # 1. Load raw data as DataFrame
        raw_data = self.load_data(config)

        # 2. Extract relevant data into internal format (with optional JSON logging)
        data = self.extract_data(raw_data, config)

        # 3. Apply transformations
        transformed_data = self.transform_data(data, config)

        # 4. Compute results/statistics
        results = self.compute_results(transformed_data, config)

        # 5. Handle output
        self.handle_output(results, config)

    # Default implementations for common operations
    def load_data(self, config: ConfigT) -> pd.DataFrame:
        """Load data from source. Default assumes CSV file.

        Returns a pandas DataFrame. Subclasses can override to load
        from other sources but must return a DataFrame.
        """
        if hasattr(config, "input_file") and config.input_file is not None:
            return pd.read_csv(config.input_file)
        else:
            raise NotImplementedError(
                "Subclass must implement load_data or provide input_file in config"
            )

    def _display_header(self) -> None:
        """Display the Lidra analysis tool header."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from rich import box

        console = Console()

        # Create styled header text
        header_text = Text("✨ LIDRA Metrics Analysis Tool ✨", style="bold steel_blue")

        panel = Panel.fit(
            header_text, border_style="steel_blue", box=box.DOUBLE_EDGE, padding=(0, 2)
        )
        console.print()
        console.print(panel, justify="center")
        console.print()

    def _get_log_file_path(self, config: ConfigT) -> Optional[str]:
        """Get the full path for the log file based on config."""
        if hasattr(config, "extract_data_log_file") and hasattr(
            config, "output_directory"
        ):
            if config.output_directory:
                os.makedirs(config.output_directory, exist_ok=True)
                return os.path.join(
                    config.output_directory, config.extract_data_log_file
                )
            else:
                return config.extract_data_log_file
        return None

    def extract_data(self, raw_data: pd.DataFrame, config: ConfigT) -> DataT:
        """Extract data with optional JSON logging. Do not override this method."""
        log_file_path = self._get_log_file_path(config)

        if log_file_path:
            with json_logging(log_file_path):
                return self._extract_data(raw_data, config)
        else:
            return self._extract_data(raw_data, config)

    def handle_output(self, results: ResultsT, config: ConfigT) -> None:
        if hasattr(config, "output_directory") and config.output_directory:
            # Save CSV to file
            csv_output = self.get_output(results, config, "csv")
            self._write_to_file(csv_output, config)

        # Also display to console unless quiet
        if not (hasattr(config, "quiet") and config.quiet):
            self._write_to_console(results, config)

    def _write_to_file(self, output: Any, config: ConfigT) -> None:
        """Write CSV output to file."""
        os.makedirs(config.output_directory, exist_ok=True)
        output_path = os.path.join(config.output_directory, "results.csv")

        # Convert to string if not already
        if not isinstance(output, str):
            # This shouldn't happen for CSV format, but handle it gracefully
            output = str(output)

        logger.info(f"Writing results to: {output_path}")
        with open(output_path, "w") as f:
            f.write(output)
        logger.info(f"Successfully saved results to: {output_path}")

    def _write_to_console(self, results: ResultsT, config: ConfigT) -> None:
        from rich.console import Console

        console_fmt = getattr(config, "console_format", "rich")
        console_output = self.get_output(results, config, console_fmt)
        console = Console()
        # If console_output is a Rich object (like Table), print it directly
        # Otherwise, print it as a string
        console.print()
        console.print(console_output)
        console.print()

    # Abstract methods that subclasses must implement
    @abstractmethod
    def _extract_data(self, raw_data: pd.DataFrame, config: ConfigT) -> DataT:
        """Extract relevant data from raw DataFrame into internal format.

        Subclasses should override this method, not extract_data.
        """
        pass

    @abstractmethod
    def transform_data(self, data: DataT, config: ConfigT) -> DataT:
        """Apply transformations to the data."""
        pass

    @abstractmethod
    def compute_results(self, data: DataT, config: ConfigT) -> ResultsT:
        """Compute statistics or results from transformed data."""
        pass

    @abstractmethod
    def get_output(self, results: ResultsT, config: ConfigT, format: str) -> str:
        """Get formatted output in specified format.

        Args:
            results: Computed results
            config: Configuration
            format: Output format - 'csv' or 'rich'

        Returns:
            Formatted string
        """
        pass
