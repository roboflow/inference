"""Base console output class for metrics analysis."""

from typing import Dict, Any, Union
import argparse
import pandas as pd


class ConsoleOutput:
    """Handles console output formatting for metrics analysis results."""

    def __init__(
        self,
        results: Union[Dict[str, pd.DataFrame], pd.DataFrame],
        metadata: Dict[str, Any],
        args: argparse.Namespace,
    ):
        self.results = results
        self.metadata = metadata
        self.args = args
        self._is_combined = isinstance(results, pd.DataFrame)

    def print(self) -> None:
        """Main entry point - prints formatted output based on args.format."""
        print(self._ascii_banner())
        self._print_title()
        self._print_header()
        self._print_tables_format()
        self._print_footer_notes()

    def _ascii_banner(self) -> str:
        """Create ASCII art banner for LIDRA."""
        banner = [
            "╔══════════════════════════════════════════════════════════╗",
            "║  ██╗     ██╗██████╗ ██████╗  █████╗                      ║",
            "║  ██║     ██║██╔══██╗██╔══██╗██╔══██╗                     ║",
            "║  ██║     ██║██║  ██║██████╔╝███████║                     ║",
            "║  ██║     ██║██║  ██║██╔══██╗██╔══██║                     ║",
            "║  ███████╗██║██████╔╝██║  ██║██║  ██║                     ║",
            "║  ╚══════╝╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝                     ║",
            "║                   Metrics Analysis Results               ║",
            "╚══════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(banner)

    def _print_title(self) -> None:
        print(self._title_text())

    def _title_text(self) -> str:
        """Create header text for LIDRA."""
        return "✨ LIDRA Metrics Analysis Tool ✨"

    def _print_header(self) -> None:
        """Print common header information."""
        print(self._header_text())

    def _header_text(self) -> str:
        header_text = [f"Input file: {self.args.input}"]
        if self._is_combined:
            header_text.append("Format: Single-file (combined tables)")
        else:
            header_text.append(f"Format: {self.args.format} | Mode: {self.args.mode}")
        return "\n".join(header_text)

    def _print_tables_format(self) -> None:
        """Print results in separate tables format."""
        report_config = self.metadata.get("report_config", {}) if self.metadata else {}

        for table_name, df in self.results.items():
            print(f"\n--- {table_name} ---")
            print(df.to_string(index=True, float_format="%.6f"))

    def _print_footer_notes(self) -> None:
        """Print footer notes based on analysis mode."""
        if not self.args.verbose:
            return
        print(self._default_footer_notes_text())

    def _default_footer_notes_text(self) -> str:
        return "This prints when verbose is enabled"
