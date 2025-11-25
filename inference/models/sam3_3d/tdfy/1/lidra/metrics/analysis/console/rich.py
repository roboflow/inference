"""Rich library implementation for enhanced terminal output."""

from typing import Dict, Any, Union, List, Optional
import pandas as pd
from .base import ConsoleOutput

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich.columns import Columns
    from rich.padding import Padding
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from loguru import logger


class ValueFormatter:
    """Handles formatting of different value types."""

    def __init__(self, precision: int = 6):
        self.precision = precision

    def format(self, value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, float):
            return f"{value:.{self.precision}f}"
        return str(value)


class TableStyler:
    """Encapsulates table styling configuration."""

    def __init__(self, theme: str = "ocean"):
        self.theme = theme
        self._themes = {
            "ocean": {
                "title_style": "bold pale_turquoise1",
                "border_style": "steel_blue",
                "header_style": "bold pale_turquoise1",
                "row_styles": ["on grey3", "on grey7"],
                "metric_style": "light_sky_blue1",
                "value_style": "bold bright_white",
                "caption_style": "dim pale_turquoise1",
                "box": box.ROUNDED,
            },
            "simple": {
                "title_style": "bold",
                "border_style": "dim",
                "header_style": "bold",
                "row_styles": ["", "dim"],
                "metric_style": "cyan",
                "value_style": "white",
                "caption_style": "dim",
                "box": box.SIMPLE,
            },
        }

    def get_style(self, element: str) -> str:
        """Get style for a specific element."""
        return self._themes[self.theme].get(f"{element}_style", "")

    def get_box(self):
        """Get table box style."""
        return self._themes[self.theme]["box"]

    def get_row_styles(self) -> List[str]:
        """Get alternating row styles."""
        return self._themes[self.theme]["row_styles"]


class RichOutput(ConsoleOutput):
    """Rich-based console output with modern terminal UI."""

    def __init__(
        self,
        results: Union[Dict[str, pd.DataFrame], pd.DataFrame],
        metadata: Dict[str, Any],
        args,
    ):
        """Initialize with Rich console."""
        super().__init__(results, metadata, args)

        if not RICH_AVAILABLE:
            raise ImportError(
                "Rich library is not installed. " "Install it with: pip install rich"
            )

        # Initialize Rich console
        self.console = Console()

        # Initialize helpers
        precision = getattr(args, "precision", 6)
        self.formatter = ValueFormatter(precision)
        self.styler = TableStyler(theme="ocean")

    def _build_table(self, title: str, df: pd.DataFrame, max_width: int = 20) -> None:
        # Create a proper Rich table for transposed comparison
        table = Table(
            title=title,
            title_style=self.styler.get_style("title"),
            border_style=self.styler.get_style("border"),
            box=self.styler.get_box(),
            show_lines=False,
            header_style=self.styler.get_style("header"),
            row_styles=self.styler.get_row_styles(),
        )

        # Add metric column(s)
        if "table" in df.index.names:
            table.add_column("Table", style="bright_white", no_wrap=True)
        table.add_column("Metric", style=self.styler.get_style("metric"), no_wrap=True)

        # Add experiment columns with proper width and wrapping
        for exp_col in df.columns:
            # Limit column width to 20 chars with wrapping
            table.add_column(
                str(exp_col),
                justify="right",
                style=self.styler.get_style("value"),
                no_wrap=False,  # Enable wrapping
                overflow="fold",  # Fold long content
                max_width=max_width,  # Maximum column width
            )

        # Add rows
        for idx, row in df.iterrows():
            row_data = []
            # Handle multi-level index
            if isinstance(idx, tuple):
                row_data.extend([str(i) for i in idx])
            else:
                row_data.append(str(idx))

            # Add values
            for col in df.columns:
                value = row[col]
                if pd.notna(value):
                    row_data.append(self.formatter.format(value))
                else:
                    row_data.append("N/A")

            table.add_row(*row_data)
        return table

    def print(
        self,
    ) -> None:
        """Main entry point - prints formatted output based on args.format."""
        # Check if this is a comparison
        self._print_title()

        self._print_header()

        self.console.print(Rule(style="steel_blue"))
        self.console.print()

        # Sometimes we pivot, sometimes we print multiple tables...
        # So we ensure there's at least one table and there's at least one pivot
        if self.args.format == "tables":
            index = ["metric"]
        else:
            index = ["table", "metric"]

        self._print_tables_format(df_dict=self.results, index=index)

        self.console.print(Rule(style="steel_blue"))
        self.console.print()

        self._print_footer_notes(text=self._default_footer_notes_text())

    def _print_title(self, text: Optional[str] = None) -> None:
        if text is None:
            text = self._default_title_text()
        text = Text(text, style="bold steel_blue")
        title_panel = Panel.fit(
            text, border_style="steel_blue", box=box.DOUBLE_EDGE, padding=(0, 2)
        )
        self.console.print()
        self.console.print(title_panel, justify="center")
        self.console.print()

    def _default_title_text(self) -> str:
        """Create header text for LIDRA."""
        return "✨ LIDRA Metrics Analysis Tool ✨"

    def _print_header(self) -> None:
        pass

    def _print_tables_format(
        self,
        df_dict: Dict[str, pd.DataFrame],
        index: List[str] = ("table", "metric"),
    ) -> None:
        """Print results as separate Rich tables."""
        report_config = self.metadata.get("report_config", {}) if self.metadata else {}

        for table_name, df in df_dict.items():
            # df["experiment"] = "Metric"
            df = df.copy()
            if "experiment" not in df.columns:
                df["experiment"] = "Metric"
            df = df.reset_index()
            df = self._pivot_table(df, index=index, columns=["experiment"])

            if self.args.transpose:
                if not self.args.tsv:
                    raise ValueError(
                        "Transpose is only supported with TSV printing -- for copying to Sheets/Excel"
                    )
                df = df.T

            table = self._build_table(f"{table_name.upper()} Metrics", df)

            # Sorting caption footer
            if (
                hasattr(self.args, "mode")
                and self.args.mode == "best"
                and "tables" in report_config
                and table_name in report_config["tables"]
            ):
                table_config = report_config["tables"][table_name]
                if "best_of_trials" in table_config:
                    bot = table_config["best_of_trials"]
                    direction = "⬆" if bot["select_max"] else "⬇"
                    sort_info = f"Selection: best-of-N [bold]{bot['select_column']} {direction}[/bold]"
                    table.caption = f"[dim italic]{sort_info}[/dim italic]"
                    table.caption_style = self.styler.get_style("caption")

            if self.args.tsv:
                self.console.print(Text(table.title, style=table.title_style))
                print("\n")
                print(df.to_csv(sep="\t", index=True))
                self.console.print(table.caption, style=table.caption_style)
                print("\n")
                continue

            self.console.print(table)

    def _print_footer_notes(self, text: str = None) -> None:
        """Print footer notes using Rich formatting."""
        if not self.args.verbose:
            return

        if text is None:
            text = self._default_footer_notes_text()

        self.console.print(Rule(style="dim steel_blue"))
        text = Text(text, style="dim white")

        summary_panel = Panel(
            text,
            title="Summary",
            border_style="green",
            box=box.ROUNDED,
        )

        self.console.print(summary_panel)

    # ========== Shared methods ==========
    def _order_columns(self, df: pd.DataFrame, prefix: List[str]) -> pd.DataFrame:
        """Order columns by prefix."""
        return df[prefix + [col for col in df.columns if col not in prefix]]

    def _pivot_table(
        self,
        df: pd.DataFrame,
        index: List[str] = ("table", "metric"),
        columns: List[str] = ("experiment",),
        values: List[str] = ("value",),
    ) -> pd.DataFrame:
        """Print comparison results with experiments side by side."""
        df = self._order_columns(df, list(index) + list(columns))

        required_columns = set(list(index) + list(columns) + list(values))
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"Cannot transpose comparison without {required_columns} columns"
            )

        # Pivot the data: experiments as columns, metrics as rows
        if columns:
            df = df.pivot_table(
                index=index,
                columns=columns,
                values="value",
                aggfunc="first",
            )
        return df
