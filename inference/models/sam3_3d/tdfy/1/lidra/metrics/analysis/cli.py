"""Command-line interface for metrics analysis."""

import argparse
import sys
from pathlib import Path

from .processor.multi_trial import TrialMetricsProcessor, RichOutput
from .utils import combine_tables
from omegaconf import OmegaConf
from hydra.utils import instantiate

LIDRA_CONF_ROOT = str(Path(__file__).parent.parent.parent.parent / "etc" / "lidra")
DEFAULT_REPORT_PATH = (
    Path(LIDRA_CONF_ROOT) / "reports" / "tdfy" / "multi_trial_stage_1.yaml"
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze lidra evaluation metrics CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with directory output
  python -m scripts.ssax.analysis -i /path/to/metrics.csv -o output_dir/
  
  # Single file output (backward compatibility)
  python -m scripts.ssax.analysis -i /path/to/metrics.csv -o results.csv --single-file
  
  # Console output with specific tables
  python -m scripts.ssax.analysis -i /path/to/metrics.csv --tables shape oriented
  
  # Console output in single-file format
  python -m scripts.ssax.analysis -i /path/to/metrics.csv --format single
  
  # Analyze only first 3 trials per sample
  python -m scripts.ssax.analysis -i /path/to/metrics.csv --max-trials 3
  
  # Use custom report configuration
  python -m scripts.ssax.analysis -i /path/to/metrics.csv -o output_dir/ --report-config custom_report.yaml
  
  # Drop rows with any missing metric values (also drops incomplete trials by default)
  python -m scripts.ssax.analysis -i /path/to/metrics.csv --drop-na
  
  # Drop rows with missing values in specific columns
  python -m scripts.ssax.analysis -i /path/to/metrics.csv --drop-na-columns chamfer_l1 iou
  
  # Drop rows with missing values but keep incomplete trials
  python -m scripts.ssax.analysis -i /path/to/metrics.csv --drop-na --no-drop-incomplete-trials
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input metrics CSV file",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output path (directory by default, file with --single-file)",
    )

    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Output single CSV file instead of directory (backward compatibility)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed statistics",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["mean", "best"],
        default="best",
        help="Analysis mode: 'mean' for overall metrics, 'best' for best-of-trials (default: best)",
    )

    parser.add_argument(
        "--tables",
        type=str,
        nargs="+",
        help="Specify which tables to analyze (default: all)",
    )

    parser.add_argument(
        "--report-config",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help="Path to custom report configuration YAML file",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip metadata.yaml generation for directory output",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["tables", "single"],
        default="tables",
        help="Console output format: 'tables' for separate tables (default), 'single' for combined format",
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        help="Keep only the first N trials per sample (e.g., --max-trials 3 keeps trials 0, 1, 2)",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )

    parser.add_argument(
        "--override",
        "-o",
        type=str,
        nargs="+",
        help="Hydra-style config overrides (e.g., tables.shape.best_of_trials.select_max=true)",
    )

    parser.add_argument(
        "--tsv",
        action="store_true",
        help="Output results in TSV format for easy copying to Excel",
    )

    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Transpose the output table",
    )

    parser.add_argument(
        "--design",
        type=str,
        choices=["simple", "rich"],
        default="rich",
        help="Console output design: simple=Basic text, rich=Rich tables",
    )

    parser.add_argument(
        "--drop-na",
        action="store_true",
        help="Drop rows (sample UUIDs) that have missing values in any metric column",
    )

    parser.add_argument(
        "--drop-na-columns",
        type=str,
        nargs="+",
        help="Drop rows with missing values only in specified columns (e.g., --drop-na-columns chamfer_l1 iou)",
    )

    parser.add_argument(
        "--no-drop-incomplete-trials",
        action="store_true",
        help="Don't drop samples with incomplete trials when using --drop-na (incomplete trials are dropped by default)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.input).exists():
        parser.error(f"Input file does not exist: {args.input}")

    return args


def main(args=None):
    """Main execution function."""
    if args is None:
        args = parse_arguments()

    # Load report configuration
    if args.report_config:
        cfg = OmegaConf.load(args.report_config)

    # Apply overrides if provided
    if args.override:
        overrides = OmegaConf.from_dotlist(args.override)
        cfg = OmegaConf.merge(cfg, overrides)

    # Instantiate the report
    report = instantiate(cfg)

    # Filter tables if specified (before creating analyzer)
    if args.tables:
        # Create a new Report with only the requested tables
        filtered_tables = {k: v for k, v in report.tables.items() if k in args.tables}
        if not filtered_tables:
            raise ValueError(
                f"No valid tables found. Available: {list(report.tables.keys())}"
            )
        from .definitions import Report

        report = Report(tables=filtered_tables)

    # Create processor
    processor = TrialMetricsProcessor.from_csv(args.input, report)
    if args.max_trials is not None:
        processor.filter_trials(args.max_trials)

    # Filter out rows with missing values if requested
    if args.drop_na or args.drop_na_columns:
        processor.drop_missing_values(
            drop_all=args.drop_na,
            columns=args.drop_na_columns,
            drop_incomplete_trials=not args.no_drop_incomplete_trials,
        )

    output_results = processor.create_report(
        args.mode, concat_tables=args.format == "single"
    )

    # Generate metadata (always needed for console output or saving)
    metadata = processor.generate_run_metadata(args.mode, output_results, args.input)

    # Handle different output modes
    if args.output:
        save_metadata = None if args.no_metadata else metadata
        processor.save_results(
            args.output,
            output_results,
            save_metadata,
        )
        print(f"\nResults saved to directory: {args.output}")

    if not args.quiet:
        console = RichOutput(output_results, metadata, args)
        console.print()


if __name__ == "__main__":
    main()
