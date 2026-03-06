from abc import ABC, abstractmethod
from ..definitions import Report
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from dataclasses import asdict
from datetime import datetime


class Base(ABC):

    # TODO: Abstract methods

    def save_results(
        self,
        output_dir: str,
        results: Dict[str, pd.DataFrame],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save analysis results to directory structure.

        Directory structure:
            output_dir/
                shape.csv           # Results for shape table
                oriented.csv        # Results for oriented table
                metadata.yaml       # Analysis metadata (if provided)

        Args:
            output_dir: Directory to save results
            results: Dict mapping table_name -> DataFrame
            metadata: Optional metadata dict to save as YAML
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if isinstance(results, pd.DataFrame):
            results.to_csv(output_path / "combined.csv")
        else:
            # Save each table to CSV
            for table_name, df in results.items():
                csv_path = output_path / f"{table_name}.csv"
                df.to_csv(csv_path)

        # Save metadata if provided
        if metadata:
            metadata_path = output_path / "metadata.yaml"
            with open(metadata_path, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    def generate_run_metadata(
        self,
        mode: str,
        results: Dict[str, pd.DataFrame],
        input_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate metadata for the analysis."""
        metadata = {
            "run_info": {
                "timestamp": datetime.now().isoformat(),
                "input_file": input_file or "unknown",
                "mode": mode,
                "version": "1.0",
            },
            "input_data_summary": {
                "total_rows": len(self.df),
                "unique_samples": self.df["sample_uuid"].nunique(),
                "metrics_count": self._count_available_metrics(),
            },
            "report_config": asdict(self.report),
        }

        # Add trial filtering info if rows were filtered
        if hasattr(self, "original_row_count") and self.original_row_count != len(
            self.df
        ):
            metadata["input_data_summary"]["original_rows"] = self.original_row_count
            metadata["input_data_summary"]["rows_after_filtering"] = len(self.df)

        return metadata
