"""
Integration tests for AnalysisComparisonProcessor using real data.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd

from ..processor.analysis_comparison import (
    AnalysisComparisonProcessor,
    AnalysisComparisonConfig,
)


class TestAnalysisComparisonIntegration(unittest.TestCase):
    """Integration tests using real evaluation results."""

    def test_should_compare_real_evaluation_results(self):
        """Test comparing two real evaluation results."""
        # The experiments mentioned by the user
        experiments = [
            "v2_mot_6drotation_normalized_layout_ptmap_Feb-May.metrics.r3_anything.epoch=99-step=38400-v1",
            "v2_mot_6drotation_normalized_layout_ptmap_Feb-May.metrics.r3_anything.epoch=74-step=28800-v1",
        ]

        # Create test config
        config = AnalysisComparisonConfig(
            experiments=experiments,
            experiment_names=["Epoch 99", "Epoch 74"],
            analysis_config="stage1",
            statistic="mean",
            save_dir=None,  # Use temp directory
            metrics_filter=["f1", "chamfer_distance", "oriented_rot_error_deg"],
            sort_by="Epoch 99",
            ascending=False,
        )

        processor = AnalysisComparisonProcessor()

        # Mock the experiment resolution to return test paths
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock evaluation data files
            eval_data_1 = self._create_mock_eval_data()
            eval_data_2 = self._create_mock_eval_data(slightly_worse=True)

            eval_path_1 = Path(tmpdir) / "eval1.json"
            eval_path_2 = Path(tmpdir) / "eval2.json"

            import json

            with open(eval_path_1, "w") as f:
                json.dump(eval_data_1, f)
            with open(eval_path_2, "w") as f:
                json.dump(eval_data_2, f)

                # Mock resolution and analysis execution

                # Mock the MultiTrialProcessor to generate expected CSV files
                with patch(
                    "lidra.metrics.analysis.v3.processor.multi_trial.MultiTrialProcessor"
                ) as mock_mtp_class:
                    mock_processor = MagicMock()
                    mock_mtp_class.return_value = mock_processor

                    # Set up side effect to create CSV files when process() is called
                    def create_csv_on_process(cfg):
                        output_dir = Path(cfg.output_directory)
                        output_dir.mkdir(parents=True, exist_ok=True)

                        # Create realistic metrics data
                        if "Epoch 99" in str(output_dir):
                            df = pd.DataFrame(
                                {
                                    "mean": [0.95, 0.12, 5.2, 0.08, 0.10],
                                    "std": [0.02, 0.03, 1.5, 0.02, 0.03],
                                    "p50": [0.96, 0.11, 4.8, 0.07, 0.09],
                                    "p95": [0.98, 0.18, 8.1, 0.12, 0.15],
                                },
                                index=[
                                    "f1",
                                    "chamfer_distance",
                                    "oriented_rot_error_deg",
                                    "scale_abs_rel_error",
                                    "trans_abs_rel_error",
                                ],
                            )
                        else:  # Epoch 74
                            df = pd.DataFrame(
                                {
                                    "mean": [0.93, 0.15, 6.5, 0.10, 0.12],
                                    "std": [0.03, 0.04, 2.0, 0.03, 0.04],
                                    "p50": [0.94, 0.14, 6.0, 0.09, 0.11],
                                    "p95": [0.97, 0.22, 10.2, 0.15, 0.18],
                                },
                                index=[
                                    "f1",
                                    "chamfer_distance",
                                    "oriented_rot_error_deg",
                                    "scale_abs_rel_error",
                                    "trans_abs_rel_error",
                                ],
                            )

                        df.index.name = "metric"
                        df.to_csv(output_dir / "results.csv")

                    mock_processor.process.side_effect = create_csv_on_process

                    # Run the comparison
                    processor.process(config)

                    # Verify analyses were run
                    self.assertEqual(mock_processor.process.call_count, 2)

                    # Check that results directory was created
                    self.assertIsNotNone(processor.results_dir)
                    self.assertTrue(processor.results_dir.exists())

    def test_should_run_parallel_analyses(self):
        """Test parallel execution of analyses."""
        experiments = [f"exp_{i}" for i in range(4)]

        config = AnalysisComparisonConfig(
            experiments=experiments,
            analysis_config="stage1",
            statistic="p95",
            parallel_analyses=True,
            max_workers=2,
        )

        processor = AnalysisComparisonProcessor()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_dir = tmpdir

            # Create mock data files
            for i, exp in enumerate(experiments):
                data_file = Path(tmpdir) / f"{exp}.json"
                with open(data_file, "w") as f:
                    f.write('{"dummy": "data"}')

                # Mock everything needed

                with patch(
                    "lidra.metrics.analysis.v3.processor.multi_trial.MultiTrialProcessor"
                ) as mock_mtp_class:
                    mock_processor = MagicMock()
                    mock_mtp_class.return_value = mock_processor

                    # Create CSV files
                    def create_csv(cfg):
                        output_dir = Path(cfg.output_directory)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        df = pd.DataFrame(
                            {"mean": [0.5], "std": [0.1], "p95": [0.7]},
                            index=["test_metric"],
                        )
                        df.index.name = "metric"
                        df.to_csv(output_dir / "results.csv")

                    mock_processor.process.side_effect = create_csv

                    # Verify parallel execution was used
                    with patch(
                        "concurrent.futures.ThreadPoolExecutor"
                    ) as mock_executor_class:
                        mock_executor = MagicMock()
                        mock_executor_class.return_value.__enter__.return_value = (
                            mock_executor
                        )

                        # Mock futures
                        futures = []
                        for i in range(4):
                            future = MagicMock()
                            future.result.return_value = (
                                Path(tmpdir) / f"exp_{i}" / "results.csv"
                            )
                            futures.append(future)

                        mock_executor.submit.side_effect = futures

                        with patch(
                            "concurrent.futures.as_completed", return_value=futures
                        ):
                            processor.process(config)

                            # Verify thread pool was created with correct max_workers
                            mock_executor_class.assert_called_once_with(max_workers=2)
                            # Verify all experiments were submitted
                            self.assertEqual(mock_executor.submit.call_count, 4)

    def _create_mock_eval_data(self, slightly_worse=False):
        """Create mock evaluation data structure."""
        base_values = {
            "f1": 0.95 if not slightly_worse else 0.93,
            "chamfer_distance": 0.12 if not slightly_worse else 0.15,
            "oriented_rot_error_deg": 5.2 if not slightly_worse else 6.5,
        }

        # Create realistic evaluation data structure
        return {
            "metrics": {
                metric: {
                    "values": [base_values[metric] + (i * 0.01) for i in range(100)]
                }
                for metric in base_values
            }
        }


if __name__ == "__main__":
    unittest.main()
