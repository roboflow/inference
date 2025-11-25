"""
Tests for AnalysisComparisonProcessor.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pandas as pd

from ..processor.analysis_comparison import AnalysisComparisonProcessor, AnalysisComparisonConfig

class TestAnalysisComparisonConfig(unittest.TestCase):
    """Test configuration validation."""
    
    def test_should_require_experiments(self):
        """Test that experiments field is required."""
        with self.assertRaises(ValueError) as cm:
            AnalysisComparisonConfig()
        self.assertIn("At least one experiment", str(cm.exception))
    
    def test_should_validate_experiment_names_length(self):
        """Test that experiment names must match experiments length."""
        with self.assertRaises(ValueError) as cm:
            AnalysisComparisonConfig(
                experiments=["/data/exp1.json", "/data/exp2.json"],
                experiment_names=["name1"]  # Mismatch
            )
        self.assertIn("must match number of experiments", str(cm.exception))
    
    def test_should_accept_valid_config(self):
        """Test valid configuration is accepted."""
        config = AnalysisComparisonConfig(
            experiments=["/data/exp1.json", "/data/exp2.json"],
            experiment_names=["Baseline", "Improved"],
            analysis_config="stage2",
            statistic="p95",
            save_dir="/tmp/results",
            parallel_analyses=True,
            max_workers=8
        )
        
        self.assertEqual(config.experiments, ["/data/exp1.json", "/data/exp2.json"])
        self.assertEqual(config.experiment_names, ["Baseline", "Improved"])
        self.assertEqual(config.analysis_config, "stage2")
        self.assertEqual(config.statistic, "p95")
        self.assertEqual(config.save_dir, "/tmp/results")
        self.assertTrue(config.parallel_analyses)
        self.assertEqual(config.max_workers, 8)

class TestAnalysisComparisonProcessor(unittest.TestCase):
    """Test AnalysisComparisonProcessor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = AnalysisComparisonProcessor()
        
        # Create sample DataFrames
        self.df1 = pd.DataFrame({
            'mean': [10.5, 20.3, 30.1],
            'std': [1.2, 2.3, 3.4],
            'p50': [10.0, 20.0, 30.0],
            'p95': [12.0, 23.0, 33.0]
        }, index=['metric_a', 'metric_b', 'metric_c'])
        
        self.df2 = pd.DataFrame({
            'mean': [11.5, 21.3, 31.1],
            'std': [1.3, 2.4, 3.5],
            'p50': [11.0, 21.0, 31.0],
            'p95': [13.0, 24.0, 34.0]
        }, index=['metric_a', 'metric_b', 'metric_c'])
    
    def test_load_data_returns_empty_dataframe(self):
        """Test that load_data returns empty DataFrame."""
        config = AnalysisComparisonConfig(experiments=["/data/exp1.json"])
        result = self.processor.load_data(config)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
    
    @patch('lidra.metrics.analysis.v3.processor.analysis_comparison.AnalysisComparisonProcessor._run_all_analyses')
    def test_extract_data_runs_analyses_and_loads_results(self, mock_run):
        """Test extract_data runs analyses and loads CSV results."""
        # Setup
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock CSV files
            exp1_path = Path(tmpdir) / "/data/exp1.json" / "results.csv"
            exp2_path = Path(tmpdir) / "/data/exp2.json" / "results.csv"
            exp1_path.parent.mkdir(parents=True)
            exp2_path.parent.mkdir(parents=True)
            
            # Ensure index name is set for CSV files
            self.df1.index.name = 'metric'
            self.df2.index.name = 'metric'
            self.df1.to_csv(exp1_path)
            self.df2.to_csv(exp2_path)
            
            # Configure mock
            mock_run.return_value = [exp1_path, exp2_path]
            
            # Run
            config = AnalysisComparisonConfig(
                experiments=["/data/exp1.json", "/data/exp2.json"],
                experiment_names=["Baseline", "Improved"],
                save_dir=tmpdir
            )
            
            results = self.processor.extract_data(pd.DataFrame(), config)
            
            # Verify
            self.assertEqual(len(results), 2)
            self.assertIn("Baseline", results)
            self.assertIn("Improved", results)
            pd.testing.assert_frame_equal(results["Baseline"], self.df1)
            pd.testing.assert_frame_equal(results["Improved"], self.df2)
    
    def test_transform_data_extracts_statistic_and_joins(self):
        """Test transform_data extracts selected statistic and joins DataFrames."""
        # Setup
        data = {
            "Baseline": self.df1,
            "Improved": self.df2
        }
        config = AnalysisComparisonConfig(
            experiments=["/data/exp1.json", "/data/exp2.json"],
            statistic="mean"
        )
        
        # Run
        result = self.processor.transform_data(data, config)
        
        # Verify
        self.assertEqual(list(result.columns), ["Baseline", "Improved"])
        self.assertEqual(list(result.index), ['metric_a', 'metric_b', 'metric_c'])
        self.assertEqual(result.loc['metric_a', 'Baseline'], 10.5)
        self.assertEqual(result.loc['metric_a', 'Improved'], 11.5)
    
    def test_transform_data_with_filtering(self):
        """Test transform_data applies metric filtering."""
        # Setup
        data = {
            "Baseline": self.df1,
            "Improved": self.df2
        }
        config = AnalysisComparisonConfig(
            experiments=["/data/exp1.json", "/data/exp2.json"],
            statistic="mean",
            metrics_filter=["metric_a", "metric_c"]
        )
        
        # Run
        result = self.processor.transform_data(data, config)
        
        # Verify
        self.assertEqual(len(result), 2)
        self.assertIn("metric_a", result.index)
        self.assertIn("metric_c", result.index)
        self.assertNotIn("metric_b", result.index)
    
    def test_transform_data_with_sorting(self):
        """Test transform_data applies sorting."""
        # Setup
        data = {
            "Baseline": self.df1,
            "Improved": self.df2
        }
        config = AnalysisComparisonConfig(
            experiments=["/data/exp1.json", "/data/exp2.json"],
            statistic="mean",
            sort_by="Improved",
            ascending=False
        )
        
        # Run
        result = self.processor.transform_data(data, config)
        
        # Verify - should be sorted by Improved column descending
        self.assertEqual(list(result.index), ['metric_c', 'metric_b', 'metric_a'])
    
    def test_compute_results_handles_missing_values(self):
        """Test compute_results fills missing values."""
        # Setup with missing data
        df = pd.DataFrame({
            'Baseline': [10.5, None, 30.1],
            'Improved': [11.5, 21.3, None]
        }, index=['metric_a', 'metric_b', 'metric_c'])
        
        config = AnalysisComparisonConfig(
            experiments=["/data/exp1.json", "/data/exp2.json"],
            show_missing_as="--"
        )
        
        # Run
        result = self.processor.compute_results(df, config)
        
        # Verify
        self.assertEqual(result.loc['metric_b', 'Baseline'], "--")
        self.assertEqual(result.loc['metric_c', 'Improved'], "--")
    
    def test_compute_results_with_transpose(self):
        """Test compute_results applies transpose."""
        # Setup
        df = pd.DataFrame({
            'Baseline': [10.5, 20.3, 30.1],
            'Improved': [11.5, 21.3, 31.1]
        }, index=['metric_a', 'metric_b', 'metric_c'])
        
        config = AnalysisComparisonConfig(
            experiments=["/data/exp1.json", "/data/exp2.json"],
            transpose=True
        )
        
        # Run
        result = self.processor.compute_results(df, config)
        
        # Verify
        self.assertEqual(list(result.index), ['Baseline', 'Improved'])
        self.assertEqual(list(result.columns), ['metric_a', 'metric_b', 'metric_c'])
    
    @patch('lidra.metrics.analysis.v3.processor.analysis_comparison.ComparisonFormatter')
    def test_get_output_uses_formatter(self, mock_formatter_class):
        """Test get_output uses ComparisonFormatter."""
        # Setup
        mock_formatter = MagicMock()
        mock_table = MagicMock()
        mock_formatter.format.return_value = mock_table
        mock_formatter_class.return_value = mock_formatter
        
        df = pd.DataFrame({'col1': [1, 2]})
        config = AnalysisComparisonConfig(
            experiments=["/data/exp1.json"],
            statistic="mean",
            console_format="rich",
            format_precision=3,
            transpose=False
        )
        
        # Run
        result = self.processor.get_output(df, config, "rich")
        
        # Verify
        mock_formatter_class.assert_called_once_with(transpose=False)
        mock_formatter.format.assert_called_once_with(results=df, config=config)
        # Result should be a string (from Console output)
        self.assertIsInstance(result, str)
    
    def test_determine_results_directory_with_save_dir(self):
        """Test _determine_results_directory uses provided save_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnalysisComparisonConfig(
                experiments=["/data/exp1.json"],
                save_dir=tmpdir
            )
            
            result = self.processor._determine_results_directory(config)
            
            self.assertEqual(result, Path(tmpdir))
    
    def test_determine_results_directory_creates_temp(self):
        """Test _determine_results_directory creates temp dir when save_dir not set."""
        config = AnalysisComparisonConfig(experiments=["/data/exp1.json"])
        
        result = self.processor._determine_results_directory(config)
        
        self.assertTrue(result.exists())
        self.assertTrue(str(result).startswith(tempfile.gettempdir()))
        self.assertIn("lidra_comparison", str(result))
    
    @patch('lidra.metrics.analysis.v3.processor.analysis_comparison.AnalysisComparisonProcessor._run_single_analysis')
    def test_run_sequential_analyses(self, mock_run_single):
        """Test _run_sequential_analyses runs analyses in order."""
        # Setup
        mock_run_single.side_effect = [
            Path("/tmp/exp1/results.csv"),
            Path("/tmp/exp2/results.csv"),
            Path("/tmp/exp3/results.csv")
        ]
        
        config = AnalysisComparisonConfig(
            experiments=["/data/exp1.json", "/data/exp2.json", "exp3"]
        )
        
        # Run
        result = self.processor._run_sequential_analyses(config)
        
        # Verify
        self.assertEqual(len(result), 3)
        self.assertEqual(mock_run_single.call_count, 3)
        mock_run_single.assert_has_calls([
            call("/data/exp1.json", 0, config),
            call("/data/exp2.json", 1, config),
            call("exp3", 2, config)
        ])
    
    @patch('lidra.metrics.analysis.v3.processor.analysis_comparison.AnalysisComparisonProcessor._run_single_analysis')
    @patch('concurrent.futures.ThreadPoolExecutor')
    def test_run_parallel_analyses(self, mock_executor_class, mock_run_single):
        """Test _run_parallel_analyses uses ThreadPoolExecutor."""
        # Setup mock executor
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Setup futures
        future1 = MagicMock()
        future1.result.return_value = Path("/tmp/exp1/results.csv")
        future2 = MagicMock()
        future2.result.return_value = Path("/tmp/exp2/results.csv")
        
        mock_executor.submit.side_effect = [future1, future2]
        
        # Mock as_completed to return futures in order
        with patch('concurrent.futures.as_completed', return_value=[future1, future2]):
            config = AnalysisComparisonConfig(
                experiments=["/data/exp1.json", "/data/exp2.json"],
                parallel_analyses=True,
                max_workers=2
            )
            
            # Run
            result = self.processor._run_parallel_analyses(config)
            
            # Verify
            self.assertEqual(len(result), 2)
            mock_executor_class.assert_called_once_with(max_workers=2)
            self.assertEqual(mock_executor.submit.call_count, 2)
    
    @patch('lidra.metrics.analysis.v3.processor.multi_trial.MultiTrialProcessor')
    def test_run_single_analysis(self, mock_processor_class):
        """Test _run_single_analysis creates config and runs processor."""
        # Setup
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.processor.results_dir = Path(tmpdir)
            
            # Mock both _resolve_experiment_data and _create_analysis_config
                            # Create a mock config that won't fail validation
                mock_config = MagicMock()
                mock_config.output_directory = str(Path(tmpdir) / "Baseline")
                
                with patch.object(self.processor, '_create_analysis_config', return_value=mock_config):
                    config = AnalysisComparisonConfig(
                        experiments=["/data/exp1.json"],
                        experiment_names=["Baseline"]
                    )
                    
                    # Run
                    result = self.processor._run_single_analysis("/data/exp1.json", 0, config)
                    
                    # Verify
                    expected_path = Path(tmpdir) / "Baseline" / "results.csv"
                    self.assertEqual(result, expected_path)
                    mock_processor.process.assert_called_once_with(mock_config)
    
    def test_end_to_end_sequential(self, mock_processor_class):
        """Test end-to-end sequential processing."""
        # Setup
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create result files that will be "generated" by MultiTrialProcessor
            exp1_dir = Path(tmpdir) / "Baseline"
            exp2_dir = Path(tmpdir) / "Improved"
            exp1_dir.mkdir(parents=True)
            exp2_dir.mkdir(parents=True)
            
            # Ensure index name is set for CSV files
            self.df1.index.name = 'metric'
            self.df2.index.name = 'metric'
            self.df1.to_csv(exp1_dir / "results.csv")
            self.df2.to_csv(exp2_dir / "results.csv")
            
            # Configure
            config = AnalysisComparisonConfig(
                experiments=["/data/exp1.json", "/data/exp2.json"],
                experiment_names=["Baseline", "Improved"],
                save_dir=tmpdir,
                statistic="mean",
                metrics_filter=["metric_a", "metric_c"],
                sort_by="Improved"
            )
            
            # Mock experiment resolution and config creation
                            # Create mock configs that won't fail validation
                mock_config1 = MagicMock()
                mock_config1.output_directory = str(exp1_dir)
                mock_config2 = MagicMock()
                mock_config2.output_directory = str(exp2_dir)
                
                with patch.object(self.processor, '_create_analysis_config', side_effect=[mock_config1, mock_config2]):
                    # Run full pipeline
                    self.processor.process(config)
                    
                    # Verify MultiTrialProcessor was called twice
                    self.assertEqual(mock_processor.process.call_count, 2)
    
    @patch('concurrent.futures.ThreadPoolExecutor')
    @patch('lidra.metrics.analysis.v3.processor.multi_trial.MultiTrialProcessor')
    def test_end_to_end_parallel(self, mock_processor_class, mock_executor_class):
        """Test end-to-end parallel processing."""
        # Setup
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create result files
            exp1_dir = Path(tmpdir) / "/data/exp1.json"
            exp2_dir = Path(tmpdir) / "/data/exp2.json"
            exp1_dir.mkdir(parents=True)
            exp2_dir.mkdir(parents=True)
            
            # Ensure index name is set for CSV files
            self.df1.index.name = 'metric'
            self.df2.index.name = 'metric'
            self.df1.to_csv(exp1_dir / "results.csv")
            self.df2.to_csv(exp2_dir / "results.csv")
            
            # Configure
            config = AnalysisComparisonConfig(
                experiments=["/data/exp1.json", "/data/exp2.json"],
                save_dir=tmpdir,
                parallel_analyses=True,
                max_workers=2,
                statistic="mean"
            )
            
            # Mock parallel execution
            future1 = MagicMock()
            future1.result.return_value = exp1_dir / "results.csv"
            future2 = MagicMock()
            future2.result.return_value = exp2_dir / "results.csv"
            
            mock_executor = MagicMock()
            mock_executor.submit.side_effect = [future1, future2]
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            
            with patch('concurrent.futures.as_completed', return_value=[future1, future2]):
                                    # Create mock configs
                    mock_config1 = MagicMock()
                    mock_config1.output_directory = str(exp1_dir)
                    mock_config2 = MagicMock()
                    mock_config2.output_directory = str(exp2_dir)
                    
                    with patch.object(self.processor, '_create_analysis_config', side_effect=[mock_config1, mock_config2]):
                        # Run
                        self.processor.process(config)
                        
                        # Verify parallel execution was used
                        mock_executor_class.assert_called_once_with(max_workers=2)

if __name__ == '__main__':
    unittest.main()