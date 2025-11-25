"""Tests for base Processor class."""

import unittest
from unittest.mock import Mock, patch, mock_open
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..base import Processor


@dataclass
class TestConfig:
    """Test configuration class."""

    input_file: Optional[str] = None
    output_file: Optional[str] = None


class TestProcessorImpl(Processor[TestConfig, dict, dict]):
    """Concrete test implementation of Processor."""

    def extract_data(self, raw_data: pd.DataFrame, config: TestConfig) -> dict:
        return {"test": raw_data.values.flatten()}

    def transform_data(self, data: dict, config: TestConfig) -> dict:
        return {k: v * 2 for k, v in data.items()}

    def compute_results(self, data: dict, config: TestConfig) -> dict:
        return {k: {"mean": np.mean(v)} for k, v in data.items()}

    def format_output(self, results: dict, config: TestConfig) -> str:
        return str(results)


class TestProcessor(unittest.TestCase):
    def setUp(self):
        """Create a concrete test processor."""
        self.processor = TestProcessorImpl()

    def test_process_pipeline(self):
        """Test the complete pipeline executes in order."""
        config = TestConfig(input_file="test.csv")
        df = pd.DataFrame({"col1": [1, 2, 3]})

        with patch.object(self.processor, "load_data", return_value=df) as mock_load:
            with patch.object(
                self.processor,
                "extract_data",
                return_value={"test": np.array([1, 2, 3])},
            ) as mock_extract:
                with patch.object(
                    self.processor,
                    "transform_data",
                    return_value={"test": np.array([2, 4, 6])},
                ) as mock_transform:
                    with patch.object(
                        self.processor,
                        "compute_results",
                        return_value={"test": {"mean": 4.0}},
                    ) as mock_compute:
                        with patch.object(
                            self.processor, "format_output", return_value="output"
                        ) as mock_format:
                            with patch.object(
                                self.processor, "write_output"
                            ) as mock_write:
                                self.processor.process(config)

                                # Verify call order
                                mock_load.assert_called_once_with(config)
                                mock_extract.assert_called_once_with(df, config)
                                mock_transform.assert_called_once()
                                mock_compute.assert_called_once()
                                mock_format.assert_called_once()
                                mock_write.assert_called_once_with("output", config)

    def test_default_load_data(self):
        """Test default CSV loading."""
        config = TestConfig(input_file="test.csv")
        test_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        with patch("lidra.metrics.analysis.v3.base.pd.read_csv") as mock_read:
            mock_read.return_value = test_df
            result = self.processor.load_data(config)

            mock_read.assert_called_once_with("test.csv")
            pd.testing.assert_frame_equal(result, test_df)

    def test_load_data_no_input_file(self):
        """Test error when no input_file in config."""
        config = TestConfig()  # No input_file
        with self.assertRaises(NotImplementedError):
            self.processor.load_data(config)

    def test_write_output_to_file(self):
        """Test writing to file when output_file specified."""
        config = TestConfig(output_file="out.txt")
        output = "test output"

        m = mock_open()
        with patch("builtins.open", m):
            self.processor.write_output(output, config)

        m.assert_called_once_with("out.txt", "w")
        m().write.assert_called_once_with(output)

    def test_write_output_to_console(self):
        """Test printing when no output_file specified."""
        config = TestConfig()  # No output_file
        output = "test output"

        with patch("builtins.print") as mock_print:
            self.processor.write_output(output, config)
            mock_print.assert_called_once_with(output)

    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods must be implemented."""

        # Create a partial implementation
        class IncompleteProcessor(Processor[TestConfig, dict, dict]):
            pass

        # Should not be able to instantiate
        with self.assertRaises(TypeError):
            IncompleteProcessor()


if __name__ == "__main__":
    unittest.main()
