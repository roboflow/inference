"""Tests for statistical functionals."""

import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from ..functionals import (
    mean,
    std,
    percentile,
    FUNCTIONALS,
    get_functionals,
    StatisticalFunctional,
)


class TestFunctionals(unittest.TestCase):
    def setUp(self):
        """Create test data."""
        self.data = np.array([1, 2, 3, 4, 5])
        self.single = np.array([42])
        self.empty = np.array([])

    def test_mean(self):
        """Test mean functional."""
        assert_almost_equal(mean(self.data), 3.0)
        assert_almost_equal(mean(self.single), 42.0)
        # Mean of empty array should raise warning but numpy handles it

    def test_std(self):
        """Test standard deviation with Bessel's correction."""
        # Manual calculation: sqrt(sum((x-mean)^2)/(n-1))
        expected_std = np.sqrt(sum((self.data - 3) ** 2) / 4)
        assert_almost_equal(std(self.data), expected_std)
        assert_almost_equal(std(self.data), np.std(self.data, ddof=1))

        # Single value should return 0
        assert_almost_equal(std(self.single), 0.0)

    def test_percentiles(self):
        """Test percentile functionals."""
        # Test with simple data
        data = np.arange(100)  # 0 to 99

        # Check specific percentiles
        assert_almost_equal(FUNCTIONALS["p5"](data), 4.95)
        assert_almost_equal(FUNCTIONALS["p50"](data), 49.5)  # median
        assert_almost_equal(FUNCTIONALS["p95"](data), 94.05)

        # Check percentile creation
        p10 = percentile(10)
        assert_almost_equal(p10(data), 9.9)
        self.assertEqual(p10.__name__, "p10")

    def test_min_max(self):
        """Test min and max functionals."""
        assert_almost_equal(FUNCTIONALS["min"](self.data), 1.0)
        assert_almost_equal(FUNCTIONALS["max"](self.data), 5.0)

        # Test with negative values
        neg_data = np.array([-5, -2, 0, 3, 7])
        assert_almost_equal(FUNCTIONALS["min"](neg_data), -5.0)
        assert_almost_equal(FUNCTIONALS["max"](neg_data), 7.0)

    def test_get_functionals(self):
        """Test functional retrieval."""
        # Request valid functionals
        funcs = get_functionals(["mean", "std", "p95"])
        self.assertEqual(len(funcs), 3)
        self.assertIn("mean", funcs)
        self.assertIn("std", funcs)
        self.assertIn("p95", funcs)

        # Request mix of valid and invalid
        funcs = get_functionals(["mean", "invalid", "std", "nonexistent"])
        self.assertEqual(len(funcs), 2)
        self.assertIn("mean", funcs)
        self.assertIn("std", funcs)
        self.assertNotIn("invalid", funcs)
        self.assertNotIn("nonexistent", funcs)

        # Empty list
        funcs = get_functionals([])
        self.assertEqual(len(funcs), 0)

    def test_functional_consistency(self):
        """Test functionals match numpy equivalents."""
        # Generate random data
        np.random.seed(42)
        data = np.random.randn(1000)

        # Test all match numpy
        assert_almost_equal(FUNCTIONALS["min"](data), np.min(data))
        assert_almost_equal(FUNCTIONALS["max"](data), np.max(data))
        assert_almost_equal(FUNCTIONALS["mean"](data), np.mean(data))
        assert_almost_equal(FUNCTIONALS["std"](data), np.std(data, ddof=1))
        assert_almost_equal(FUNCTIONALS["p50"](data), np.median(data))

    def test_functional_types(self):
        """Test that all functionals return float."""
        data = np.array([1, 2, 3, 4, 5])

        for name, func in FUNCTIONALS.items():
            result = func(data)
            self.assertIsInstance(result, float, f"{name} should return float")

    def test_edge_cases(self):
        """Test edge cases for functionals."""
        # Test with all same values
        same_data = np.array([5, 5, 5, 5, 5])
        assert_almost_equal(mean(same_data), 5.0)
        assert_almost_equal(std(same_data), 0.0)
        assert_almost_equal(FUNCTIONALS["min"](same_data), 5.0)
        assert_almost_equal(FUNCTIONALS["max"](same_data), 5.0)

        # Test with very large values
        large_data = np.array([1e10, 2e10, 3e10])
        assert_almost_equal(mean(large_data), 2e10)

        # Test with very small values
        small_data = np.array([1e-10, 2e-10, 3e-10])
        assert_almost_equal(mean(small_data), 2e-10)

    def test_percentile_edge_cases(self):
        """Test percentile edge cases."""
        # Single value
        assert_almost_equal(FUNCTIONALS["p50"](self.single), 42.0)

        # Two values
        two_vals = np.array([10, 20])
        assert_almost_equal(FUNCTIONALS["p0"](two_vals), 10.0)  # min
        assert_almost_equal(FUNCTIONALS["p50"](two_vals), 15.0)  # median
        assert_almost_equal(FUNCTIONALS["p100"](two_vals), 20.0)  # max

    def test_custom_percentiles(self):
        """Test creating custom percentiles."""
        # Create non-standard percentiles
        p1 = percentile(1)
        p99 = percentile(99)
        p33 = percentile(33.33)

        data = np.arange(1000)
        assert_almost_equal(p1(data), 9.99, decimal=2)
        assert_almost_equal(p99(data), 989.01, decimal=2)
        assert_almost_equal(p33(data), 332.967, decimal=2)

        # Check names
        self.assertEqual(p1.__name__, "p1")
        self.assertEqual(p99.__name__, "p99")
        self.assertEqual(p33.__name__, "p33")  # Truncates to int


if __name__ == "__main__":
    unittest.main()
