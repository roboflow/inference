from unittest.mock import MagicMock, patch

import pytest

from inference.core.utils.cuda_health import CudaHealthChecker


class TestCudaHealthChecker:
    def setup_method(self):
        """Create a fresh checker for each test."""
        self.checker = CudaHealthChecker()

    def test_cpu_environment_no_torch(self):
        """When torch is not installed, should always return healthy."""
        with patch.dict("sys.modules", {"torch": None}):
            self.checker._gpu_available = None  # reset cache
            is_healthy, error = self.checker.check_health()
            assert is_healthy is True
            assert error is None

    def test_cpu_environment_no_cuda(self):
        """When torch is available but CUDA is not, should return healthy."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            self.checker._gpu_available = None
            is_healthy, error = self.checker.check_health()
            assert is_healthy is True
            assert error is None

    def test_healthy_gpu(self):
        """When CUDA operations succeed, should return healthy."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.synchronize.return_value = None
        mock_torch.cuda.mem_get_info.return_value = (4_000_000_000, 8_000_000_000)

        self.checker._gpu_available = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            is_healthy, error = self.checker.check_health()
            assert is_healthy is True
            assert error is None
            mock_torch.cuda.synchronize.assert_called_once()
            mock_torch.cuda.mem_get_info.assert_called_once()

    def test_cuda_synchronize_failure(self):
        """When torch.cuda.synchronize() fails, should detect CUDA corruption."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.synchronize.side_effect = RuntimeError(
            "CUDA error: an illegal memory access was encountered"
        )

        self.checker._gpu_available = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            is_healthy, error = self.checker.check_health()
            assert is_healthy is False
            assert "illegal memory access" in error
            assert self.checker.is_failed is True

    def test_mem_get_info_failure(self):
        """When mem_get_info fails (after synchronize succeeds), should detect failure."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.synchronize.return_value = None
        mock_torch.cuda.mem_get_info.side_effect = RuntimeError("CUDA runtime error")

        self.checker._gpu_available = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            is_healthy, error = self.checker.check_health()
            assert is_healthy is False
            assert "CUDA runtime error" in error

    def test_failure_is_cached(self):
        """After first CUDA failure, subsequent checks should return cached failure
        without calling torch again."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.synchronize.side_effect = RuntimeError("CUDA error")

        self.checker._gpu_available = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            # First call: detects failure
            is_healthy1, error1 = self.checker.check_health()
            assert is_healthy1 is False
            assert mock_torch.cuda.synchronize.call_count == 1

            # Second call: returns cached failure, no new CUDA calls
            mock_torch.cuda.synchronize.reset_mock()
            is_healthy2, error2 = self.checker.check_health()
            assert is_healthy2 is False
            assert error2 == error1
            mock_torch.cuda.synchronize.assert_not_called()

    def test_failure_info(self):
        """failure_info should return error details after failure."""
        assert self.checker.failure_info is None

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.synchronize.side_effect = RuntimeError("CUDA error")

        self.checker._gpu_available = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            self.checker.check_health()

        info = self.checker.failure_info
        assert info is not None
        assert "CUDA error" in info["error"]
        assert info["failed_at"] is not None

    def test_gpu_available_is_cached(self):
        """_is_gpu_environment() should only check torch once."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert self.checker._is_gpu_environment() is False
            assert self.checker._is_gpu_environment() is False
            # Only called once despite two invocations
            mock_torch.cuda.is_available.assert_called_once()
