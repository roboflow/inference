"""Run inside a parallel image with PARALLEL_IMAGE_TEST=true.

Set PARALLEL_TEST_DEVICE=cuda on GPU runners; CPU validation defaults to cpu.
"""

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PARALLEL_IMAGE_TEST", "false").lower() != "true",
    reason="Requires the built parallel image and its Celery dependencies.",
)


def test_task_import_allows_tensor_preprocessing_after_fork() -> None:
    """Exercise the task import before the prefork worker's first tensor allocation."""
    code = textwrap.dedent("""
        import multiprocessing
        import os

        # Import the real task application before forking, as Celery does.
        from inference.enterprise.parallel.tasks import app
        from inference.enterprise.parallel.utils import _image_to_numpy
        import numpy as np
        import torch

        device = os.environ.get("PARALLEL_TEST_DEVICE", "cpu")
        assert device in {"cpu", "cuda"}
        assert os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") == "1"
        if device == "cuda":
            assert torch.cuda.is_available(), "CUDA coverage must not fall back to CPU"
            assert not torch.cuda.is_initialized(), "Task import initialized CUDA"

        def preprocess_in_child():
            image = torch.arange(24, dtype=torch.float32, device=device).reshape(3, 2, 4)
            actual = _image_to_numpy(image)
            np.testing.assert_array_equal(actual, np.arange(24).reshape(3, 2, 4))

        worker = multiprocessing.get_context("fork").Process(target=preprocess_in_child)
        worker.start()
        try:
            worker.join(timeout=30)
            assert worker.exitcode == 0, f"Prefork worker failed: {worker.exitcode}"
        finally:
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
            worker.close()
        """)
    env = dict(os.environ, USE_PYTORCH_FOR_PREPROCESSING="True")

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stdout + result.stderr
