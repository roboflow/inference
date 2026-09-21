import inspect
from unittest.mock import MagicMock

import numpy as np
import pytest
from typer.testing import CliRunner

from inference_cli import benchmark
from inference_cli.lib import benchmark_adapter
from inference_cli.lib.benchmark import inference_models_speed


@pytest.mark.parametrize(
    "flags,allowed",
    [
        ([], False),
        (["--allow-untrusted-packages"], True),
        (["--no-allow-untrusted-packages"], False),
    ],
)
def test_cli_requires_explicit_untrusted_package_opt_in(monkeypatch, flags, allowed):
    adapter = MagicMock()
    monkeypatch.setattr(benchmark, "run_inference_models_benchmark", adapter)
    result = CliRunner().invoke(
        benchmark.benchmark_app,
        ["inference-models-speed", "--model_id", "fixture/1", *flags],
    )
    assert result.exit_code == 0, result.output
    assert adapter.call_args.kwargs["allow_untrusted_packages"] is allowed


@pytest.mark.parametrize("allowed", [None, True, False])
def test_benchmark_preserves_safe_default_and_explicit_opt_in(monkeypatch, allowed):
    loader = MagicMock()
    monkeypatch.setattr(inference_models_speed, "AutoModel", loader)
    monkeypatch.setattr(inference_models_speed, "run_model_warm_up", MagicMock())
    monkeypatch.setattr(inference_models_speed, "run_benchmark", MagicMock())
    options = {} if allowed is None else {"allow_untrusted_packages": allowed}
    inference_models_speed.run_inference_models_benchmark(
        "fixture/1",
        [np.zeros((2, 2, 3))],
        MagicMock(),
        turn_images_to_tensors=False,
        **options,
    )
    assert loader.from_pretrained.call_args.kwargs["allow_untrusted_packages"] is bool(
        allowed
    )
    assert (
        inspect.signature(benchmark_adapter.run_inference_models_benchmark)
        .parameters["allow_untrusted_packages"]
        .default
        is False
    )
