import inspect
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from typer.testing import CliRunner

from inference_cli import benchmark
from inference_cli.lib import benchmark_adapter
from inference_cli.lib.benchmark import inference_models_speed

ROOT = Path(__file__).resolve().parents[4]


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
        **options
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


def test_all_current_jetson_images_retain_trust_filter_and_disable_stream_default():
    images = list((ROOT / "docker/dockerfiles").glob("Dockerfile.onnx.jetson.*"))
    assert {p.name for p in images} >= {
        "Dockerfile.onnx.jetson.5.1.1",
        "Dockerfile.onnx.jetson.6.2.0",
        "Dockerfile.onnx.jetson.7.2.0",
    }
    for image in images:
        source = image.read_text()
        assert "ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES=True" not in source
        assert "ALLOW_INFERENCE_EXP_UNTRUSTED_MODELS=True" not in source
        assert "ENABLE_STREAM_API=True" not in source


def test_all_first_party_images_and_desktop_bundles_boot_without_stream_token():
    import ast
    import re

    images = list((ROOT / "docker/dockerfiles").glob("Dockerfile*"))
    for image in images:
        assert not re.search(
            r"\bENABLE_STREAM_API(?:=|\s+)[\"']?(?:true|1|yes)\b",
            image.read_text(),
            re.I,
        ), image
    for platform in ["osx", "windows"]:
        path = ROOT / "app_bundles" / platform / "run_inference.py"
        defaults = [
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setdefault"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "ENABLE_STREAM_API"
        ]
        assert len(defaults) == 1, path
        assert ast.literal_eval(defaults[0].args[1]) == "False", path
