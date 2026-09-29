import os

import pytest
import torch

pytest.importorskip(
    "onnxruntime",
    reason="onnxruntime is not installed (requires the onnx-* extra)",
)

from inference_models.models.common import onnx


@pytest.fixture
def onnxruntime_version(monkeypatch: pytest.MonkeyPatch):
    def set_version(version: str) -> None:
        monkeypatch.setattr(onnx.onnxruntime, "__version__", version)

    return set_version


def test_coreml_provider_passes_through_without_opt_in(tmp_path) -> None:
    providers = onnx.set_onnx_execution_provider_defaults(
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        model_package_path=str(tmp_path),
        device=torch.device("cpu"),
    )

    assert providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_coreml_provider_gets_mlprogram_defaults_when_opted_in(
    tmp_path, onnxruntime_version
) -> None:
    onnxruntime_version("1.22.1")

    providers = onnx.set_onnx_execution_provider_defaults(
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        model_package_path=str(tmp_path),
        device=torch.device("cpu"),
        default_onnx_coreml_options=True,
    )

    provider_name, provider_options = providers[0]
    assert provider_name == "CoreMLExecutionProvider"
    assert provider_options["ModelFormat"] == "MLProgram"
    assert provider_options["MLComputeUnits"] == "CPUAndGPU"
    assert provider_options["ModelCacheDirectory"] == os.path.join(
        str(tmp_path), "coreml_cache", "ort-1.22.1-MLProgram-CPUAndGPU"
    )
    assert providers[1] == "CPUExecutionProvider"


def test_coreml_cache_directory_is_versioned_by_onnxruntime(
    tmp_path, onnxruntime_version
) -> None:
    onnxruntime_version("1.21.1")
    older = onnx.get_default_coreml_provider_options(str(tmp_path))
    onnxruntime_version("1.22.1")
    newer = onnx.get_default_coreml_provider_options(str(tmp_path))

    assert older["ModelCacheDirectory"] != newer["ModelCacheDirectory"]


@pytest.mark.parametrize("version", ["1.20.1", "1.15.1", "not-a-version"])
def test_coreml_provider_passes_through_on_onnxruntime_without_string_options(
    tmp_path, onnxruntime_version, version: str
) -> None:
    onnxruntime_version(version)

    providers = onnx.set_onnx_execution_provider_defaults(
        providers=["CoreMLExecutionProvider"],
        model_package_path=str(tmp_path),
        device=torch.device("cpu"),
        default_onnx_coreml_options=True,
    )

    assert providers == ["CoreMLExecutionProvider"]


def test_coreml_dev_build_of_supported_version_gets_options(
    tmp_path, onnxruntime_version
) -> None:
    onnxruntime_version("1.21.0.dev20250101")

    options = onnx.get_default_coreml_provider_options(str(tmp_path))

    assert options is not None
    assert options["ModelFormat"] == "MLProgram"


def test_coreml_cache_is_skipped_when_disabled(
    tmp_path, onnxruntime_version, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnxruntime_version("1.22.1")
    monkeypatch.setattr(onnx, "INFERENCE_MODELS_COREML_MODEL_CACHE_ENABLED", False)

    options = onnx.get_default_coreml_provider_options(str(tmp_path))

    assert "ModelCacheDirectory" not in options
    assert options["ModelFormat"] == "MLProgram"


def test_coreml_cache_is_skipped_for_missing_package_directory(
    tmp_path, onnxruntime_version
) -> None:
    onnxruntime_version("1.22.1")

    options = onnx.get_default_coreml_provider_options(str(tmp_path / "does-not-exist"))

    assert "ModelCacheDirectory" not in options


def test_coreml_format_and_compute_units_follow_configuration(
    tmp_path, onnxruntime_version, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnxruntime_version("1.22.1")
    monkeypatch.setattr(onnx, "INFERENCE_MODELS_COREML_MODEL_FORMAT", "NeuralNetwork")
    monkeypatch.setattr(onnx, "INFERENCE_MODELS_COREML_COMPUTE_UNITS", "ALL")

    options = onnx.get_default_coreml_provider_options(str(tmp_path))

    assert options["ModelFormat"] == "NeuralNetwork"
    assert options["MLComputeUnits"] == "ALL"
    assert options["ModelCacheDirectory"].endswith("ort-1.22.1-NeuralNetwork-ALL")


def test_user_configured_coreml_provider_is_left_untouched(
    tmp_path, onnxruntime_version
) -> None:
    onnxruntime_version("1.22.1")
    user_provider = ("CoreMLExecutionProvider", {"ModelFormat": "NeuralNetwork"})

    providers = onnx.set_onnx_execution_provider_defaults(
        providers=[user_provider],
        model_package_path=str(tmp_path),
        device=torch.device("cpu"),
        default_onnx_coreml_options=True,
    )

    assert providers == [user_provider]


def test_coreml_cache_is_skipped_in_offline_mode(
    tmp_path, onnxruntime_version, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnxruntime_version("1.22.1")
    monkeypatch.setattr(onnx, "OFFLINE_MODE", True)

    options = onnx.get_default_coreml_provider_options(str(tmp_path))

    assert "ModelCacheDirectory" not in options
    assert options["ModelFormat"] == "MLProgram"


def test_coreml_cache_is_skipped_for_read_only_package_directory(
    tmp_path, onnxruntime_version, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnxruntime_version("1.22.1")
    monkeypatch.setattr(onnx.os, "access", lambda path, mode: False)

    options = onnx.get_default_coreml_provider_options(str(tmp_path))

    assert "ModelCacheDirectory" not in options


class _SessionFactory:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls = []

    def __call__(self, path_or_bytes, providers, sess_options=None):
        self.calls.append((path_or_bytes, providers, sess_options))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("Failed to create MLModel")
        return "session"


def _coreml_providers(cache_directory: str) -> list:
    return [
        (
            "CoreMLExecutionProvider",
            {"ModelFormat": "MLProgram", "ModelCacheDirectory": cache_directory},
        ),
        "CPUExecutionProvider",
    ]


def test_session_without_coreml_cache_is_created_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _SessionFactory()
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)

    session = onnx.create_onnx_inference_session(
        model_path="/models/weights.onnx", providers=["CPUExecutionProvider"]
    )

    assert session == "session"
    assert factory.calls == [("/models/weights.onnx", ["CPUExecutionProvider"], None)]


def test_session_discards_broken_coreml_cache_and_recompiles(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_directory = tmp_path / "coreml_cache" / "ort-1.22.1-MLProgram-CPUAndGPU"
    (cache_directory / "123" / "0_dynamic_mlprogram").mkdir(parents=True)
    factory = _SessionFactory(failures=1)
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)
    providers = _coreml_providers(str(cache_directory))

    session = onnx.create_onnx_inference_session(
        model_path="/models/weights.onnx", providers=providers
    )

    assert session == "session"
    assert len(factory.calls) == 2
    assert not cache_directory.exists()


def test_session_failure_without_existing_cache_is_raised(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_directory = tmp_path / "coreml_cache" / "ort-1.22.1-MLProgram-CPUAndGPU"
    factory = _SessionFactory(failures=1)
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)

    with pytest.raises(RuntimeError):
        onnx.create_onnx_inference_session(
            model_path="/models/weights.onnx",
            providers=_coreml_providers(str(cache_directory)),
        )

    assert len(factory.calls) == 1


def test_session_passes_session_options_through(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _SessionFactory()
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)
    session_options = object()

    onnx.create_onnx_inference_session(
        model_path="/models/weights.onnx",
        providers=_coreml_providers(str(tmp_path / "coreml_cache" / "variant")),
        sess_options=session_options,
    )

    assert factory.calls[0][2] is session_options
