import os
import shutil
from typing import Optional, Tuple

import pytest
import torch

pytest.importorskip(
    "onnxruntime",
    reason="onnxruntime is not installed (requires the onnx-* extra)",
)

from inference_models.errors import InvalidEnvVariable
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


def test_invalid_coreml_model_format_is_rejected(
    tmp_path, onnxruntime_version, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnxruntime_version("1.22.1")
    monkeypatch.setattr(onnx, "INFERENCE_MODELS_COREML_MODEL_FORMAT", "MLPackage")

    with pytest.raises(InvalidEnvVariable):
        onnx.get_default_coreml_provider_options(str(tmp_path))


def test_invalid_coreml_compute_units_are_rejected(
    tmp_path, onnxruntime_version, monkeypatch: pytest.MonkeyPatch
) -> None:
    onnxruntime_version("1.22.1")
    monkeypatch.setattr(onnx, "INFERENCE_MODELS_COREML_COMPUTE_UNITS", "GPUOnly")

    with pytest.raises(InvalidEnvVariable):
        onnx.get_default_coreml_provider_options(str(tmp_path))


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


BASE_VARIANT = "ort-1.22.1-MLProgram-CPUAndGPU"


def _package(tmp_path) -> Tuple[str, str]:
    model_path = tmp_path / "weights.onnx"
    model_path.write_bytes(b"onnx")
    return str(model_path), str(tmp_path / "coreml_cache" / BASE_VARIANT)


def _coreml_providers(cache_directory: Optional[str]) -> list:
    options = {"ModelFormat": "MLProgram"}
    if cache_directory is not None:
        options["ModelCacheDirectory"] = cache_directory
    return [("CoreMLExecutionProvider", options), "CPUExecutionProvider"]


def _used_cache_directory(call) -> str:
    return call[1][0][1]["ModelCacheDirectory"]


def test_session_without_coreml_provider_is_created_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _SessionFactory()
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)

    session = onnx.create_onnx_inference_session(
        model_path="/models/weights.onnx", providers=["CPUExecutionProvider"]
    )

    assert session == "session"
    assert factory.calls == [("/models/weights.onnx", ["CPUExecutionProvider"], None)]


def test_session_keys_coreml_cache_by_model_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, base_directory = _package(tmp_path)
    factory = _SessionFactory()
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)

    onnx.create_onnx_inference_session(
        model_path=model_path, providers=_coreml_providers(base_directory)
    )
    first = _used_cache_directory(factory.calls[0])
    os.makedirs(first)
    os.utime(model_path, ns=(1, 1))
    onnx.create_onnx_inference_session(
        model_path=model_path, providers=_coreml_providers(base_directory)
    )
    second = _used_cache_directory(factory.calls[1])

    assert first.startswith(f"{base_directory}-")
    assert second.startswith(f"{base_directory}-")
    assert first != second
    assert not os.path.exists(first)
    assert factory.calls[0][1][1] == "CPUExecutionProvider"


def test_session_keeps_caches_of_other_variants(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, base_directory = _package(tmp_path)
    other_variant = tmp_path / "coreml_cache" / "ort-1.21.1-MLProgram-CPUAndGPU-abc"
    other_variant.mkdir(parents=True)
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", _SessionFactory())

    onnx.create_onnx_inference_session(
        model_path=model_path, providers=_coreml_providers(base_directory)
    )

    assert other_variant.exists()


def test_session_compiles_under_the_package_coreml_cache_lock(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, base_directory = _package(tmp_path)
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", _SessionFactory())
    lock_paths = []
    real_file_lock = onnx.FileLock

    def recording_file_lock(path, *args, **kwargs):
        lock_paths.append(path)
        return real_file_lock(path, *args, **kwargs)

    monkeypatch.setattr(onnx, "FileLock", recording_file_lock)

    onnx.create_onnx_inference_session(
        model_path=model_path, providers=_coreml_providers(base_directory)
    )

    assert lock_paths == [str(tmp_path / ".coreml_cache.lock")]


def test_session_survives_a_cache_purge_right_before_the_lock(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, base_directory = _package(tmp_path)
    factory = _SessionFactory()
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)
    real_file_lock = onnx.FileLock

    def purge_then_lock(path, *args, **kwargs):
        # The watchdog purges coreml_cache after the loader decided to use it, before it holds the lock.
        shutil.rmtree(tmp_path / "coreml_cache", ignore_errors=True)
        return real_file_lock(path, *args, **kwargs)

    (tmp_path / "coreml_cache").mkdir()
    monkeypatch.setattr(onnx, "FileLock", purge_then_lock)

    session = onnx.create_onnx_inference_session(
        model_path=model_path, providers=_coreml_providers(base_directory)
    )

    assert session == "session"
    assert len(factory.calls) == 1
    assert _used_cache_directory(factory.calls[0]).startswith(f"{base_directory}-")


def test_session_discards_broken_coreml_cache_and_recompiles(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, base_directory = _package(tmp_path)
    probe = _SessionFactory()
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", probe)
    onnx.create_onnx_inference_session(
        model_path=model_path, providers=_coreml_providers(base_directory)
    )
    cache_directory = _used_cache_directory(probe.calls[0])
    os.makedirs(os.path.join(cache_directory, "123", "0_dynamic_mlprogram"))
    factory = _SessionFactory(failures=1)
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)

    session = onnx.create_onnx_inference_session(
        model_path=model_path, providers=_coreml_providers(base_directory)
    )

    assert session == "session"
    assert len(factory.calls) == 2
    assert _used_cache_directory(factory.calls[1]) == cache_directory
    assert not os.path.exists(cache_directory)


def test_session_falls_back_to_bare_coreml_provider_when_options_fail(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, base_directory = _package(tmp_path)
    factory = _SessionFactory(failures=1)
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)

    session = onnx.create_onnx_inference_session(
        model_path=model_path, providers=_coreml_providers(base_directory)
    )

    assert session == "session"
    assert len(factory.calls) == 2
    assert factory.calls[1][1] == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_session_falls_back_to_bare_coreml_provider_without_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _SessionFactory(failures=1)
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)

    session = onnx.create_onnx_inference_session(
        model_path="/models/weights.onnx", providers=_coreml_providers(None)
    )

    assert session == "session"
    assert factory.calls[0][1] == _coreml_providers(None)
    assert factory.calls[1][1] == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_session_failure_of_bare_coreml_provider_is_raised(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, base_directory = _package(tmp_path)
    factory = _SessionFactory(failures=2)
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)

    with pytest.raises(RuntimeError):
        onnx.create_onnx_inference_session(
            model_path=model_path, providers=_coreml_providers(base_directory)
        )

    assert len(factory.calls) == 2


def test_session_passes_session_options_through(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, base_directory = _package(tmp_path)
    factory = _SessionFactory()
    monkeypatch.setattr(onnx.onnxruntime, "InferenceSession", factory)
    session_options = object()

    onnx.create_onnx_inference_session(
        model_path=model_path,
        providers=_coreml_providers(base_directory),
        sess_options=session_options,
    )

    assert factory.calls[0][2] is session_options
