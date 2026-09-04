import sys
from pathlib import Path
from types import SimpleNamespace

PROCESSOR_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "processor"
)
sys.path.insert(0, str(PROCESSOR_DIR))

import inference_runtime_compat  # noqa: E402


def test_legacy_runtime_uses_regular_serializer_when_tensor_flag_is_missing(
    monkeypatch,
):
    regular_serializer = object()
    requested = []

    def fake_import(name):
        requested.append(name)
        if name == "inference.core.env":
            return SimpleNamespace()
        if name.endswith(".serializers"):
            return SimpleNamespace(serialize_wildcard_kind=regular_serializer)
        raise AssertionError(f"legacy runtime must not import {name}")

    monkeypatch.setattr(
        inference_runtime_compat.importlib, "import_module", fake_import
    )

    enabled, serializer = inference_runtime_compat.resolve_workflow_serializer()

    assert enabled is False
    assert serializer is regular_serializer
    assert not any(name.endswith("serializers_tensor") for name in requested)


def test_v14_runtime_uses_tensor_serializer_when_effective_flag_is_enabled(
    monkeypatch,
):
    tensor_serializer = object()

    def fake_import(name):
        if name == "inference.core.env":
            return SimpleNamespace(ENABLE_TENSOR_DATA_REPRESENTATION=True)
        if name.endswith(".serializers_tensor"):
            return SimpleNamespace(serialize_wildcard_kind=tensor_serializer)
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(
        inference_runtime_compat.importlib, "import_module", fake_import
    )

    enabled, serializer = inference_runtime_compat.resolve_workflow_serializer()

    assert enabled is True
    assert serializer is tensor_serializer


def test_freshest_mode_capability_follows_pipeline_signature():
    class LegacyPipeline:
        @classmethod
        def init_with_workflow(cls, video_reference, decoding_buffer_size=16):
            pass

    class V14Pipeline:
        @classmethod
        def init_with_workflow(
            cls,
            video_reference,
            decoding_buffer_size=16,
            video_processing_mode=None,
        ):
            pass

    assert not inference_runtime_compat.pipeline_supports_freshest_mode(LegacyPipeline)
    assert inference_runtime_compat.pipeline_supports_freshest_mode(V14Pipeline)
