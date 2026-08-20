import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROCESSOR_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "processor"
)
sys.path.insert(0, str(PROCESSOR_DIR))

from video_ingest import (  # noqa: E402
    GSTREAMER_CUDA_INGEST,
    PYAV_INGEST,
    build_cuda_producer,
    process_runtime_identity,
    producer_runtime_identity,
    resolve_video_ingest_mode,
    verify_cuda_frame,
)


def test_pyav_is_the_unchanged_default(monkeypatch):
    monkeypatch.delenv("PROCESSOR_VIDEO_INGEST_MODE", raising=False)
    monkeypatch.delenv("ENABLE_TENSOR_DATA_REPRESENTATION", raising=False)

    assert resolve_video_ingest_mode() == PYAV_INGEST


def test_cuda_ingest_requires_tensor_mode_at_process_start(monkeypatch):
    monkeypatch.setenv("PROCESSOR_VIDEO_INGEST_MODE", GSTREAMER_CUDA_INGEST)
    monkeypatch.delenv("ENABLE_TENSOR_DATA_REPRESENTATION", raising=False)

    with pytest.raises(ValueError, match="requires.*TENSOR_DATA_REPRESENTATION"):
        resolve_video_ingest_mode()

    monkeypatch.setenv("ENABLE_TENSOR_DATA_REPRESENTATION", "true")
    assert resolve_video_ingest_mode() == GSTREAMER_CUDA_INGEST


def test_legacy_runtime_rejects_cuda_even_if_tensor_env_was_left_set(monkeypatch):
    monkeypatch.setenv("PROCESSOR_VIDEO_INGEST_MODE", GSTREAMER_CUDA_INGEST)
    monkeypatch.setenv("ENABLE_TENSOR_DATA_REPRESENTATION", "true")

    with pytest.raises(ValueError, match="requires.*TENSOR_DATA_REPRESENTATION"):
        resolve_video_ingest_mode(tensor_runtime_available=False)


def test_unknown_ingest_mode_fails_before_the_worker_claims_jobs(monkeypatch):
    monkeypatch.setenv("PROCESSOR_VIDEO_INGEST_MODE", "best-effort-auto")

    with pytest.raises(ValueError, match="must be one of"):
        resolve_video_ingest_mode()


def test_runtime_identity_is_bounded_and_contains_no_credentials(monkeypatch):
    monkeypatch.setenv("ENABLE_TENSOR_DATA_REPRESENTATION", "yes")
    monkeypatch.setenv("ROBOFLOW_RTSP_LATENCY_MS", "80")
    monkeypatch.setenv("ROBOFLOW_API_KEY", "must-not-leak")

    runtime = process_runtime_identity(GSTREAMER_CUDA_INGEST)

    assert runtime == {
        "videoIngestMode": GSTREAMER_CUDA_INGEST,
        "tensorRepresentationEnabled": True,
        "rtspLatencyMs": 80,
    }
    assert "must-not-leak" not in str(runtime)


def test_legacy_runtime_identity_does_not_claim_tensor_support(monkeypatch):
    monkeypatch.setenv("ENABLE_TENSOR_DATA_REPRESENTATION", "true")

    runtime = process_runtime_identity(PYAV_INGEST, tensor_runtime_available=False)

    assert runtime["tensorRepresentationEnabled"] is False


def test_cuda_factory_constructs_tensor_producer_and_reports_it(monkeypatch):
    created = []

    class FakeCudaProducer:
        def __init__(self, video_reference, output_tensor):
            self.video_reference = video_reference
            self.output_tensor = output_tensor
            self.tensor_bridge_stats = {"zeroCopyFrames": 7}

    module_name = "inference.core.interfaces.camera.gstreamer_cuda_producer"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        SimpleNamespace(GstreamerCudaVideoFrameProducer=FakeCudaProducer),
    )

    producer = build_cuda_producer("rtsp://relay/source", created.append)

    assert producer.video_reference == "rtsp://relay/source"
    assert producer.output_tensor is True
    assert created == [producer]
    assert producer_runtime_identity(producer) == {
        "videoProducer": "FakeCudaProducer",
        "tensorBridge": {"zeroCopyFrames": 7},
    }


def test_producer_stats_drop_unbounded_or_non_numeric_values():
    producer = SimpleNamespace(
        tensor_bridge_stats={
            "frames": 4,
            "ratio": 0.5,
            "healthy": True,
            "secret/value": "do-not-return",
            "nan": float("nan"),
        }
    )

    runtime = producer_runtime_identity(producer)

    assert runtime["tensorBridge"] == {"frames": 4, "ratio": 0.5}


def test_pyav_stream_identity_records_exact_bounded_fixture_metadata():
    producer = SimpleNamespace(
        source_stream_metadata={
            "width": 3840,
            "height": 2160,
            "fps": 59.94005994005994,
            "fpsNumerator": 60000,
            "fpsDenominator": 1001,
            "codec": "h264",
            "url": "rtsp://secret@relay/source",
        }
    )

    runtime = producer_runtime_identity(producer)

    assert runtime["sourceStream"] == {
        "width": 3840,
        "height": 2160,
        "fps": 59.94005994005994,
        "fpsNumerator": 60000,
        "fpsDenominator": 1001,
        "codec": "h264",
    }
    assert "secret" not in str(runtime)


def test_cuda_frame_verification_rejects_host_fallback():
    verify_cuda_frame(SimpleNamespace(is_cuda=True))
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        verify_cuda_frame(SimpleNamespace(is_cuda=False))


def test_processor_uses_fail_loud_cuda_and_freshest_frame_mode():
    source = (PROCESSOR_DIR / "processor.py").read_text()

    assert "build_cuda_producer(" in source
    assert '"video_processing_mode": "freshest"' in source
    assert '"decoding_buffer_size": 1' in source
    assert "BufferFillingStrategy.DROP_OLDEST" in source
    assert "discover_hardware_video_frame_producer" not in source


def test_processor_selects_tensor_serializer_and_materializes_at_sink_only():
    source = (PROCESSOR_DIR / "processor.py").read_text()

    assert "resolve_workflow_serializer()" in source
    assert "INFERENCE_PIPELINE_SUPPORTS_FRESHEST_MODE" in source
    assert "self.raw_frames.set(key, value)" in source
    assert "image = workflow_image.numpy_image" in source


@pytest.mark.parametrize("dockerfile", ("Dockerfile", "Dockerfile.overlay"))
def test_processor_images_include_ingest_selector(dockerfile):
    source = (PROCESSOR_DIR / dockerfile).read_text()

    assert "COPY video_ingest.py /app/video_ingest.py" in source
    assert "COPY file_replay.py /app/file_replay.py" in source
    assert "COPY inference_runtime_compat.py /app/inference_runtime_compat.py" in source
    assert "COPY run_lifecycle.py /app/run_lifecycle.py" in source


def test_full_processor_restores_pip_removed_from_v14_runtime():
    source = (PROCESSOR_DIR / "Dockerfile").read_text()

    assert "python3-pip" in source
    assert "python3 -m pip install" in source


def test_nvdec_cloud_build_uses_full_immutable_base_seam():
    source = (PROCESSOR_DIR / "cloudbuild.nvdec.yaml").read_text()

    assert "--file=Dockerfile" in source
    assert "--build-arg=BASE_IMAGE=${_BASE_IMAGE}" in source
    assert "--build-arg=VIDEO_PROC_GIT_SHA=${_GIT_SHA}" in source
    assert "VIDEO_PROC_RUNTIME_VARIANT=nvdec-tensor-v1.4" in source
    assert "Dockerfile.overlay" not in source
