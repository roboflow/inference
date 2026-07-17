"""On-device Jetson tensor-runtime check. No CI job runs this: it needs a
Jetson with the BSP media stack, torch, and the inference package installed.
Run it manually inside a Jetson image, e.g.:

    docker run --rm --runtime nvidia --entrypoint python3 \
        -v "$PWD/docker/scripts/verify_jetson_tensor_runtime.py:/tmp/verify.py:ro" \
        <jetson image> /tmp/verify.py
"""

import gc
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from inference.core.interfaces.camera.jetson_producer import JetsonVideoFrameProducer
from inference.core.interfaces.camera.jetson_tensor_bridge import (
    NativeJetsonTensorPipeline,
)


def _run_gstreamer(*arguments: str) -> None:
    subprocess.run(
        ["gst-launch-1.0", "-q", *arguments],
        check=True,
        timeout=60,
    )


def _create_h26x(path: Path, encoder: str, parser: str, pattern: str = "red") -> None:
    _run_gstreamer(
        "videotestsrc",
        "num-buffers=8",
        f"pattern={pattern}",
        "!",
        "video/x-raw,format=I420,width=320,height=180,framerate=30/1",
        "!",
        "nvvidconv",
        "!",
        "video/x-raw(memory:NVMM),format=NV12",
        "!",
        encoder,
        "!",
        parser,
        "!",
        "filesink",
        f"location={path}",
    )


def _create_jpeg(path: Path) -> None:
    _run_gstreamer(
        "videotestsrc",
        "num-buffers=1",
        "pattern=red",
        "!",
        "video/x-raw,format=I420,width=320,height=180,framerate=1/1",
        "!",
        "nvvidconv",
        "!",
        "video/x-raw(memory:NVMM),format=I420",
        "!",
        "nvjpegenc",
        "!",
        "filesink",
        f"location={path}",
    )


def _validate_source(path: Path, minimum_frames: int) -> None:
    producer = JetsonVideoFrameProducer(str(path), output_tensor=True)
    properties = producer.discover_source_properties()
    assert properties.width == 320
    assert properties.height == 180

    success, first = producer.retrieve()
    assert success and first is not None
    assert first.is_cuda
    assert first.dtype == torch.uint8
    assert tuple(first.shape) == (3, 180, 320)
    assert first.device.index == 0
    assert first.data_ptr() != 0
    assert first.is_contiguous()

    channel_means = first.float().mean(dim=(1, 2))
    assert channel_means[0] > channel_means[1] + 80
    assert channel_means[0] > channel_means[2] + 80
    first_snapshot = first.clone()
    tensors: List[torch.Tensor] = [first]
    tensor = None

    while len(tensors) < 5 and producer.grab():
        success, tensor = producer.retrieve()
        assert success and tensor is not None
        assert tensor.is_cuda
        assert tuple(tensor.shape) == (3, 180, 320)
        tensors.append(tensor)

    assert len(tensors) >= minimum_frames
    stats = producer.tensor_bridge_stats
    assert stats["frames"] == len(tensors)
    assert stats["descriptor_maps"] == len(tensors)
    assert stats["host_pixel_maps"] == 0
    assert stats["host_to_device_copies"] == 0
    assert stats["device_to_host_copies"] == 0
    assert stats["array_flatten_copies"] == 0
    assert stats["conversion_kernels"] == len(tensors)
    assert stats["nvmm_frames"] == len(tensors)

    producer.release()
    assert torch.equal(first, first_snapshot)
    del first_snapshot
    del first
    tensor = None
    tensors.clear()
    gc.collect()
    torch.cuda.synchronize()


def _validate_numpy_source(path: Path) -> None:
    producer = JetsonVideoFrameProducer(str(path), output_tensor=False)
    properties = producer.discover_source_properties()
    assert properties.width == 320
    assert properties.height == 180

    success, image = producer.retrieve()
    assert success and image is not None
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert image.shape == (180, 320, 3)
    assert image.flags.c_contiguous
    channel_means = image.mean(axis=(0, 1))
    assert channel_means[2] > channel_means[1] + 80
    assert channel_means[2] > channel_means[0] + 80
    producer.release()


def _validate_repeated_grab_advances(path: Path) -> None:
    baseline = JetsonVideoFrameProducer(str(path), output_tensor=True)
    baseline.discover_source_properties()
    success, first_frame = baseline.retrieve()
    assert success and first_frame is not None
    assert baseline.grab()
    success, second_frame = baseline.retrieve()
    assert success and second_frame is not None
    assert not torch.equal(first_frame, second_frame)
    baseline.release()

    producer = JetsonVideoFrameProducer(str(path), output_tensor=True)
    producer.discover_source_properties()
    assert producer.grab()
    assert producer.grab()
    success, frame_after_second_grab = producer.retrieve()
    assert success and frame_after_second_grab is not None
    assert torch.equal(frame_after_second_grab, second_frame)
    producer.release()


def _validate_threaded_retrieve(path: Path) -> None:
    # Mirrors VideoSource's threading: the producer is constructed (and
    # discovery runs) on the caller thread while retrieve happens on a
    # dedicated consumer thread that has made no prior CUDA call. This is the
    # configuration production uses and single-threaded checks miss.
    producer = JetsonVideoFrameProducer(str(path), output_tensor=True)
    producer.discover_source_properties()
    result = {}

    def consume() -> None:
        try:
            success, tensor = producer.retrieve()
            result["success"] = success
            result["shape"] = tuple(tensor.shape)
        except Exception as error:  # noqa: BLE001 - surfaced via assert below
            result["error"] = error

    thread = threading.Thread(target=consume)
    thread.start()
    thread.join(timeout=30.0)

    assert not thread.is_alive()
    assert "error" not in result, f"retrieve failed off-thread: {result.get('error')!r}"
    assert result["success"]
    assert result["shape"] == (3, 180, 320)
    producer.release()


def _validate_interrupt_unblocks_pull() -> None:
    pipeline = NativeJetsonTensorPipeline(
        "appsrc is-live=true ! appsink name=rf_tensor_sink wait-on-eos=false",
        device_id=0,
    )
    entered_grab = threading.Event()
    result = {}

    def pull_sample() -> None:
        entered_grab.set()
        result["grabbed"] = pipeline.grab()

    thread = threading.Thread(target=pull_sample)
    thread.start()
    assert entered_grab.wait(timeout=1.0)
    time.sleep(0.1)
    started = time.monotonic()
    pipeline.interrupt()
    thread.join(timeout=2.0)
    elapsed = time.monotonic() - started

    assert not thread.is_alive()
    assert result["grabbed"] is False
    assert elapsed < 2.0
    pipeline.close()


def _validate_live_rtsp_source(url: str) -> None:
    """Validate the production RTSP -> NVDEC -> CUDA-tensor path.

    This is deliberately opt-in because it needs a reachable camera.  It is
    stronger than a gst-launch smoke test: it exercises the Python producer,
    native NVMM bridge, and DLPack handoff used by VideoSource in production.
    """

    raw_minimum_frames = os.getenv("ROBOFLOW_JETSON_TEST_RTSP_MINIMUM_FRAMES", "10")
    try:
        minimum_frames = int(raw_minimum_frames)
    except ValueError as error:
        raise ValueError(
            "ROBOFLOW_JETSON_TEST_RTSP_MINIMUM_FRAMES must be an integer"
        ) from error
    if minimum_frames < 1:
        raise ValueError(
            "ROBOFLOW_JETSON_TEST_RTSP_MINIMUM_FRAMES must be at least one"
        )

    producer = JetsonVideoFrameProducer(url, output_tensor=True)
    try:
        properties = producer.discover_source_properties()
        assert properties.width > 0 and properties.height > 0

        tensors: List[torch.Tensor] = []
        while len(tensors) < minimum_frames:
            success, tensor = producer.retrieve()
            assert success and tensor is not None
            assert tensor.is_cuda
            assert tensor.dtype == torch.uint8
            assert tuple(tensor.shape) == (
                3,
                properties.height,
                properties.width,
            )
            tensors.append(tensor)
            if len(tensors) < minimum_frames:
                assert producer.grab()

        assert producer._native_pipeline.has_factory("nvv4l2decoder")
        stats = producer.tensor_bridge_stats
        assert stats["frames"] >= minimum_frames
        assert stats["nvmm_frames"] >= minimum_frames
        assert stats["host_pixel_maps"] == 0
        assert stats["host_to_device_copies"] == 0
        assert stats["device_to_host_copies"] == 0
        assert stats["array_flatten_copies"] == 0
        print(
            "RTSP_TENSOR_PROBE_OK "
            f"frames={minimum_frames} size={properties.width}x{properties.height}"
        )
    finally:
        producer.release()

    # Production arrives through VideoSource, not by constructing the producer
    # directly.  Assert that first-frame discovery did not silently replace the
    # Jetson implementation with the OpenCV fallback.
    from inference.core.interfaces.camera.video_source import VideoSource

    video_source = VideoSource.init(video_reference=url, allow_tensor_frames=True)
    try:
        video_source.start()
        assert isinstance(video_source._video, JetsonVideoFrameProducer)
        deadline = time.monotonic() + 30.0
        frame = None
        while time.monotonic() < deadline:
            if video_source._video.grab():
                success, frame = video_source._video.retrieve()
                if success and frame is not None:
                    break
            time.sleep(0.1)
        assert frame is not None
        assert isinstance(frame, torch.Tensor)
        assert frame.is_cuda
        print("RTSP_VIDEO_SOURCE_TENSOR_PROBE_OK")
    finally:
        video_source.terminate()


def main() -> None:
    assert torch.cuda.is_available()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        h264_path = root / "test.h264"
        h265_path = root / "test.h265"
        jpeg_path = root / "test.jpg"
        changing_h264_path = root / "changing.h264"
        _create_h26x(h264_path, "nvv4l2h264enc", "h264parse")
        _create_h26x(h265_path, "nvv4l2h265enc", "h265parse")
        _create_h26x(
            changing_h264_path,
            "nvv4l2h264enc",
            "h264parse",
            pattern="ball",
        )
        _create_jpeg(jpeg_path)
        _validate_source(h264_path, minimum_frames=5)
        _validate_source(h265_path, minimum_frames=5)
        _validate_source(jpeg_path, minimum_frames=1)
        _validate_numpy_source(h264_path)
        _validate_repeated_grab_advances(changing_h264_path)
        _validate_threaded_retrieve(h264_path)
        _validate_interrupt_unblocks_pull()

    rtsp_url = os.getenv("ROBOFLOW_JETSON_TEST_RTSP_URL")
    if rtsp_url:
        _validate_live_rtsp_source(rtsp_url)


if __name__ == "__main__":
    main()
