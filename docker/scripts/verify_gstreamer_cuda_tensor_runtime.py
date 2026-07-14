import gc
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from inference.core.interfaces.camera.gstreamer_cuda_producer import (
    GstreamerCudaVideoFrameProducer,
)
from inference.core.interfaces.camera.gstreamer_cuda_tensor_bridge import (
    NativeGstreamerCudaTensorPipeline,
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
        "nvjpegenc",
        "!",
        "filesink",
        f"location={path}",
    )


def _validate_source(path: Path, minimum_frames: int) -> None:
    producer = GstreamerCudaVideoFrameProducer(str(path), gpu_id=0)
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
    assert stats["cuda_maps"] == len(tensors)
    assert stats["host_pixel_maps"] == 0
    assert stats["host_to_device_copies"] == 0
    assert stats["device_to_host_copies"] == 0
    assert stats["stream_synchronizations"] == len(tensors)
    assert stats["active_leases"] == len(tensors)
    assert tuple(first.stride()) == (
        stats["last_channel_stride"],
        stats["last_row_stride"],
        1,
    )

    producer.release()
    assert torch.equal(first, first_snapshot)
    del first_snapshot
    del first
    tensor = None
    tensors.clear()
    gc.collect()
    torch.cuda.synchronize()


def _validate_numpy_source(path: Path) -> None:
    producer = GstreamerCudaVideoFrameProducer(str(path), gpu_id=0, output_tensor=False)
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
    baseline = GstreamerCudaVideoFrameProducer(str(path), gpu_id=0)
    baseline.discover_source_properties()
    success, first_frame = baseline.retrieve()
    assert success and first_frame is not None
    assert baseline.grab()
    success, second_frame = baseline.retrieve()
    assert success and second_frame is not None
    assert not torch.equal(first_frame, second_frame)
    baseline.release()

    producer = GstreamerCudaVideoFrameProducer(str(path), gpu_id=0)
    producer.discover_source_properties()
    assert producer.grab()
    assert producer.grab()
    success, frame_after_second_grab = producer.retrieve()
    assert success and frame_after_second_grab is not None
    assert torch.equal(frame_after_second_grab, second_frame)
    producer.release()


def _validate_interrupt_unblocks_pull() -> None:
    pipeline = NativeGstreamerCudaTensorPipeline(
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


def main() -> None:
    assert torch.cuda.is_available()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        h264_path = root / "test.h264"
        h265_path = root / "test.h265"
        jpeg_path = root / "test.jpg"
        changing_h264_path = root / "changing.h264"
        _create_h26x(h264_path, "nvh264enc", "h264parse")
        _create_h26x(h265_path, "nvh265enc", "h265parse")
        _create_h26x(changing_h264_path, "nvh264enc", "h264parse", pattern="ball")
        _create_jpeg(jpeg_path)
        _validate_source(h264_path, minimum_frames=5)
        _validate_source(h265_path, minimum_frames=5)
        _validate_source(jpeg_path, minimum_frames=1)
        _validate_numpy_source(h264_path)
        _validate_repeated_grab_advances(changing_h264_path)
        _validate_interrupt_unblocks_pull()


if __name__ == "__main__":
    main()
