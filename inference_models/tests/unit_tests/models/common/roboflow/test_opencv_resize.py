import importlib.util
import threading
from typing import List

import cv2
import numpy as np
import pytest
import torch

from inference_models.models.common.roboflow import opencv_resize
from inference_models.models.common.roboflow.opencv_resize import (
    resize_uint8_tensor_like_opencv,
    resize_uint8_tensor_like_opencv_eager,
)

OPENCV_HAS_CAROTENE_RESIZE = "carotene" in cv2.getBuildInformation().lower()
CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = importlib.util.find_spec("triton") is not None
PARITY_SIZES = [
    ((640, 640), (448, 448)),
    ((1280, 1280), (640, 640)),
    ((486, 480), (640, 640)),
    ((1, 1), (7, 5)),
    ((9, 1), (1, 9)),
    ((2, 2), (1, 1)),
    ((375, 517), (375, 640)),
    ((375, 517), (512, 517)),
]


def _random_size_pairs(count: int, seed: int) -> List[tuple]:
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(count):
        source = tuple(int(value) for value in rng.integers(1, 700, 2))
        target = tuple(int(value) for value in rng.integers(1, 700, 2))
        pairs.append((source, target))
    for _ in range(count // 10):
        target = tuple(int(value) for value in rng.integers(1, 350, 2))
        pairs.append(((2 * target[0], 2 * target[1]), target))
    return pairs


def _opencv_resize(image: np.ndarray, target_hw: tuple) -> torch.Tensor:
    resized = cv2.resize(image, (target_hw[1], target_hw[0]))
    resized = resized.reshape(target_hw[0], target_hw[1], -1)
    return torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)


def _random_image(source_hw: tuple, channels: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, 256, (source_hw[0], source_hw[1], channels), dtype=np.uint8
    )


def _channel_counts() -> List[int]:
    if OPENCV_HAS_CAROTENE_RESIZE:
        return [2, 3]
    return [1, 2, 3, 4]


@pytest.mark.parametrize("source_hw, target_hw", PARITY_SIZES)
@pytest.mark.parametrize("channels", _channel_counts())
def test_eager_resize_matches_opencv_for_parity_sizes(
    source_hw: tuple, target_hw: tuple, channels: int
) -> None:
    # given
    image = _random_image(source_hw, channels=channels, seed=0)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    # when
    result = resize_uint8_tensor_like_opencv_eager(
        tensor, height=target_hw[0], width=target_hw[1]
    )

    # then
    assert torch.equal(result, _opencv_resize(image, target_hw))


@pytest.mark.parametrize("channels", _channel_counts())
def test_eager_resize_matches_opencv_on_random_size_sweep(channels: int) -> None:
    # given
    mismatches = []

    # when
    for index, (source_hw, target_hw) in enumerate(_random_size_pairs(300, channels)):
        image = _random_image(source_hw, channels=channels, seed=index)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        result = resize_uint8_tensor_like_opencv_eager(
            tensor, height=target_hw[0], width=target_hw[1]
        )
        if not torch.equal(result, _opencv_resize(image, target_hw)):
            mismatches.append((source_hw, target_hw))

    # then
    assert mismatches == []


def test_resize_of_cpu_batch_matches_opencv_per_image() -> None:
    # given
    images = [_random_image((97, 131), channels=3, seed=seed) for seed in range(3)]
    batch = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)

    # when
    result = resize_uint8_tensor_like_opencv(batch, height=64, width=200)

    # then
    expected = torch.cat([_opencv_resize(image, (64, 200)) for image in images])
    assert result.dtype == torch.uint8
    assert torch.equal(result, expected)


def test_resize_of_single_channel_cpu_batch_keeps_channel_dimension() -> None:
    # given
    image = _random_image((50, 40), channels=1, seed=0)
    batch = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    # when
    result = resize_uint8_tensor_like_opencv(batch, height=20, width=30)

    # then
    assert result.shape == (1, 1, 20, 30)
    assert torch.equal(result, _opencv_resize(image, (20, 30)))


@pytest.fixture
def fresh_triton_state(monkeypatch: pytest.MonkeyPatch) -> List[threading.Thread]:
    monkeypatch.setattr(opencv_resize, "_TRITON_RESIZES", {})
    monkeypatch.setattr(opencv_resize, "_TRITON_STARTED", set())
    monkeypatch.setattr(torch.cuda, "device", lambda device: _NullContext())
    verify_on_gpu = opencv_resize._verify_triton_resize
    monkeypatch.setattr(
        opencv_resize,
        "_verify_triton_resize",
        lambda triton_resize, device: verify_on_gpu(
            triton_resize, device=torch.device("cpu")
        ),
    )
    joined_threads = []
    monkeypatch.setattr(
        opencv_resize.atexit,
        "register",
        lambda join: joined_threads.append(join.__self__),
    )
    return joined_threads


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args) -> None:
        return None


def _wait_for_compilation(joined_threads: List[threading.Thread]) -> None:
    for thread in joined_threads:
        thread.join(timeout=30)


def test_triton_resize_is_used_only_after_successful_verification(
    fresh_triton_state: List[threading.Thread], monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    release_compilation = threading.Event()

    def fake_triton_resize(image: torch.Tensor, height: int, width: int):
        return resize_uint8_tensor_like_opencv_eager(image, height=height, width=width)

    def fake_load() -> object:
        release_compilation.wait(timeout=30)
        return fake_triton_resize

    monkeypatch.setattr(opencv_resize, "_load_triton_resize", fake_load)

    # when
    before_ready = opencv_resize._get_triton_resize(0)
    release_compilation.set()
    _wait_for_compilation(fresh_triton_state)
    after_ready = opencv_resize._get_triton_resize(0)

    # then
    assert before_ready is None
    assert after_ready is fake_triton_resize
    assert len(fresh_triton_state) == 1
    assert fresh_triton_state[0].daemon


def test_triton_resize_stays_eager_when_triton_is_missing(
    fresh_triton_state: List[threading.Thread], monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    def missing_triton() -> object:
        raise ModuleNotFoundError("No module named 'triton'")

    monkeypatch.setattr(opencv_resize, "_load_triton_resize", missing_triton)

    # when
    opencv_resize._get_triton_resize(0)
    _wait_for_compilation(fresh_triton_state)

    # then
    assert opencv_resize._get_triton_resize(0) is None


def test_triton_resize_stays_eager_when_compilation_fails(
    fresh_triton_state: List[threading.Thread], monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    def failing_kernel(image: torch.Tensor, height: int, width: int):
        raise RuntimeError("compilation failed")

    monkeypatch.setattr(opencv_resize, "_load_triton_resize", lambda: failing_kernel)

    # when
    opencv_resize._get_triton_resize(0)
    _wait_for_compilation(fresh_triton_state)

    # then
    assert opencv_resize._get_triton_resize(0) is None


def test_triton_resize_stays_eager_when_kernel_differs_from_eager(
    fresh_triton_state: List[threading.Thread], monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    def wrong_kernel(image: torch.Tensor, height: int, width: int):
        resized = resize_uint8_tensor_like_opencv_eager(
            image, height=height, width=width
        )
        return resized ^ 1

    monkeypatch.setattr(opencv_resize, "_load_triton_resize", lambda: wrong_kernel)

    # when
    opencv_resize._get_triton_resize(0)
    _wait_for_compilation(fresh_triton_state)

    # then
    assert opencv_resize._get_triton_resize(0) is None


def test_verification_rejects_kernel_differing_from_eager() -> None:
    # given
    def wrong_kernel(image: torch.Tensor, height: int, width: int):
        resized = resize_uint8_tensor_like_opencv_eager(
            image, height=height, width=width
        )
        return resized ^ 1

    # when
    with pytest.raises(RuntimeError):
        opencv_resize._verify_triton_resize(wrong_kernel, device=torch.device("cpu"))


def test_verification_accepts_kernel_equal_to_eager() -> None:
    # given
    def exact_kernel(image: torch.Tensor, height: int, width: int):
        return resize_uint8_tensor_like_opencv_eager(image, height=height, width=width)

    # when
    opencv_resize._verify_triton_resize(exact_kernel, device=torch.device("cpu"))


def test_triton_resize_stays_eager_when_thread_cannot_start(
    fresh_triton_state: List[threading.Thread], monkeypatch: pytest.MonkeyPatch
) -> None:
    # given
    class UnstartableThread:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(opencv_resize.threading, "Thread", UnstartableThread)

    # when
    result = opencv_resize._get_triton_resize(0)

    # then
    assert result is None
    assert fresh_triton_state == []
    assert opencv_resize._get_triton_resize(0) is None


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_eager_resize_on_cuda_matches_cpu(channels: int) -> None:
    # given
    pairs = PARITY_SIZES + _random_size_pairs(100, channels)

    # when
    mismatches = []
    for index, (source_hw, target_hw) in enumerate(pairs):
        image = _random_image(source_hw, channels=channels, seed=index)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        on_cpu = resize_uint8_tensor_like_opencv_eager(
            tensor, height=target_hw[0], width=target_hw[1]
        )
        on_cuda = resize_uint8_tensor_like_opencv_eager(
            tensor.cuda(), height=target_hw[0], width=target_hw[1]
        )
        if not torch.equal(on_cuda.cpu(), on_cpu):
            mismatches.append((source_hw, target_hw))

    # then
    assert mismatches == []


@pytest.mark.skipif(
    not (CUDA_AVAILABLE and TRITON_AVAILABLE), reason="CUDA with Triton not available"
)
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_triton_resize_matches_eager_on_cuda(channels: int) -> None:
    # given
    from inference_models.models.common.roboflow.opencv_resize_triton import (
        resize_uint8_tensor_like_opencv_triton,
    )

    pairs = PARITY_SIZES + _random_size_pairs(100, channels)

    # when
    mismatches = []
    for index, (source_hw, target_hw) in enumerate(pairs):
        image = _random_image(source_hw, channels=channels, seed=index)
        tensor = torch.from_numpy(image).cuda().permute(2, 0, 1).unsqueeze(0)
        if not opencv_resize._fits_triton_kernel(
            tensor, height=target_hw[0], width=target_hw[1]
        ):
            continue
        expected = resize_uint8_tensor_like_opencv_eager(
            tensor, height=target_hw[0], width=target_hw[1]
        )
        result = resize_uint8_tensor_like_opencv_triton(
            tensor, target_hw[0], target_hw[1]
        )
        if not torch.equal(result, expected):
            mismatches.append((source_hw, target_hw))

    # then
    assert mismatches == []


@pytest.mark.skipif(
    not (CUDA_AVAILABLE and TRITON_AVAILABLE), reason="CUDA with Triton not available"
)
def test_cuda_resize_switches_to_triton_and_stays_exact() -> None:
    # given
    image = _random_image((486, 480), channels=3, seed=0)
    tensor = torch.from_numpy(image).cuda().permute(2, 0, 1).unsqueeze(0)
    device_index = tensor.device.index

    # when
    opencv_resize._get_triton_resize(device_index)
    for thread in threading.enumerate():
        if thread.name == f"opencv-resize-triton-{device_index}":
            thread.join(timeout=300)
    result = resize_uint8_tensor_like_opencv(tensor, height=640, width=640)

    # then
    assert opencv_resize._get_triton_resize(device_index) is not None
    assert torch.equal(result.cpu(), _opencv_resize(image, (640, 640)))


@pytest.mark.parametrize(
    "shape, target_hw, expected",
    [
        ((21845, 3, 2, 2), (1, 1), True),
        ((21846, 3, 2, 2), (1, 1), False),
        ((1, 3, 1, 1), (1, 65535 * 256), True),
        ((1, 3, 1, 1), (1, 65535 * 256 + 1), False),
        ((1, 3, 5, 5), (5, 5), False),
        ((1, 2, 4, 4), (2, 2), False),
        ((1, 3, 4, 4), (2, 2), True),
        ((0, 3, 4, 4), (3, 3), False),
    ],
)
def test_triton_kernel_is_used_only_within_its_limits(
    shape: tuple, target_hw: tuple, expected: bool
) -> None:
    # given
    image = torch.empty(shape, dtype=torch.uint8)

    # when
    result = opencv_resize._fits_triton_kernel(
        image, height=target_hw[0], width=target_hw[1]
    )

    # then
    assert result is expected


def test_eager_resize_does_not_widen_unsampled_source_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    image = torch.from_numpy(_random_image((400, 600), channels=3, seed=0))
    image = image.permute(2, 0, 1).unsqueeze(0)
    widened_sizes = []
    original_to = torch.Tensor.to

    def recording_to(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        widened_sizes.append(tensor.numel())
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", recording_to)

    # when
    result = resize_uint8_tensor_like_opencv_eager(image, height=40, width=60)

    # then
    monkeypatch.undo()
    assert max(widened_sizes) == 3 * 40 * 60
    assert torch.equal(
        result, _opencv_resize(image[0].permute(1, 2, 0).numpy(), (40, 60))
    )


@pytest.mark.skipif(
    not (CUDA_AVAILABLE and TRITON_AVAILABLE), reason="CUDA with Triton not available"
)
def test_cuda_resize_beyond_triton_grid_limit_falls_back_to_eager() -> None:
    # given
    images = _random_image((2, 2), channels=3 * 21846, seed=0)
    batch = torch.from_numpy(images).permute(2, 0, 1).reshape(21846, 3, 2, 2)
    device_index = torch.cuda.current_device()
    opencv_resize._get_triton_resize(device_index)
    for thread in threading.enumerate():
        if thread.name == f"opencv-resize-triton-{device_index}":
            thread.join(timeout=300)

    # when
    result = resize_uint8_tensor_like_opencv(batch.cuda(), height=1, width=1)

    # then
    assert opencv_resize._get_triton_resize(device_index) is not None
    assert torch.equal(
        result.cpu(), resize_uint8_tensor_like_opencv(batch, height=1, width=1)
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
@pytest.mark.parametrize("use_triton", [False, True])
def test_cached_coefficients_survive_eviction_while_used_on_another_stream(
    use_triton: bool,
) -> None:
    # given
    if use_triton and not TRITON_AVAILABLE:
        pytest.skip("Triton is not available")
    if use_triton:
        from inference_models.models.common.roboflow.opencv_resize_triton import (
            resize_uint8_tensor_like_opencv_triton as resize,
        )
    else:

        def resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
            return resize_uint8_tensor_like_opencv_eager(
                image, height=height, width=width
            )

    image = torch.from_numpy(_random_image((480, 486), channels=3, seed=0))
    image = image.cuda().permute(2, 0, 1).unsqueeze(0)
    opencv_resize._opencv_linear_coefficients.cache_clear()
    expected = resize(image, 640, 643).cpu()
    side_stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    # when
    with torch.cuda.stream(side_stream):
        torch.cuda._sleep(200_000_000)
        result = resize(image, 640, 643)
    opencv_resize._opencv_linear_coefficients.cache_clear()
    garbage = [
        torch.zeros((size,), dtype=dtype, device="cuda")
        for size in (643, 640)
        for dtype in (torch.int64, torch.int64, torch.int32, torch.int32)
    ]
    torch.cuda.synchronize()

    # then
    assert len(garbage) == 8
    assert torch.equal(result.cpu(), expected)
