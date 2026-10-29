"""``ImageData.resize(interpolation="area")`` against OpenCV ``INTER_AREA``.

V1 ``image_preprocessing`` resizes with ``cv2.INTER_AREA``. OpenCV is the
oracle here, called only through ``_cv2_area`` (host boundary of the test).
Integer laws (any axis enlarged, integer shrink) must match bit for bit; the
fractional shrink may differ by 1 only where the exact mean is a ``.5`` tie.
Integer-shrink blocks of more than 8,421,504 pixels are the deliberate
exception: OpenCV's ``int32`` sum overflows there, so the oracle is the exact
mean rounded half to even.
"""

import datetime
from fractions import Fraction

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.data.image import CompositeSource, ImageData

PATTERNS = ("constant", "checker", "ramp", "random")

EXACT_CASES = {
    "shrink 2x2": ((8, 12), (4, 6)),
    "shrink 2x2 odd width": ((14, 22), (7, 11)),
    "shrink 3x": ((9, 33), (3, 11)),
    "shrink 2x3": ((40, 60), (20, 20)),
    "shrink one axis 4x": ((8, 12), (8, 3)),
    "shrink HD 2x": ((1080, 1920), (540, 960)),
    "shrink HD 3x": ((1080, 1920), (360, 640)),
    "same size": ((7, 9), (7, 9)),
    "enlarge 2x": ((4, 5), (8, 10)),
    "enlarge 1.5x": ((4, 6), (6, 9)),
    "enlarge odd": ((5, 7), (13, 9)),
    "enlarge one axis": ((10, 10), (10, 25)),
    "mixed": ((9, 4), (4, 6)),
    "mixed HD": ((1080, 400), (640, 640)),
    "opencv rounded row position": ((54, 59), (58, 40)),
    "opencv rounded column position": ((25, 14), (1, 58)),
    "single row enlarge": ((1, 10), (1, 25)),
    "single column shrink": ((12, 1), (4, 1)),
    "single pixel enlarge": ((1, 1), (5, 7)),
}

FRACTIONAL_SHRINK_CASES = {
    "shrink 2.5x": ((10, 10), (4, 4)),
    "shrink odd": ((7, 11), (3, 4)),
    "shrink one axis": ((10, 10), (10, 4)),
    "single row shrink": ((1, 10), (1, 3)),
    "shrink HD to 640": ((1080, 1920), (640, 640)),
    "shrink HD 1.5x": ((720, 1280), (480, 853)),
}

# 255 * 8_421_504 is the largest uint8 block sum that fits int32.
INT32_SAFE_PIXEL_COUNT = 8_421_504

LARGE_BLOCK_CASES = {
    # 2160 * 4096 = 8_847_360 pixels; OpenCV 4.12 returns 0 for all 255.
    "4K-like to one pixel": ((2160, 4096), (1, 1)),
    "two large blocks side by side": ((2160, 8192), (1, 2)),
    # Single-axis blocks: the first (width) sum alone, then the second
    # (height) sum alone, holds more than the int32-safe pixel count.
    "one wide row": ((1, INT32_SAFE_PIXEL_COUNT + 1), (1, 1)),
    "one tall column": ((INT32_SAFE_PIXEL_COUNT + 1, 1), (1, 1)),
}

ACCELERATORS = [
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS unavailable"
        ),
    ),
]


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize(
    "source_hw, size_hw", list(EXACT_CASES.values()), ids=list(EXACT_CASES)
)
def test_integer_laws_match_opencv_exactly(source_hw, size_hw, pattern):
    pixels = _pattern(pattern, source_hw)

    resized = ImageData.from_tensor(pixels).resize(size_hw, interpolation="area")

    assert torch.equal(resized.tensor_image, _cv2_area(pixels, size_hw))


@pytest.mark.parametrize("channels", [1, 3])
def test_single_channel_matches_opencv(channels):
    pixels = _pattern("random", (33, 17), channels=channels)

    for size_hw in [(11, 17), (34, 17), (13, 5)]:
        resized = ImageData.from_tensor(pixels).resize(size_hw, interpolation="area")

        assert resized.channels == channels
        assert torch.equal(resized.tensor_image, _cv2_area(pixels, size_hw))


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize(
    "source_hw, size_hw",
    list(FRACTIONAL_SHRINK_CASES.values()),
    ids=list(FRACTIONAL_SHRINK_CASES),
)
def test_fractional_shrink_differs_from_opencv_only_at_rounding_ties(
    source_hw, size_hw, pattern
):
    pixels = _pattern(pattern, source_hw)

    resized = ImageData.from_tensor(pixels).resize(size_hw, interpolation="area")

    difference = (resized.tensor_image.int() - _cv2_area(pixels, size_hw).int()).abs()
    assert difference.max() <= 1
    exact_means = _exact_area_means(pixels, size_hw)[difference > 0]
    distance_to_tie = np.abs(exact_means - np.floor(exact_means) - 0.5)
    assert np.all(distance_to_tie < 1e-3)


def test_overlaps_below_opencv_threshold_are_skipped_like_opencv():
    # 1001 -> 1000 rows: the first output overlaps source row 1 by 0.001 pixel.
    pixels = _pattern("random", (1001, 8))

    resized = ImageData.from_tensor(pixels).resize((1000, 8), interpolation="area")

    assert torch.equal(resized.tensor_image, _cv2_area(pixels, (1000, 8)))


@pytest.mark.parametrize("size_hw", [(4, 6), (3, 7), (20, 30)])
def test_constant_image_stays_constant(size_hw):
    pixels = torch.full((3, 8, 12), 173, dtype=torch.uint8)

    resized = ImageData.from_tensor(pixels).resize(size_hw, interpolation="area")

    assert torch.equal(resized.tensor_image, torch.full((3, *size_hw), 173))


@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
@pytest.mark.parametrize("value", [1, 128, 254, 255])
@pytest.mark.parametrize(
    "source_hw, size_hw", list(LARGE_BLOCK_CASES.values()), ids=list(LARGE_BLOCK_CASES)
)
def test_constant_large_blocks_keep_their_value(source_hw, size_hw, value, device):
    pixels = torch.full((1, *source_hw), value, dtype=torch.uint8, device=device)

    resized = ImageData.from_tensor(pixels).resize(size_hw, interpolation="area")

    assert resized.device.type == device
    assert torch.equal(resized.tensor_image.cpu(), torch.full((1, *size_hw), value))


@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
@pytest.mark.parametrize(
    "source_hw, size_hw", list(LARGE_BLOCK_CASES.values()), ids=list(LARGE_BLOCK_CASES)
)
def test_random_large_blocks_round_their_exact_mean(source_hw, size_hw, device):
    pixels = _pattern("random", source_hw, channels=1)

    resized = ImageData.from_tensor(pixels.to(device)).resize(
        size_hw, interpolation="area"
    )

    assert torch.equal(resized.tensor_image.cpu(), _exact_block_means(pixels, size_hw))


@pytest.mark.parametrize("device", ["cpu", *ACCELERATORS])
@pytest.mark.parametrize(
    "low, expected",
    [(126, 126), (127, 128), (254, 254)],
    ids=["126.5", "127.5", "254.5"],
)
def test_large_block_tie_rounds_half_to_even(low, expected, device):
    # Half the pixels low, half low + 1: the exact mean is low + 0.5.
    pixels = torch.full((1, 2160, 4096), low, dtype=torch.uint8)
    pixels[:, :1080] += 1

    resized = ImageData.from_tensor(pixels.to(device)).resize(
        (1, 1), interpolation="area"
    )

    assert _exact_block_means(pixels, (1, 1)).item() == expected
    assert resized.tensor_image.item() == expected


@pytest.mark.parametrize("pattern", ["constant", "random"])
@pytest.mark.parametrize(
    "source_hw",
    [(2048, 2048), (2688, 3133)],
    ids=["below int32 bound", "at int32 bound"],
)
def test_blocks_up_to_int32_bound_match_opencv_exactly(source_hw, pattern):
    # 2688 * 3133 = INT32_SAFE_PIXEL_COUNT: OpenCV's int32 sum still fits.
    pixels = _pattern(pattern, source_hw, channels=1)
    if pattern == "constant":
        pixels.fill_(255)

    resized = ImageData.from_tensor(pixels).resize((1, 1), interpolation="area")

    assert torch.equal(resized.tensor_image, _cv2_area(pixels, (1, 1)))
    assert torch.equal(resized.tensor_image, _exact_block_means(pixels, (1, 1)))


def test_shrinking_a_checkerboard_averages_instead_of_sampling():
    pixels = _pattern("checker", (8, 12))

    halved = ImageData.from_tensor(pixels).resize((4, 6), interpolation="area")
    fractional = ImageData.from_tensor(pixels).resize((3, 5), interpolation="area")

    # (0 + 255 + 255 + 0) / 4 = 127.5 rounds half up on the 2x2 path.
    assert torch.equal(halved.tensor_image, torch.full((3, 4, 6), 128))
    assert fractional.tensor_image.min() >= 100
    assert fractional.tensor_image.max() <= 155


def test_shrinking_a_ramp_keeps_it_monotonic():
    pixels = _pattern("ramp", (6, 100))

    resized = ImageData.from_tensor(pixels).resize((6, 37), interpolation="area")

    steps = resized.tensor_image[:, :, 1:].int() - resized.tensor_image[:, :, :-1].int()
    assert steps.min() >= 0


@pytest.mark.parametrize(
    "size_hw", [(20, 30), (13, 17), (60, 50)], ids=["integer", "fractional", "enlarge"]
)
def test_non_contiguous_pixels_resize_like_contiguous_pixels(size_hw):
    channels_last = _pattern("random", (40, 60)).permute(1, 2, 0).contiguous()
    permuted = channels_last.permute(2, 0, 1)
    strided = _pattern("random", (80, 180))[:, ::2, 1::3]
    assert not permuted.is_contiguous() and not strided.is_contiguous()

    for pixels in (permuted, strided):
        resized = ImageData.from_tensor(pixels).resize(size_hw, interpolation="area")
        expected = ImageData.from_tensor(pixels.contiguous()).resize(
            size_hw, interpolation="area"
        )

        assert torch.equal(resized.tensor_image, expected.tensor_image)


def test_result_is_a_new_uint8_tensor_even_at_same_size():
    pixels = _pattern("random", (7, 9))

    resized = ImageData.from_tensor(pixels).resize((7, 9), interpolation="area")

    assert resized.tensor_image.dtype == torch.uint8
    assert resized.tensor_image.data_ptr() != pixels.data_ptr()
    assert torch.equal(resized.tensor_image, pixels)


def test_area_keeps_resize_geometry_and_provenance():
    video = VideoMetadata(
        video_identifier="camera-3",
        frame_number=41,
        frame_timestamp=datetime.datetime(
            2026, 10, 12, 8, 30, tzinfo=datetime.timezone.utc
        ),
        fps=25.0,
    )
    root = ImageData.from_tensor(
        _pattern("random", (40, 60)), image_id="root", video_metadata=video
    )
    crop = root.crop((10, 4, 50, 34), image_id="crop")

    area = crop.resize((15, 16), interpolation="area", image_id="small")
    bilinear = crop.resize((15, 16), image_id="small")

    assert area.size_hw == (15, 16)
    assert area.image_id == "small"
    assert area.parent == bilinear.parent
    assert area.root == bilinear.root
    assert area.parent.frame_id == "crop"
    assert area.parent.scale_xy == (40 / 16, 30 / 15)
    assert area.root.frame_id == "root"
    assert area.root.map_xy(16, 15) == (50.0, 34.0)
    assert area.video_metadata is video
    assert area.prediction_metadata() == bilinear.prediction_metadata()
    assert crop.resize((15, 16), interpolation="area").image_id.startswith("resize-")


def test_area_keeps_composite_sources():
    tile = ImageData.from_tensor(_pattern("random", (20, 20)), image_id="tile")
    canvas = ImageData.composite(
        _pattern("random", (20, 40)),
        sources=[CompositeSource.place(tile, index=(0,), canvas_xyxy=(0, 0, 20, 20))],
        image_id="mosaic",
    )

    resized = canvas.resize((10, 20), interpolation="area")

    assert resized.is_composite
    assert resized.composite_sources == canvas.composite_sources


def test_default_interpolation_is_still_bilinear():
    image = ImageData.from_tensor(_pattern("checker", (10, 10)))

    default = image.resize((4, 4))
    bilinear = image.resize((4, 4), interpolation="bilinear")

    assert torch.equal(default.tensor_image, bilinear.tensor_image)


def test_unknown_interpolation_lists_area():
    image = ImageData.from_tensor(_pattern("constant", (4, 4)))

    with pytest.raises(ValueError, match="'area'"):
        image.resize((2, 2), interpolation="cubic")


@pytest.mark.parametrize("device", ACCELERATORS)
@pytest.mark.parametrize(
    "source_hw, size_hw",
    [((1080, 1920), (540, 960)), ((1080, 400), (640, 640)), ((54, 59), (58, 40))],
    ids=["integer shrink", "mixed", "opencv rounded position"],
)
def test_accelerator_integer_laws_match_cpu_and_stay_on_device(
    device, source_hw, size_hw
):
    pixels = _pattern("random", source_hw)

    on_device = ImageData.from_tensor(pixels.to(device)).resize(
        size_hw, interpolation="area"
    )

    assert on_device.device.type == device
    assert torch.equal(on_device.tensor_image.cpu(), _cv2_area(pixels, size_hw))


@pytest.mark.parametrize("device", ACCELERATORS)
def test_accelerator_fractional_shrink_stays_within_rounding_of_cpu(device):
    pixels = _pattern("random", (720, 1280))
    on_cpu = ImageData.from_tensor(pixels).resize((480, 853), interpolation="area")

    on_device = ImageData.from_tensor(pixels.to(device)).resize(
        (480, 853), interpolation="area"
    )

    assert on_device.device.type == device
    difference = (on_device.tensor_image.cpu().int() - on_cpu.tensor_image.int()).abs()
    assert difference.max() <= 1


def _pattern(name, size_hw, *, channels=3):
    height, width = size_hw
    rows = torch.arange(height)[:, None]
    columns = torch.arange(width)[None, :]
    if name == "random":
        generator = torch.Generator().manual_seed(height * 7919 + width)
        pixels = torch.randint(
            0, 256, (channels, height, width), generator=generator, dtype=torch.uint8
        )

        return pixels

    if name == "constant":
        plane = torch.full((height, width), 173)
    elif name == "checker":
        plane = (rows + columns) % 2 * 255
    else:
        plane = (columns * 255 // max(width - 1, 1)).expand(height, width)
    pixels = plane.to(torch.uint8).expand(channels, height, width).contiguous()

    return pixels


def _cv2_area(pixels, size_hw):
    # Host oracle boundary: V1 calls cv2.resize on an HWC uint8 array.
    channels = pixels.shape[0]
    hwc = pixels.permute(1, 2, 0).contiguous().numpy()
    resized = cv2.resize(
        hwc, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA
    ).reshape(size_hw[0], size_hw[1], channels)
    expected = torch.from_numpy(resized).permute(2, 0, 1)

    return expected


def _exact_block_means(pixels, size_hw):
    # Mathematical integer-shrink oracle: exact block mean, Python's
    # half-to-even round of a Fraction; block sums in host int64.
    channels, source_height, source_width = pixels.shape
    factor_y, factor_x = source_height // size_hw[0], source_width // size_hw[1]
    values = pixels.numpy().astype(np.int64)
    blocks = values.reshape(channels, size_hw[0], factor_y, size_hw[1], factor_x)
    sums = blocks.sum(axis=(2, 4))
    means = [round(Fraction(int(total), factor_y * factor_x)) for total in sums.flat]
    expected = torch.tensor(means, dtype=torch.uint8).reshape(channels, *size_hw)

    return expected


def _exact_area_means(pixels, size_hw):
    # float64 area average: each output pixel is the mean over its interval.
    rows = _overlap_matrix(pixels.shape[1], size_hw[0])
    columns = _overlap_matrix(pixels.shape[2], size_hw[1])
    means = rows @ pixels.double().numpy() @ columns.T

    return means


def _overlap_matrix(source, target):
    output_edges = np.arange(target + 1) * source / target
    source_starts = np.arange(source)[None, :]
    overlap = np.minimum(output_edges[1:, None], source_starts + 1) - np.maximum(
        output_edges[:-1, None], source_starts
    )
    matrix = np.clip(overlap, 0, None) * target / source

    return matrix
