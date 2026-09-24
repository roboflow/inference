"""WP-A04 characterization of the utilities the stream package copies.

Every behavior is pinned with concrete values first, then each stream-owned
copy is checked against the unchanged legacy helper it replaces
(`inference.core.utils.{async_utils,drawing,preprocess,environment,function}`).
The legacy helpers stay the oracle: they are separate objects, never aliases
of the copies.
"""

import ast
import asyncio
import threading
import time
import warnings
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pytest

from inference.core.exceptions import InvalidEnvironmentVariableError
from inference.core.interfaces.stream import support
from inference.core.interfaces.stream.support import async_queue, decorators
from inference.core.interfaces.stream.support import environment as stream_environment
from inference.core.interfaces.stream.support import images
from inference.core.utils import async_utils as legacy_async_utils
from inference.core.utils import drawing as legacy_drawing
from inference.core.utils import environment as legacy_environment
from inference.core.utils import function as legacy_function
from inference.core.utils import preprocess as legacy_preprocess
from inference.core.warnings import InferenceExperimentalFeatureWarning

QUEUE_CLASSES = [legacy_async_utils.Queue, async_queue.Queue]
LETTERBOX_FUNCTIONS = [legacy_preprocess.letterbox_image, images.letterbox_image]
RESIZE_FUNCTIONS = [
    legacy_preprocess.resize_image_keeping_aspect_ratio,
    images.resize_image_keeping_aspect_ratio,
]
CREATE_TILES_FUNCTIONS = [legacy_drawing.create_tiles, images.create_tiles]
STR2BOOL_FUNCTIONS = [legacy_environment.str2bool, stream_environment.str2bool]
SAFE_ENV_TO_TYPE_FUNCTIONS = [
    legacy_environment.safe_env_to_type,
    stream_environment.safe_env_to_type,
]
EXPERIMENTAL_DECORATORS = [legacy_function.experimental, decorators.experimental]


def test_copies_are_separate_objects_from_the_legacy_helpers() -> None:
    # Guards the oracle: parametrising both sides only compares anything
    # while they are distinct implementations.
    for implementations in (
        QUEUE_CLASSES,
        LETTERBOX_FUNCTIONS,
        RESIZE_FUNCTIONS,
        CREATE_TILES_FUNCTIONS,
        STR2BOOL_FUNCTIONS,
        SAFE_ENV_TO_TYPE_FUNCTIONS,
        EXPERIMENTAL_DECORATORS,
    ):
        legacy, copy = implementations
        assert legacy is not copy
        assert copy.__module__.startswith("inference.core.interfaces.stream.support")


def _start_loop_in_thread() -> Tuple[asyncio.AbstractEventLoop, threading.Thread]:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    return loop, thread


def _stop_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


# --------------------------------------------------------------------------
# Queue
# --------------------------------------------------------------------------


@pytest.mark.timeout(30)
@pytest.mark.parametrize("queue_class", QUEUE_CLASSES)
def test_queue_created_in_sync_code_runs_its_own_background_loop(
    queue_class: type,
) -> None:
    queue = queue_class(maxsize=3)

    queue.sync_put(1)
    queue.sync_put("two")
    queue.sync_put_nowait([3])

    assert queue._loop.is_running()
    assert queue._thread.daemon is True
    assert queue.sync_full() is True
    assert [queue.sync_get(), queue.sync_get_nowait(), queue.sync_get()] == [
        1,
        "two",
        [3],
    ]
    assert queue.sync_empty() is True


@pytest.mark.timeout(30)
@pytest.mark.parametrize("queue_class", QUEUE_CLASSES)
def test_queue_created_inside_running_loop_binds_that_loop(queue_class: type) -> None:
    async def scenario() -> List[Any]:
        queue = queue_class(maxsize=2)
        assert queue._loop is asyncio.get_running_loop()
        assert not hasattr(queue, "_thread")

        await queue.async_put("a")
        await queue.async_put_nowait("b")
        full = await queue.async_full()
        first = await queue.async_get()
        second = await queue.async_get_nowait()
        empty = await queue.async_empty()

        return [full, first, second, empty]

    assert asyncio.run(scenario()) == [True, "a", "b", True]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("queue_class", QUEUE_CLASSES)
def test_queue_created_for_foreign_loop_shares_it_between_sync_and_async_sides(
    queue_class: type,
) -> None:
    # The manager's WebRTC path: the loop runs in its own thread and the
    # queue is created from sync code, then used from both sides.
    loop, thread = _start_loop_in_thread()
    try:
        queue = queue_class(loop=loop, maxsize=10)
        queue.sync_put("from-sync")
        received = asyncio.run_coroutine_threadsafe(queue.async_get(), loop).result(
            timeout=5
        )
        asyncio.run_coroutine_threadsafe(queue.async_put("from-async"), loop).result(
            timeout=5
        )

        assert queue._loop is loop
        assert received == "from-sync"
        assert queue.sync_get(timeout=5) == "from-async"
    finally:
        _stop_loop(loop, thread)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("queue_class", QUEUE_CLASSES)
def test_queue_waits_time_out_with_builtin_timeout_error(queue_class: type) -> None:
    # Both implementations propagate whatever `asyncio.wait_for` raises,
    # unchanged. `asyncio.TimeoutError` *is* the builtin `TimeoutError` from
    # Python 3.11 onward, but on 3.10 it is still its own distinct type; this
    # asserts against `asyncio.TimeoutError` so the test pins the same legacy
    # behavior on every supported interpreter instead of only on 3.11+.
    queue = queue_class()

    with pytest.raises(asyncio.TimeoutError):
        queue.sync_get(timeout=0.05)

    async def scenario() -> None:
        async_queue = queue_class()
        await async_queue.async_get(timeout=0.01)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(scenario())


@pytest.mark.timeout(30)
@pytest.mark.parametrize("queue_class", QUEUE_CLASSES)
def test_queue_nowait_operations_raise_asyncio_queue_errors(queue_class: type) -> None:
    queue = queue_class(maxsize=1)

    with pytest.raises(asyncio.QueueEmpty):
        queue.sync_get_nowait()
    queue.sync_put_nowait("only")
    with pytest.raises(asyncio.QueueFull):
        queue.sync_put_nowait("overflow")

    assert queue.sync_get_nowait() == "only"


@pytest.mark.timeout(30)
@pytest.mark.parametrize("queue_class", QUEUE_CLASSES)
def test_queue_wakes_blocked_sync_consumer_and_delivers_close_sentinel(
    queue_class: type,
) -> None:
    # WebRTCVideoFrameProducer.retrieve blocks in sync_get() and treats a
    # `None` item as the end of the stream.
    queue = queue_class(maxsize=10)
    received: List[Optional[str]] = []

    def consume() -> None:
        while True:
            item = queue.sync_get()
            received.append(item)
            if item is None:
                return None

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()
    time.sleep(0.1)
    assert consumer.is_alive()

    queue.sync_put("frame-1")
    queue.sync_put(None)
    consumer.join(timeout=5)

    assert not consumer.is_alive()
    assert received == ["frame-1", None]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("queue_class", QUEUE_CLASSES)
def test_queue_sync_put_waits_for_space_on_a_full_queue(queue_class: type) -> None:
    queue = queue_class(maxsize=1)
    queue.sync_put("first")
    finished = threading.Event()

    def produce() -> None:
        queue.sync_put("second")
        finished.set()

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()

    assert not finished.wait(timeout=0.2)
    assert queue.sync_get() == "first"
    assert finished.wait(timeout=5)
    assert queue.sync_get() == "second"


@pytest.mark.timeout(30)
@pytest.mark.parametrize("queue_class", QUEUE_CLASSES)
def test_queue_wakes_blocked_async_consumer_from_another_thread(
    queue_class: type,
) -> None:
    # VideoTransformTrack awaits frames on the loop while the pipeline thread
    # produces them with sync_put.
    loop, thread = _start_loop_in_thread()
    try:
        queue = queue_class(loop=loop, maxsize=10)
        pending = asyncio.run_coroutine_threadsafe(queue.async_get(), loop)
        time.sleep(0.1)
        assert not pending.done()

        queue.sync_put(np.array([1, 2, 3]))
        result = pending.result(timeout=5)

        assert result.tolist() == [1, 2, 3]
    finally:
        _stop_loop(loop, thread)


# --------------------------------------------------------------------------
# Images: resize / letterbox
# --------------------------------------------------------------------------


def _gradient_image(height: int, width: int, channels: Optional[int] = 3) -> np.ndarray:
    size = height * width * (channels or 1)
    values = (np.arange(size) * 7 % 256).astype(np.uint8)
    if channels is None:
        return values.reshape((height, width))

    return values.reshape((height, width, channels))


@pytest.mark.parametrize("resize", RESIZE_FUNCTIONS)
@pytest.mark.parametrize(
    "shape, desired_size, expected_shape",
    [
        ((4, 4, 3), (8, 4), (4, 4, 3)),
        ((4, 4, 3), (4, 8), (4, 4, 3)),
        ((2, 6, 3), (3, 3), (1, 3, 3)),
        ((6, 2, 3), (3, 3), (3, 1, 3)),
        ((5, 7, 3), (10, 6), (6, 8, 3)),
        ((5, 7), (10, 6), (6, 8)),
    ],
)
def test_resize_keeping_aspect_ratio_output_shapes(
    resize: Callable,
    shape: Tuple[int, ...],
    desired_size: Tuple[int, int],
    expected_shape: Tuple[int, ...],
) -> None:
    image = _gradient_image(*shape[:2], channels=shape[2] if len(shape) == 3 else None)

    result = resize(image=image, desired_size=desired_size)

    assert result.shape == expected_shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("letterbox", LETTERBOX_FUNCTIONS)
def test_letterbox_pads_narrow_image_on_both_sides_with_color(
    letterbox: Callable,
) -> None:
    image = np.full((2, 2, 3), 9, dtype=np.uint8)

    result = letterbox(image=image, desired_size=(5, 2), color=(1, 2, 3))

    assert result.shape == (2, 5, 3)
    assert result[:, :1].tolist() == [[[1, 2, 3]], [[1, 2, 3]]]
    assert result[:, 1:3].tolist() == [[[9, 9, 9]] * 2] * 2
    assert result[:, 3:].tolist() == [[[1, 2, 3]] * 2] * 2


@pytest.mark.parametrize("letterbox", LETTERBOX_FUNCTIONS)
def test_letterbox_pads_wide_image_top_and_bottom(letterbox: Callable) -> None:
    image = np.full((1, 4, 3), 200, dtype=np.uint8)

    result = letterbox(image=image, desired_size=(4, 4))

    assert result.shape == (4, 4, 3)
    assert result[0].tolist() == [[0, 0, 0]] * 4
    assert result[1].tolist() == [[200, 200, 200]] * 4
    assert result[2:].tolist() == [[[0, 0, 0]] * 4] * 2


@pytest.mark.parametrize("letterbox", LETTERBOX_FUNCTIONS)
def test_letterbox_keeps_grayscale_two_dimensional(letterbox: Callable) -> None:
    image = np.full((2, 4), 50, dtype=np.uint8)

    result = letterbox(image=image, desired_size=(4, 4), color=(7, 8, 9))

    assert result.shape == (4, 4)
    assert result.tolist() == [[7] * 4, [50] * 4, [50] * 4, [7] * 4]


@pytest.mark.parametrize("letterbox", LETTERBOX_FUNCTIONS)
def test_letterbox_rejects_non_ndarray_images(letterbox: Callable) -> None:
    # Legacy only takes a tensor path under USE_PYTORCH_FOR_PREPROCESSING
    # (off here); the sinks materialise frames before letterboxing.
    assert legacy_preprocess.USE_PYTORCH_FOR_PREPROCESSING is False

    with pytest.raises(ValueError, match="Received an image of unknown type"):
        letterbox(image=[[0, 0], [0, 0]], desired_size=(4, 4))


# --------------------------------------------------------------------------
# Images: tiles
# --------------------------------------------------------------------------


@pytest.mark.parametrize("create_tiles", CREATE_TILES_FUNCTIONS)
def test_create_tiles_concrete_two_image_grid(create_tiles: Callable) -> None:
    # avg tile size: width round(avg(2, 4)) = 3, height 2; first image padded
    # on the right, second resized to 3x1 and padded at the bottom.
    images = [
        np.full((2, 2, 3), 7, dtype=np.uint8),
        np.full((2, 4, 3), 9, dtype=np.uint8),
    ]

    result = create_tiles(images=images, tile_margin=1)

    assert result.dtype == np.uint8
    assert result.tolist() == [
        [
            [7, 7, 7],
            [7, 7, 7],
            [0, 0, 0],
            [255, 255, 255],
            [9, 9, 9],
            [9, 9, 9],
            [9, 9, 9],
        ],
        [
            [7, 7, 7],
            [7, 7, 7],
            [0, 0, 0],
            [255, 255, 255],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ]


@pytest.mark.parametrize("create_tiles", CREATE_TILES_FUNCTIONS)
def test_create_tiles_fills_missing_grid_cells_with_padding_color(
    create_tiles: Callable,
) -> None:
    images = [np.full((1, 1, 3), value, dtype=np.uint8) for value in (10, 20, 30)]

    result = create_tiles(
        images=images,
        grid_size=(2, 2),
        tile_padding_color=(1, 1, 1),
        tile_margin=1,
        tile_margin_color=(5, 5, 5),
    )

    assert result[:, :, 0].tolist() == [
        [10, 5, 20],
        [5, 5, 5],
        [30, 5, 1],
    ]


@pytest.mark.parametrize("create_tiles", CREATE_TILES_FUNCTIONS)
def test_create_tiles_keeps_single_grayscale_image_two_dimensional(
    create_tiles: Callable,
) -> None:
    image = np.full((4, 4), 3, dtype=np.uint8)

    result = create_tiles(images=[image])

    assert result.shape == (4, 4)
    assert result.tolist() == image.tolist()


@pytest.mark.parametrize("create_tiles", CREATE_TILES_FUNCTIONS)
def test_create_tiles_rejects_multiple_grayscale_images(create_tiles: Callable) -> None:
    images = [np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)]

    with pytest.raises(ValueError, match="must have same number of dimensions"):
        create_tiles(images=images)


@pytest.mark.parametrize("create_tiles", CREATE_TILES_FUNCTIONS)
def test_create_tiles_error_messages(create_tiles: Callable) -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(
        ValueError, match=r"^Could not create image tiles from empty list of images\.$"
    ):
        create_tiles(images=[])
    with pytest.raises(
        ValueError,
        match=r"^Could not aggregate images shape - provided unknown mode: median\. "
        r"Supported modes: \['min', 'max', 'avg'\]\.$",
    ):
        create_tiles(images=[image], tile_scaling="median")
    with pytest.raises(
        ValueError, match=r"^Grid of size: \(1, 1\) cannot fit 2 images\.$"
    ):
        create_tiles(images=[image, image], grid_size=(1, 1))


def _tile_images(shapes: List[Tuple[int, int]]) -> List[np.ndarray]:
    return [
        _gradient_image(height, width) + np.uint8(index)
        for index, (height, width) in enumerate(shapes)
    ]


TILE_CASES = [
    pytest.param(_tile_images([(5, 5)]), {}, (5, 5, 3), id="single-square"),
    pytest.param(_tile_images([(4, 9)]), {}, (4, 9, 3), id="single-wide"),
    pytest.param(
        _tile_images([(4, 6), (8, 3), (5, 5)]), {}, (6, 45, 3), id="one-row-of-three"
    ),
    pytest.param(_tile_images([(4, 6)] * 4), {}, (23, 27, 3), id="four-negotiated-2x2"),
    pytest.param(
        _tile_images([(4, 6), (8, 3), (5, 5), (3, 9), (6, 6)]),
        {},
        (25, 48, 3),
        id="five-negotiated-2x3",
    ),
    pytest.param(
        _tile_images([(6, 4)] * 7), {}, (48, 42, 3), id="seven-negotiated-3x3"
    ),
    pytest.param(
        _tile_images([(4, 6), (8, 3), (5, 5)]),
        {"tile_scaling": "min"},
        (4, 39, 3),
        id="scaling-min",
    ),
    pytest.param(
        _tile_images([(4, 6), (8, 3), (5, 5)]),
        {"tile_scaling": "max"},
        (8, 48, 3),
        id="scaling-max",
    ),
    pytest.param(
        _tile_images([(4, 6), (8, 3), (5, 5)]),
        {"grid_size": (None, 1)},
        (48, 5, 3),
        id="grid-columns-only",
    ),
    pytest.param(
        _tile_images([(4, 6), (8, 3), (5, 5)]),
        {"grid_size": (1, None)},
        (6, 45, 3),
        id="grid-rows-only",
    ),
    pytest.param(
        _tile_images([(4, 6), (8, 3), (5, 5)]),
        {"grid_size": (3, 2)},
        (48, 25, 3),
        id="grid-with-empty-rows",
    ),
    pytest.param(
        _tile_images([(4, 6), (8, 3)]),
        {
            "single_tile_size": (10, 4),
            "tile_padding_color": (3, 4, 5),
            "tile_margin": 2,
            "tile_margin_color": (9, 8, 7),
        },
        (4, 22, 3),
        id="explicit-non-square-tile",
    ),
    pytest.param(
        _tile_images([(4, 6), (8, 3)]),
        {"tile_margin": 0},
        (6, 8, 3),
        id="no-margin",
    ),
]


@pytest.mark.parametrize("create_tiles", CREATE_TILES_FUNCTIONS)
@pytest.mark.parametrize("images, kwargs, expected_shape", TILE_CASES)
def test_create_tiles_output_shapes(
    create_tiles: Callable,
    images: List[np.ndarray],
    kwargs: dict,
    expected_shape: Tuple[int, ...],
) -> None:
    result = create_tiles(images=images, **kwargs)

    assert result.shape == expected_shape
    assert result.dtype == np.uint8


# --------------------------------------------------------------------------
# Environment parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("str2bool", STR2BOOL_FUNCTIONS)
@pytest.mark.parametrize(
    "value, expected",
    [(True, True), (False, False), ("true", True), ("TRUE", True), ("False", False)],
)
def test_str2bool_accepted_values(
    str2bool: Callable, value: Any, expected: bool
) -> None:
    assert str2bool(value) is expected


@pytest.mark.parametrize("str2bool", STR2BOOL_FUNCTIONS)
@pytest.mark.parametrize("value", ["yes", "1", "", 1, None])
def test_str2bool_rejects_other_values_with_legacy_error(
    str2bool: Callable, value: Any
) -> None:
    with pytest.raises(InvalidEnvironmentVariableError) as error:
        str2bool(value)

    assert str(error.value) == (
        "Expected a boolean environment variable (true or false) but got " f"'{value}'"
    )
    assert type(error.value).__name__ == "InvalidEnvironmentVariableError"


@pytest.mark.parametrize("safe_env_to_type", SAFE_ENV_TO_TYPE_FUNCTIONS)
def test_safe_env_to_type_reads_only_set_variables(
    safe_env_to_type: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("A04_UNSET_VARIABLE", raising=False)
    monkeypatch.setenv("A04_SET_VARIABLE", "0.25")

    assert safe_env_to_type("A04_UNSET_VARIABLE", default_value=3) == 3
    assert safe_env_to_type("A04_UNSET_VARIABLE") is None
    assert safe_env_to_type("A04_SET_VARIABLE", default_value=3) == "0.25"
    assert safe_env_to_type("A04_SET_VARIABLE", 3, float) == 0.25


@pytest.mark.parametrize("safe_env_to_type", SAFE_ENV_TO_TYPE_FUNCTIONS)
@pytest.mark.parametrize("str2bool", STR2BOOL_FUNCTIONS)
def test_safe_env_to_type_propagates_boolean_conversion_error(
    safe_env_to_type: Callable,
    str2bool: Callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("A04_BOOLEAN_VARIABLE", "maybe")

    with pytest.raises(InvalidEnvironmentVariableError):
        safe_env_to_type("A04_BOOLEAN_VARIABLE", False, str2bool)


# --------------------------------------------------------------------------
# Experimental decorator
# --------------------------------------------------------------------------


@pytest.mark.parametrize("experimental", EXPERIMENTAL_DECORATORS)
def test_experimental_warns_with_legacy_category_at_caller(
    experimental: Callable,
) -> None:
    @experimental(reason="Try at own risk.")
    def feature(value: int, *, scale: int = 2) -> int:
        """Doubles by default."""
        return value * scale

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = feature(3, scale=5)

    assert result == 15
    assert feature.__name__ == "feature"
    assert feature.__doc__ == "Doubles by default."
    assert len(caught) == 1
    assert caught[0].category is InferenceExperimentalFeatureWarning
    assert not issubclass(caught[0].category, (DeprecationWarning, UserWarning))
    assert str(caught[0].message) == "feature is experimental: Try at own risk."
    assert caught[0].filename == __file__


# --------------------------------------------------------------------------
# Copies against the legacy helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, desired_size, color",
    [
        ((4, 4, 3), (8, 4), (0, 0, 0)),
        ((5, 7, 3), (10, 6), (10, 20, 30)),
        ((7, 5, 3), (6, 10), (255, 0, 0)),
        ((3, 11, 3), (4, 4), (1, 2, 3)),
        ((11, 3, 3), (4, 4), (1, 2, 3)),
        ((16, 16, 3), (7, 5), (9, 9, 9)),
        ((5, 7), (10, 6), (40, 50, 60)),
        ((1, 1, 3), (3, 2), (4, 5, 6)),
    ],
)
def test_letterbox_copy_is_pixel_identical_to_legacy(
    shape: Tuple[int, ...],
    desired_size: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    image = _gradient_image(*shape[:2], channels=shape[2] if len(shape) == 3 else None)

    expected = legacy_preprocess.letterbox_image(
        image=image, desired_size=desired_size, color=color
    )
    actual = images.letterbox_image(image=image, desired_size=desired_size, color=color)

    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("images_to_tile, kwargs, expected_shape", TILE_CASES)
def test_create_tiles_copy_is_pixel_identical_to_legacy(
    images_to_tile: List[np.ndarray],
    kwargs: dict,
    expected_shape: Tuple[int, ...],
) -> None:
    expected = legacy_drawing.create_tiles(
        images=[image.copy() for image in images_to_tile], **kwargs
    )
    actual = images.create_tiles(
        images=[image.copy() for image in images_to_tile], **kwargs
    )

    assert actual.shape == expected_shape
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    "rows, columns, images_count",
    [
        (rows, columns, images_count)
        for rows, columns in [(1, 1), (2, 3), (3, 2), (4, 4), (1, 7)]
        for images_count in [1, 2, 5, 7]
        if images_count <= rows * columns
    ],
)
def test_tile_rows_match_legacy_create_batches(
    rows: int, columns: int, images_count: int
) -> None:
    # `_generate_tiles` slices rows where the legacy helper used
    # `inference.core.models.utils.batching.create_batches`.
    tiles = [np.full((2, 3, 3), index, dtype=np.uint8) for index in range(images_count)]
    tile_kwargs = {
        "grid_size": (rows, columns),
        "single_tile_size": (3, 2),
        "tile_padding_color": (7, 7, 7),
        "tile_margin": 1,
        "tile_margin_color": (9, 9, 9),
    }

    expected = legacy_drawing.create_tiles(images=list(tiles), **tile_kwargs)
    actual = images.create_tiles(images=list(tiles), **tile_kwargs)

    assert np.array_equal(actual, expected)


def test_create_tiles_copy_does_not_mutate_caller_list() -> None:
    tiles = [np.zeros((2, 2, 3), dtype=np.uint8)]

    images.create_tiles(images=tiles, grid_size=(1, 3))

    assert len(tiles) == 1


def test_str2bool_copy_raises_the_legacy_class_itself() -> None:
    with pytest.raises(InvalidEnvironmentVariableError) as error:
        stream_environment.str2bool("nope")

    assert type(error.value) is InvalidEnvironmentVariableError


def _imported_modules(module: Any) -> List[str]:
    tree = ast.parse(open(module.__file__, encoding="utf-8").read())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module)

    return sorted(set(names))


def test_support_modules_are_configuration_independent() -> None:
    assert _imported_modules(support) == []
    assert _imported_modules(async_queue) == ["asyncio", "threading", "typing"]
    assert _imported_modules(decorators) == [
        "functools",
        "inference.core.interfaces.stream.warnings",
        "warnings",
    ]
    assert _imported_modules(stream_environment) == [
        "inference.core.interfaces.stream.exceptions",
        "os",
        "typing",
    ]
    assert _imported_modules(images) == [
        "cv2",
        "functools",
        "itertools",
        "math",
        "numpy",
        "typing",
    ]
