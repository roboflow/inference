"""``ObjectCache`` and the cache blocks, called directly.

Blocks run inside a hand-built call context, so the namespace rule can be
checked without compiling a workflow::

    scope="global"  ──▶ GLOBAL_NAMESPACE
    scope="source"  ──▶ video_identifier ──▶ SampleContext.source_id ──▶ "default"
"""

import threading
from datetime import datetime
from typing import Any, Dict, Optional

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.state.object_cache import (
    DEFAULT_SOURCE,
    GLOBAL_NAMESPACE,
    CacheGetBlock,
    CacheNamespace,
    CacheSetBlock,
    ObjectCache,
)
from roboflow_workflows.v2.data.batch import SampleContext
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.context import (
    ExecutionContext,
    use_execution_context,
)

DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "mps",
        id="mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="needs MPS"
        ),
    ),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
]
CAM_A = CacheNamespace(scope="source", source="cam-a")
CAM_B = CacheNamespace(scope="source", source="cam-b")
MISSING = object()


class _Scope:
    """Call scope answering one sample context for the call's index."""

    def __init__(self, sample: Optional[SampleContext]) -> None:
        self._sample = sample

    def sample_at(self, index, *, field=None):
        return self._sample


def _image(
    value: int = 0, *, video: Optional[str] = None, device: str = "cpu"
) -> ImageData:
    pixels = torch.full((3, 4, 4), value, dtype=torch.uint8, device=device)
    metadata = None
    if video is not None:
        metadata = VideoMetadata(
            video_identifier=video, frame_number=1, frame_timestamp=datetime(2026, 1, 1)
        )
    image = ImageData.from_tensor(pixels, video_metadata=metadata)

    return image


def _call(block: Any, *, source: Optional[str] = None, **params: Any) -> Dict:
    sample = SampleContext(source_id=source) if source is not None else None
    context = ExecutionContext(
        step_path=("cache",),
        block_type=block.type,
        session_id="session",
        run_id="run",
        indices=((0,),),
        call_scope=_Scope(sample),
    )
    with use_execution_context(context):
        result = block.run(**params)

    return result


def _set(cache: ObjectCache, *, image: ImageData, key: str, value: Any, **options):
    block = CacheSetBlock(object_cache=cache)
    params = {"scope": "source", "after": None, **options}
    source = params.pop("source", None)
    result = _call(block, source=source, image=image, key=key, value=value, **params)

    return result["output"]


def _get(cache: ObjectCache, *, image: ImageData, key: str, **options) -> Any:
    block = CacheGetBlock(object_cache=cache)
    params = {"scope": "source", "after": None, **options}
    source = params.pop("source", None)
    result = _call(block, source=source, image=image, key=key, **params)

    return result["output"]


# ObjectCache


def test_cache_returns_the_stored_object_and_keeps_a_stored_none_distinct():
    cache = ObjectCache()
    value = {"count": 1}
    cache.set(CAM_A, "value", value)
    cache.set(CAM_A, "nothing", None)

    assert cache.get(CAM_A, "value") is value
    assert cache.get(CAM_A, "nothing", MISSING) is None
    assert cache.get(CAM_A, "absent", MISSING) is MISSING


def test_cache_namespaces_are_independent_and_global_is_its_own():
    cache = ObjectCache()
    cache.set(CAM_A, "key", "a")
    cache.set(CAM_B, "key", "b")
    cache.set(GLOBAL_NAMESPACE, "key", "global")

    assert cache.get(CAM_A, "key") == "a"
    assert cache.get(CAM_B, "key") == "b"
    assert cache.get(GLOBAL_NAMESPACE, "key") == "global"
    # A source literally named "global" is still a source namespace.
    assert cache.get(CacheNamespace("source", "global"), "key", MISSING) is MISSING


def _filled_cache() -> ObjectCache:
    cache = ObjectCache()
    cache.set(CAM_A, "x", 1)
    cache.set(CAM_A, "y", 2)
    cache.set(CAM_B, "x", 3)
    cache.set(GLOBAL_NAMESPACE, "x", 4)

    return cache


@pytest.mark.parametrize(
    "options, removed, remaining",
    [
        ({}, 4, set()),
        ({"scope": "global"}, 1, {CAM_A, CAM_B}),
        ({"scope": "source"}, 3, {GLOBAL_NAMESPACE}),
        ({"scope": "source", "source": "cam-a"}, 2, {CAM_B, GLOBAL_NAMESPACE}),
        ({"scope": "source", "source": "unknown"}, 0, {CAM_A, CAM_B, GLOBAL_NAMESPACE}),
    ],
)
def test_clear_removes_exactly_the_named_scope(options, removed, remaining):
    cache = _filled_cache()

    assert cache.clear(**options) == removed

    kept = {
        namespace
        for namespace in (CAM_A, CAM_B, GLOBAL_NAMESPACE)
        if cache.get(namespace, "x", MISSING) is not MISSING
    }
    assert kept == remaining


@pytest.mark.parametrize(
    "options",
    [{"source": "cam-a"}, {"scope": "global", "source": "cam-a"}, {"scope": "all"}],
)
def test_clear_rejects_an_ambiguous_request_and_keeps_entries(options):
    cache = _filled_cache()

    with pytest.raises(ValueError):
        cache.clear(**options)

    assert cache.get(CAM_A, "x") == 1


def test_concurrent_writers_and_clearers_leave_a_consistent_cache():
    cache = ObjectCache()
    errors = []
    start = threading.Barrier(9)

    def write(worker: int) -> None:
        try:
            start.wait()
            for number in range(500):
                cache.set(
                    CacheNamespace("source", f"cam-{worker}"), f"k{number}", number
                )
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    def clear() -> None:
        start.wait()
        for _ in range(200):
            cache.clear(scope="source", source="cam-0")

    threads = [threading.Thread(target=write, args=(worker,)) for worker in range(8)]
    threads.append(threading.Thread(target=clear))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    for worker in range(1, 8):
        namespace = CacheNamespace("source", f"cam-{worker}")
        assert [cache.get(namespace, f"k{n}") for n in range(500)] == list(range(500))


# Blocks


def test_blocks_are_non_prunable_and_declare_their_retained_payloads():
    assert CacheSetBlock.__block_spec__.prunable is False
    assert CacheGetBlock.__block_spec__.prunable is False
    assert CacheSetBlock.retains_inputs == ("value",)
    assert CacheGetBlock.retains_outputs == ("output",)
    assert set(CacheSetBlock.outputs) == {"output"}
    assert set(CacheGetBlock.outputs) == {"output"}


def test_get_returns_false_for_a_missing_key_and_none_for_a_stored_none():
    cache = ObjectCache()
    cache.set(CacheNamespace("source", DEFAULT_SOURCE), "nothing", None)

    assert _get(cache, image=_image(), key="absent") is False
    assert _get(cache, image=_image(), key="nothing") is None


def test_set_passes_the_value_through_and_stores_it_by_reference():
    cache = ObjectCache()
    predictions = {"boxes": [[0, 0, 1, 1]]}

    output = _set(cache, image=_image(), key="p", value=predictions)

    assert output is predictions
    assert _get(cache, image=_image(), key="p") is predictions
    predictions["boxes"].append([1, 1, 2, 2])
    assert len(_get(cache, image=_image(), key="p")["boxes"]) == 2


@pytest.mark.parametrize(
    "video, source, expected",
    [
        ("video-1", "cam-x", CacheNamespace("source", "video-1")),
        (None, "cam-x", CacheNamespace("source", "cam-x")),
        ("", "cam-x", CacheNamespace("source", "cam-x")),
        (None, None, CacheNamespace("source", DEFAULT_SOURCE)),
    ],
    ids=["video-identifier-first", "then-sample-source", "empty-identifier", "default"],
)
def test_source_scope_namespace_order(video, source, expected):
    cache = ObjectCache()

    _set(cache, image=_image(video=video), key="k", value="v", source=source)

    assert cache.get(expected, "k") == "v"
    assert cache.clear() == 1


def test_global_scope_ignores_video_and_source():
    cache = ObjectCache()
    _set(cache, image=_image(video="video-1"), key="k", value="v", scope="global")

    read = _get(cache, image=_image(video="video-2"), key="k", scope="global")
    other = _get(cache, image=_image(video="video-1"), key="k")

    assert read == "v"
    assert cache.get(GLOBAL_NAMESPACE, "k") == "v"
    assert other is False


@pytest.mark.parametrize("device", DEVICES)
def test_set_stores_an_owned_copy_of_a_direct_image(device):
    cache = ObjectCache()
    image = _image(7, video="cam-a", device=device)

    output = _set(cache, image=image, key="frame", value=image)
    image.tensor_image.fill_(99)
    stored = _get(cache, image=image, key="frame")

    assert output is image
    assert stored is not image
    assert stored.image_id == image.image_id
    assert stored.video_metadata is image.video_metadata
    assert stored.device == image.device
    assert int(stored.tensor_image.max()) == 7
    assert _get(cache, image=image, key="frame") is stored


def test_images_inside_a_container_are_kept_by_reference():
    cache = ObjectCache()
    image = _image(7)

    _set(cache, image=image, key="frames", value=[image])
    image.tensor_image.fill_(99)
    stored = _get(cache, image=image, key="frames")

    assert stored[0] is image
    assert int(stored[0].tensor_image.max()) == 99


def test_a_reader_mutating_the_output_changes_the_cached_value():
    cache = ObjectCache()
    _set(cache, image=_image(), key="counts", value={"cars": 1})

    _get(cache, image=_image(), key="counts")["cars"] = 5

    assert _get(cache, image=_image(), key="counts") == {"cars": 5}
