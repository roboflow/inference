import datetime
import threading

import numpy as np

from inference.core.workflows.core_steps.cache.cache_get.v1 import CacheGetBlockV1
from inference.core.workflows.core_steps.cache.cache_set.v1 import CacheSetBlockV1
from inference.core.workflows.core_steps.cache.memory_cache import WorkflowMemoryCache
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def _image_for(video_id: str) -> WorkflowImageData:
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


def _reset_memory_cache() -> None:
    WorkflowMemoryCache.cache.clear()
    WorkflowMemoryCache._retain_counts.clear()


def test_cache_on_video() -> None:
    # given
    metadata = VideoMetadata(
        video_identifier="vid",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
    cache_get_block = CacheGetBlockV1()
    cache_set_block = CacheSetBlockV1()

    # empty result
    get_empty = cache_get_block.run(image=image, key="foo")
    assert get_empty == {
        "output": False,
    }

    # set then get
    cache_set_block.run(image=image, key="foo", value="bar")
    get_full = cache_get_block.run(image=image, key="foo")
    assert get_full == {
        "output": "bar",
    }
    cache_set_block.close()
    cache_get_block.close()


def test_cache_with_no_metadata() -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )
    cache_get_block = CacheGetBlockV1()
    cache_set_block = CacheSetBlockV1()

    # empty result
    get_empty = cache_get_block.run(image=image, key="foo")
    assert get_empty == {
        "output": False,
    }

    # set then get
    cache_set_block.run(image=image, key="foo", value="bar")
    get_full = cache_get_block.run(image=image, key="foo")
    assert get_full == {
        "output": "bar",
    }
    cache_set_block.close()
    cache_get_block.close()


def test_cache_on_multiple_videos() -> None:
    # given
    metadata_1 = VideoMetadata(
        video_identifier="vid_1",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image_1 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata_1,
    )

    metadata_2 = VideoMetadata(
        video_identifier="vid_2",
        frame_number=10,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
    )
    image_2 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata_2,
    )

    cache_get_block = CacheGetBlockV1()
    cache_set_block = CacheSetBlockV1()

    # empty result
    get_empty = cache_get_block.run(image=image_1, key="foo")
    assert get_empty == {
        "output": False,
    }

    # set then get
    cache_set_block.run(image=image_1, key="foo", value="bar")
    get_full = cache_get_block.run(image=image_1, key="foo")
    assert get_full == {
        "output": "bar",
    }

    # make sure it doesn't bleed over
    get_empty = cache_get_block.run(image=image_2, key="foo")
    assert get_empty == {
        "output": False,
    }
    cache_set_block.close()
    cache_get_block.close()


def test_shared_namespace_survives_first_instance_close() -> None:
    # given - Cache Set and Cache Get share namespace vid_1 (the product)
    _reset_memory_cache()
    cache_set = CacheSetBlockV1()
    cache_get = CacheGetBlockV1()
    image = _image_for("vid_1")  # shared video id is the product contract

    cache_set.run(image=image, key="from_a", value="a")
    assert cache_get.run(image=image, key="from_a") == {"output": "a"}

    # when - only the set instance is closed
    cache_set.close()

    # then - the get instance can still read the shared keys
    assert "vid_1" in WorkflowMemoryCache.cache
    assert cache_get.run(image=image, key="from_a") == {"output": "a"}

    # when - both owners have closed
    cache_get.close()

    # then - retain count hit 0 and the namespace is gone
    assert "vid_1" not in WorkflowMemoryCache.cache


def test_close_releases_only_namespaces_this_instance_retained() -> None:
    # given - one instance touches vid_1 then vid_2; another holds only vid_2
    _reset_memory_cache()
    owner = CacheSetBlockV1()
    other = CacheSetBlockV1()

    owner.run(image=_image_for("vid_1"), key="foo", value="bar")
    owner.run(image=_image_for("vid_2"), key="foo", value="baz")
    other.run(image=_image_for("vid_2"), key="other", value="kept")

    assert "vid_1" in WorkflowMemoryCache.cache
    assert "vid_2" in WorkflowMemoryCache.cache

    # when - owner is closed
    owner.close()

    # then - vid_1 is gone (only owner retained it); vid_2 stays for `other`
    assert "vid_1" not in WorkflowMemoryCache.cache
    assert "vid_2" in WorkflowMemoryCache.cache
    assert WorkflowMemoryCache.cache["vid_2"]["other"] == "kept"

    other.close()
    assert "vid_2" not in WorkflowMemoryCache.cache


def test_cache_block_close_is_safe_before_first_run() -> None:
    # given - blocks that never ran
    cache_set_block = CacheSetBlockV1()
    cache_get_block = CacheGetBlockV1()

    # when / then - cleanup raises nothing and touches no namespace
    cache_set_block.close()
    cache_get_block.close()


def test_release_namespace_at_zero_frees_the_underlying_dict() -> None:
    # given - retain twice (two instances), release twice
    _reset_memory_cache()
    WorkflowMemoryCache.get_dict("vid_x")
    WorkflowMemoryCache.get_dict("vid_x")
    assert "vid_x" in WorkflowMemoryCache.cache
    assert WorkflowMemoryCache._retain_counts["vid_x"] == 2

    # when - first release: count drops to 1, dict stays
    WorkflowMemoryCache.release_namespace("vid_x")
    assert "vid_x" in WorkflowMemoryCache.cache
    assert WorkflowMemoryCache._retain_counts["vid_x"] == 1

    # when - second release: count hits 0, dict is gone (memory reclaimed)
    WorkflowMemoryCache.release_namespace("vid_x")
    assert "vid_x" not in WorkflowMemoryCache.cache
    assert "vid_x" not in WorkflowMemoryCache._retain_counts


def test_concurrent_retain_release_does_not_corrupt_refcount() -> None:
    # given - many threads retaining and releasing the same namespace
    _reset_memory_cache()
    n_threads = 16
    n_ops_per_thread = 200
    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        barrier.wait()
        for _ in range(n_ops_per_thread):
            WorkflowMemoryCache.get_dict("shared_vid")
            WorkflowMemoryCache.release_namespace("shared_vid")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # then - every retain was paired with a release; count is 0 and dict is gone
    assert WorkflowMemoryCache._retain_counts.get("shared_vid") is None
    assert "shared_vid" not in WorkflowMemoryCache.cache


def test_concurrent_run_on_same_block_does_not_race_on_first_retain() -> None:
    # Regression: without a per-instance lock, Thread A could add a namespace
    # to _namespaces but pause before get_dict created the dict. Thread B
    # would then see the namespace in the set, skip get_dict, and read
    # WorkflowMemoryCache.cache[namespace] directly — KeyError.
    #
    # The lock makes check→add→get_dict atomic within a single block instance.
    _reset_memory_cache()
    n_threads = 8
    n_runs_per_thread = 100
    barrier = threading.Barrier(n_threads)
    errors: list = []

    def worker_set() -> None:
        barrier.wait()
        for i in range(n_runs_per_thread):
            try:
                block.run(image=_image_for("race_vid"), key=f"k_{i}", value=i)
            except Exception as exc:
                errors.append(exc)

    def worker_get() -> None:
        barrier.wait()
        for i in range(n_runs_per_thread):
            try:
                result = block.run(image=_image_for("race_vid"), key=f"k_{i}")
                assert result["output"] is False or result["output"] is not None
            except Exception as exc:
                errors.append(exc)

    # Test both blocks — same race exists in cache_set and cache_get
    for block_cls, worker in (
        (CacheSetBlockV1, worker_set),
        (CacheGetBlockV1, worker_get),
    ):
        _reset_memory_cache()
        errors.clear()
        block = block_cls()
        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"{block_cls.__name__} raised: {errors[:3]}"
        block.close()


def test_cache_blocks_declare_only_the_stateful_video_soft_restriction() -> None:
    # Cache blocks have no remote path: they always run in-process, so where
    # model steps execute must not gate them. Only the shared stateful-video
    # caveat applies.
    from inference.core.workflows.prototypes.block import (
        STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    )

    for block_cls in (CacheGetBlockV1, CacheSetBlockV1):
        manifest = block_cls.get_manifest()
        assert manifest.get_restrictions() == [STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION]
        assert block_cls.get_init_parameters() == []


def test_cache_blocks_run_without_step_execution_mode() -> None:
    _reset_memory_cache()
    image = _image_for("vid")
    cache_set = CacheSetBlockV1()
    cache_get = CacheGetBlockV1()
    try:
        assert cache_set.run(image=image, key="k", value=1) == {"output": 1}
        assert cache_get.run(image=image, key="k") == {"output": 1}
    finally:
        cache_set.close()
        cache_get.close()
