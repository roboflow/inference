"""Cache blocks in compiled workflows: real compiler, session, pipeline and updates.

Who holds the cache::

    Catalogue(providers={"object_cache": Factory(ObjectCache)})
      create_session()                          own cache per session
      create_session({"object_cache": cache})   the caller's cache, shared

Most workflows read before they write, so each run sees the previous run::

    get(k) ──after──▶ set(k, value)             run n reads run n-1
    set(k, value) ──after──▶ get(k)             run n reads run n
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.state.object_cache import (
    DEFAULT_SOURCE,
    GLOBAL_NAMESPACE,
    CacheGetBlock,
    CacheNamespace,
    CacheSetBlock,
    ObjectCache,
)
from roboflow_workflows.v2.data.batch import EntryMetadata, InputValue, SampleContext
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import MutationConflictError, ResourceError
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions

DEFAULT = CacheNamespace(scope="source", source=DEFAULT_SOURCE)
MISSING = object()


class _Nothing(Block):
    """Test block: a present ``None`` output."""

    type = "test/nothing@v1"
    outputs = {"value": Output(source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND)

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        return {"value": None}


class _Echo(Block):
    """Test block: returns its input object; a ``None`` input skips it.

    ``source="value"`` declares the pass-through, so the mutation checks follow
    the object from ``value`` to the output.
    """

    type = "test/echo@v1"
    outputs = {"value": Output(source="value")}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value: Any) -> Dict[str, Any]:
        return {"value": value}


class _First(Block):
    """Test block: returns the first element of its input, a part of it."""

    type = "test/first@v1"
    outputs = {"value": Output(source="value")}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value: Any) -> Dict[str, Any]:
        return {"value": value[0]}


class _Fresh(Block):
    """Test block: a new ``{"count": 0}`` per call."""

    type = "test/fresh@v1"
    outputs = {"value": Output(source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND)

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        return {"value": {"count": 0}}


class _Wrap(Block):
    """Test block: the image inside a one-element list."""

    type = "test/wrap@v1"
    outputs = {"value": Output(source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND)

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        return {"value": [image]}


class _Bump(Block):
    """Test block: increments ``value["count"]`` in place."""

    type = "test/bump@v1"
    mutates = ("value",)
    outputs = {"count": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value: Any) -> Dict[str, Any]:
        value["count"] += 1

        return {"count": value["count"]}


class _IsNone(Block):
    """Test block: whether its input is ``None``; accepts a present ``None``."""

    type = "test/is_none@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref(allow_none=True)

    def run(self, *, value: Any) -> Dict[str, Any]:
        return {"value": value is None}


CACHE_BLOCKS = [CacheGetBlock, CacheSetBlock]
TEST_BLOCKS = [_Nothing, _Echo, _First, _Fresh, _Wrap, _Bump, _IsNone]
CATALOGUE = create_catalogue().with_blocks(TEST_BLOCKS)
IMAGE_INPUT = {"type": "WorkflowImage", "name": "image"}
VALUE_INPUT = {"type": "WorkflowParameter", "name": "value"}
AFTER_INPUT = {"type": "WorkflowParameter", "name": "after"}


def _image(value: int = 0, *, video: Optional[str] = None) -> ImageData:
    metadata = None
    if video is not None:
        metadata = VideoMetadata(
            video_identifier=video, frame_number=1, frame_timestamp=datetime(2026, 1, 1)
        )
    pixels = torch.full((3, 4, 4), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, video_metadata=metadata)

    return image


def _definition(
    steps: List[dict], outputs: Dict[str, str], *, inputs: Optional[list] = None
) -> dict:
    definition = {
        "version": "2.0",
        "inputs": inputs or [IMAGE_INPUT],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }

    return definition


def _get(name: str, key: str = "k", **options: Any) -> dict:
    step = {"type": CacheGetBlock.type, "name": name, "image": "$inputs.image"}

    return {**step, "key": key, **options}


def _set(name: str, value: str, key: str = "k", **options: Any) -> dict:
    step = {"type": CacheSetBlock.type, "name": name, "image": "$inputs.image"}

    return {**step, "key": key, "value": value, **options}


def _step(block: type, name: str, **bindings: str) -> dict:
    return {"type": block.type, "name": name, **bindings}


# get(k) -> set(k, $inputs.value): every run reads the previous run's value.
READ_THEN_WRITE = _definition(
    [_get("get"), _set("set", "$inputs.value", after="$steps.get.output")],
    {"previous": "$steps.get.output"},
    inputs=[IMAGE_INPUT, VALUE_INPUT],
)
# READ_THEN_WRITE plus a read of this run's write.
READ_WRITE_READ = _definition(
    [
        _get("get"),
        _set("set", "$inputs.value", after="$steps.get.output"),
        _get("peek", after="$steps.set.output"),
    ],
    {"previous": "$steps.get.output", "now": "$steps.peek.output"},
    inputs=[IMAGE_INPUT, VALUE_INPUT],
)


def _previous(session: Any, value: Any) -> Any:
    [row] = session.run({"image": _image(), "value": value}).rows()

    return row["previous"]


# Runs, ordering and None


def test_a_value_set_in_one_run_is_read_in_the_next():
    plan = compile_workflow(READ_THEN_WRITE, catalogue=CATALOGUE)
    session = plan.create_session()

    reads = [_previous(session, value) for value in (1, 2, 3)]

    assert reads == [False, 1, 2]
    assert plan.warnings == ()
    session.close()


def test_none_is_stored_through_the_setter_and_differs_from_a_miss():
    # Run 2: "before" reads the stored None, and "set" (after=None) still runs.
    definition = _definition(
        [
            _get("before"),
            _step(_Nothing, "nothing", image="$inputs.image"),
            _set("set", "$steps.nothing.value", after="$steps.before.output"),
            _get("get", after="$steps.set.output"),
        ],
        {"before": "$steps.before.output", "get": "$steps.get.output"},
    )
    cache = ObjectCache()
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session(
        {"object_cache": cache}
    )

    first = session.run({"image": _image()}).rows()
    cache.set(GLOBAL_NAMESPACE, "marker", "untouched")
    second = session.run({"image": _image()}).rows()

    assert first == [{"before": False, "get": None}]
    assert second == [{"before": None, "get": None}]
    assert cache.get(DEFAULT, "k", MISSING) is None
    assert cache.get(DEFAULT, "absent", MISSING) is MISSING
    session.close()


def _ordered_read(after: str, *, nested: bool) -> dict:
    # Stores None, then reads "k" after `after`; "is_none" consumes the read.
    reader = _nested(CHILD_READER, after) if nested else _get("get", after=after)
    read = "$steps.child.value" if nested else "$steps.get.output"
    definition = _definition(
        [
            _step(_Nothing, "nothing", image="$inputs.image"),
            _set("set", "$steps.nothing.value"),
            _step(_Echo, "echo", value="$steps.nothing.value"),
            reader,
            _step(_IsNone, "is_none", value=read),
        ],
        {"read": read, "is_none": "$steps.is_none.value"},
    )

    return definition


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
@pytest.mark.parametrize(
    "after, status, produced",
    [
        # The getter waits for the stored None: it reads it, not "stale".
        ("$steps.set.output", "complete", {"read": (None,), "is_none": (True,)}),
        # "echo" skips on None, so its output is missing: the getter skips.
        ("$steps.echo.value", "filtered", {}),
    ],
    ids=["present-none", "missing"],
)
def test_the_getter_runs_after_a_present_none_and_skips_after_a_missing_value(
    mode, nested, after, status, produced
):
    cache = ObjectCache()
    cache.set(DEFAULT, "k", "stale")
    plan = compile_workflow(
        _ordered_read(after, nested=nested),
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode),
    )
    session = plan.create_session({"object_cache": cache})

    result = session.run({"image": _image()})
    session.close()

    contents = {
        name: tuple(batch.content) for name, batch in result.outputs.data.items()
    }
    assert result.statuses == {"read": status, "is_none": status}
    assert contents == produced
    assert cache.get(DEFAULT, "k") is None


def test_a_filtered_upstream_value_still_skips_the_setter():
    # "echo" skips on a None input, so its output is missing, not None.
    definition = _definition(
        [
            _step(_Nothing, "nothing", image="$inputs.image"),
            _step(_Echo, "echo", value="$steps.nothing.value"),
            _set("set", "$steps.echo.value"),
            _get("after_missing", after="$steps.echo.value"),
        ],
        {"set": "$steps.set.output", "read": "$steps.after_missing.output"},
    )
    cache = ObjectCache()
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session(
        {"object_cache": cache}
    )

    rows = session.run({"image": _image()}).rows()

    assert rows == [{"set": None, "read": None}]
    assert cache.get(DEFAULT, "k", MISSING) is MISSING
    session.close()


def test_pipelined_pulses_one_in_flight_read_in_submission_order():
    # More pulses in flight may interleave pulse 1's set with pulse 2's get:
    # the engine orders each step across pulses, not a get/set pair.
    session = compile_workflow(READ_WRITE_READ, catalogue=CATALOGUE).create_session()
    options = PipelineOptions(max_in_flight=1)

    with session.pipeline(options=options) as pipeline:
        futures = [
            pipeline.submit({"image": _image(), "value": value}) for value in (1, 2, 3)
        ]
        rows = [future.result(timeout=5).rows() for future in futures]

    assert rows == [
        [{"previous": False, "now": 1}],
        [{"previous": 1, "now": 2}],
        [{"previous": 2, "now": 3}],
    ]


# Who shares a cache


def test_each_session_gets_its_own_cache_from_the_catalogue_factory():
    plan = compile_workflow(READ_THEN_WRITE, catalogue=CATALOGUE)
    first, second = plan.create_session(), plan.create_session()

    _previous(first, "first")

    assert _previous(second, "second") is False
    assert _previous(first, "again") == "first"
    first.close()
    second.close()


def test_sessions_given_the_same_cache_share_it_and_see_host_writes():
    plan = compile_workflow(READ_THEN_WRITE, catalogue=CATALOGUE)
    cache = ObjectCache()
    cache.set(DEFAULT, "k", None)
    first = plan.create_session({"object_cache": cache})
    second = plan.create_session({"object_cache": cache})

    host_none = _previous(first, "first")
    shared = _previous(second, "second")

    assert host_none is None
    assert shared == "first"
    assert cache.get(DEFAULT, "k") == "second"
    first.close()
    second.close()


def test_a_session_without_a_cache_provider_names_the_missing_resource():
    catalogue = Catalogue(CACHE_BLOCKS, kinds=[IMAGE_KIND])
    plan = compile_workflow(READ_THEN_WRITE, catalogue=catalogue)

    with pytest.raises(ResourceError, match="object_cache"):
        plan.create_session()


# Graph updates and resets


def test_graph_updates_and_resets_keep_a_caller_cache_and_never_clear_it():
    cache = ObjectCache()
    plan = compile_workflow(READ_THEN_WRITE, catalogue=CATALOGUE)
    session = plan.create_session({"object_cache": cache})

    first = _previous(session, 1)
    session.update(compile_workflow(READ_WRITE_READ, catalogue=CATALOGUE))
    updated = _previous(session, 2)
    session.update(plan, reset=True)
    after_reset = _previous(session, 3)
    session.close()

    assert (first, updated, after_reset) == (False, 1, 2)
    assert cache.get(DEFAULT, "k") == 3


def test_a_graph_update_keeps_the_session_cache_and_a_reset_creates_a_new_one():
    # Core behaviour: a reset resolves resources again, so Factory values are
    # created again (engine/updates/reset.py). Pass a cache to keep it.
    plan = compile_workflow(READ_THEN_WRITE, catalogue=CATALOGUE)
    session = plan.create_session()

    first = _previous(session, 1)
    session.update(compile_workflow(READ_WRITE_READ, catalogue=CATALOGUE))
    updated = _previous(session, 2)
    session.update(plan, reset=True)
    after_reset = _previous(session, 3)
    session.close()

    assert (first, updated, after_reset) == (False, 1, False)


# Namespaces


def test_source_scope_uses_the_video_identifier_then_the_sample_source():
    definition = _definition(
        [_set("set", "$inputs.image"), _set("global", "$inputs.image", scope="global")],
        {"stored": "$steps.set.output"},
    )
    cache = ObjectCache()
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session(
        {"object_cache": cache}
    )
    images = [_image(1), _image(2, video="video-x")]
    sample = {(n,): SampleContext(source_id=f"cam{n}") for n in range(2)}
    batch = InputValue(data=images, metadata=EntryMetadata(sample=sample))

    session.run({"image": batch})

    stored = {
        source: cache.get(CacheNamespace("source", source), "k", MISSING)
        for source in ("cam0", "cam1", "video-x")
    }
    assert stored["cam0"].image_id == images[0].image_id
    assert stored["cam1"] is MISSING
    assert stored["video-x"].image_id == images[1].image_id
    # Both calls write the one global key; one of the two images wins.
    winner = cache.get(GLOBAL_NAMESPACE, "k").image_id
    assert winner in {images[0].image_id, images[1].image_id}
    session.close()


KEY_INPUT = {"type": "WorkflowParameter", "name": "key"}
SOURCE_A = CacheNamespace(scope="source", source="A")
VIDEO_X = CacheNamespace(scope="source", source="video-x")
CHILD_KEYED_READER = _definition(
    [_get("get", key="$inputs.key", after="$inputs.after")],
    {"value": "$steps.get.output"},
    inputs=[IMAGE_INPUT, KEY_INPUT, AFTER_INPUT],
)


def _sampled(image: ImageData, source: str) -> InputValue:
    sample = {(): SampleContext(source_id=source)}
    value = InputValue(data=image, metadata=EntryMetadata(sample=sample))

    return value


def _write_then_read(*, nested: bool) -> dict:
    # set(key, $inputs.value) ──after──▶ get(key). The setter's output is a
    # workflow parameter without a context, so the getter's call has no common
    # context; only its image's own SampleContext names the namespace.
    after = "$steps.set.output"
    if nested:
        reader = _nested(CHILD_KEYED_READER, after, key="$inputs.key")
    else:
        reader = _get("get", key="$inputs.key", after=after)
    read = "$steps.child.value" if nested else "$steps.get.output"
    definition = _definition(
        [_set("set", "$inputs.value", key="$inputs.key"), reader],
        {"read": read},
        inputs=[IMAGE_INPUT, VALUE_INPUT, KEY_INPUT],
    )

    return definition


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
@pytest.mark.parametrize("value", [None, 0], ids=["none", "zero"])
def test_a_read_after_a_scalar_write_uses_the_image_sample_source(mode, nested, value):
    cache = ObjectCache()
    plan = compile_workflow(
        _write_then_read(nested=nested),
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode),
    )
    session = plan.create_session({"object_cache": cache})
    later = {"n": 2}

    first = session.run(
        {"image": _sampled(_image(video=""), "A"), "value": value, "key": "sel"}
    )
    second = session.run(
        {"image": _sampled(_image(video=""), "B"), "value": later, "key": "sel"}
    )
    session.close()

    # A miss reads False, and False == 0: compare types as well.
    [read] = first.outputs.data["read"].content
    assert first.statuses == {"read": "complete"}
    assert (read, type(read)) == (value, type(value))
    assert tuple(second.outputs.data["read"].content) == (later,)
    stored = cache.get(SOURCE_A, "sel", MISSING)
    assert (stored, type(stored)) == (value, type(value))
    assert cache.get(CacheNamespace("source", "B"), "sel") is later
    assert cache.get(DEFAULT, "sel", MISSING) is MISSING


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize(
    "video, source, scope, waits_for, namespace, status",
    [
        # The video identifier wins over the image's sample source.
        ("video-x", "A", "source", "set", VIDEO_X, "complete"),
        ("", "A", "global", "set", GLOBAL_NAMESPACE, "complete"),
        # Neither a video identifier nor a source: the default namespace.
        (None, None, "source", "set", DEFAULT, "complete"),
        # "echo" skips on None, so the getter waits for a missing value.
        ("", "A", "source", "echo", SOURCE_A, "filtered"),
    ],
    ids=["video-identifier", "global", "no-source", "missing-after"],
)
def test_a_read_after_a_scalar_write_keeps_the_namespace_rules(
    mode, video, source, scope, waits_for, namespace, status
):
    after = {"set": "$steps.set.output", "echo": "$steps.echo.value"}[waits_for]
    definition = _definition(
        [
            _step(_Nothing, "nothing", image="$inputs.image"),
            _step(_Echo, "echo", value="$steps.nothing.value"),
            _set("set", "$inputs.value", scope=scope),
            _get("get", scope=scope, after=after),
        ],
        {"read": "$steps.get.output"},
        inputs=[IMAGE_INPUT, VALUE_INPUT],
    )
    cache = ObjectCache()
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session({"object_cache": cache})
    image = _image(video=video)
    if source is not None:
        image = _sampled(image, source)

    result = session.run({"image": image, "value": 0})
    session.close()

    contents = {
        name: tuple(batch.content) for name, batch in result.outputs.data.items()
    }
    produced = {"read": (0,)} if status == "complete" else {}
    assert result.statuses == {"read": status}
    assert contents == produced
    assert all(type(read) is int for read in contents.get("read", ()))
    # The setter wrote one entry, into this namespace only.
    assert cache.get(namespace, "k") == 0
    assert cache.clear(scope=namespace.scope, source=namespace.source) == 1
    assert cache.clear() == 0


# Ownership of stored images


def test_a_direct_image_is_stored_as_a_copy_and_a_listed_image_by_reference():
    definition = _definition(
        [
            _set("direct", "$inputs.image", key="direct"),
            _step(_Wrap, "wrap", image="$inputs.image"),
            _set("listed", "$steps.wrap.value", key="listed"),
        ],
        {"direct": "$steps.direct.output"},
    )
    cache = ObjectCache()
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session(
        {"object_cache": cache}
    )
    image = _image(7)

    [row] = session.run({"image": image}).rows()
    image.tensor_image.fill_(99)
    direct = cache.get(DEFAULT, "direct")
    [listed] = cache.get(DEFAULT, "listed")

    assert row["direct"] is image
    assert direct is not image and int(direct.tensor_image.max()) == 7
    assert listed is image and int(listed.tensor_image.max()) == 99
    session.close()


# Nested workflows

CHILD_READER = _definition(
    [_get("get", after="$inputs.after")],
    {"value": "$steps.get.output"},
    inputs=[IMAGE_INPUT, AFTER_INPUT],
)


def _nested(child: dict, after: str, **bindings: str) -> dict:
    step = {
        "type": "inner_workflow",
        "name": "child",
        "workflow_definition": child,
        "parameter_bindings": {"image": "$inputs.image", "after": after, **bindings},
    }

    return step


def test_a_nested_workflow_uses_the_outer_session_cache():
    definition = _definition(
        [_set("set", "$inputs.value"), _nested(CHILD_READER, "$steps.set.output")],
        {"value": "$steps.child.value"},
        inputs=[IMAGE_INPUT, VALUE_INPUT],
    )
    cache = ObjectCache()
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session(
        {"object_cache": cache}
    )

    rows = session.run({"image": _image(), "value": "outer"}).rows()

    assert rows == [{"value": "outer"}]
    assert cache.get(DEFAULT, "k") == "outer"
    session.close()


# Mutation of retained payloads

GET_THEN_BUMP = _definition(
    [_get("get"), _step(_Bump, "bump", value="$steps.get.output")],
    {"count": "$steps.bump.count"},
)
SET_THEN_BUMP = _definition(
    [
        _step(_Fresh, "fresh", image="$inputs.image"),
        _set("set", "$steps.fresh.value"),
        _step(_Bump, "bump", value="$steps.set.output"),
    ],
    {"count": "$steps.bump.count"},
)
CHILD_BUMPS = _definition(
    [_step(_Bump, "bump", value="$inputs.after")],
    {"count": "$steps.bump.count"},
    inputs=[IMAGE_INPUT, AFTER_INPUT],
)
PARENT_GET_CHILD_BUMPS = _definition(
    [_get("get"), _nested(CHILD_BUMPS, "$steps.get.output")],
    {"count": "$steps.child.count"},
)
CHILD_GET_PARENT_BUMPS = _definition(
    [
        _nested(CHILD_READER, "$inputs.image"),
        _step(_Bump, "bump", value="$steps.child.value"),
    ],
    {"count": "$steps.bump.count"},
)


@pytest.mark.parametrize(
    "definition, keeper",
    [
        (GET_THEN_BUMP, "$steps.get keeps ['output output']"),
        (SET_THEN_BUMP, "$steps.set keeps ['input value ($steps.fresh.value)']"),
        (PARENT_GET_CHILD_BUMPS, "$steps.get keeps ['output output']"),
        (CHILD_GET_PARENT_BUMPS, "$steps.child/get keeps ['output output']"),
    ],
    ids=["get-output", "set-input", "nested-child-mutates", "nested-parent-mutates"],
)
def test_mutating_a_cached_payload_warns_and_strict_compilation_rejects_it(
    definition, keeper
):
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    with pytest.raises(MutationConflictError, match=re.escape(keeper)):
        compile_workflow(
            definition,
            catalogue=CATALOGUE,
            options=CompileOptions(mutation_conflicts="error"),
        )

    assert any(keeper in warning for warning in plan.warnings)


def test_a_warned_mutation_still_changes_the_cached_value():
    cache = ObjectCache()
    kept = {"count": 0}
    cache.set(DEFAULT, "k", kept)
    session = compile_workflow(GET_THEN_BUMP, catalogue=CATALOGUE).create_session(
        {"object_cache": cache}
    )

    counts = [session.run({"image": _image()}).rows() for _ in range(2)]

    assert counts == [[{"count": 1}], [{"count": 2}]]
    assert cache.get(DEFAULT, "k") is kept and kept == {"count": 2}
    session.close()


def _passed_on(passer: type) -> dict:
    definition = _definition(
        [
            _get("get"),
            _step(passer, "pass", value="$steps.get.output"),
            _step(_Bump, "bump", value="$steps.pass.value"),
        ],
        {"count": "$steps.bump.count"},
    )

    return definition


@pytest.mark.parametrize(
    "passer, stored",
    [(_Echo, lambda kept: kept), (_First, lambda kept: [kept])],
    ids=["whole", "element"],
)
def test_a_cached_payload_passed_on_by_a_declared_source_is_still_followed(
    passer, stored
):
    # The mutation is reported through the pass-through, and still happens.
    definition = _passed_on(passer)
    keeper = (
        "$steps.bump value ($steps.pass.value) is mutated in place after "
        "$steps.get keeps ['output output']"
    )
    cache = ObjectCache()
    kept = {"count": 0}
    value = stored(kept)
    cache.set(DEFAULT, "k", value)

    plan = compile_workflow(definition, catalogue=CATALOGUE)
    session = plan.create_session({"object_cache": cache})
    counts = [session.run({"image": _image()}).rows() for _ in range(2)]

    with pytest.raises(MutationConflictError, match=re.escape(keeper)):
        compile_workflow(
            definition,
            catalogue=CATALOGUE,
            options=CompileOptions(mutation_conflicts="error"),
        )
    assert len(plan.warnings) == 1 and keeper in plan.warnings[0]
    assert counts == [[{"count": 1}], [{"count": 2}]]
    assert cache.get(DEFAULT, "k") is value and kept == {"count": 2}
    session.close()
