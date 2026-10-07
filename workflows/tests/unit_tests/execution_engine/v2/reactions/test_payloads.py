"""Event payload handoff: readiness, owned snapshots, retention and release."""

import dataclasses
import weakref
from concurrent.futures import Future
from typing import Any, List

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.data import (
    Batch,
    EntryMetadata,
    SampleContext,
)
from roboflow_workflows.execution_engine.v2.errors import (
    EventEmissionError,
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.reactions.snapshots import (
    register_snapshot,
    snapshot_fields,
)

from tests.unit_tests.execution_engine.v2.reactions.test_runtime import (
    NOTIFY,
    PASSIVE,
    PAYLOAD_HANDLER,
    WAIT,
    Probe,
    active_definition,
    handler,
    reacting,
    start,
)

BOTH = {"value": "value", "payload": "payload"}


def snapshot(value: Any) -> Any:
    return snapshot_fields({"field": value}, where="test").fields["field"]


# Snapshot rules -----------------------------------------------------------------


def test_arrays_and_tensors_are_copied_on_their_own_device_and_dtype():
    array = np.arange(6, dtype=np.uint16).reshape(2, 3)
    tensor = torch.arange(4, dtype=torch.float16)

    copied_array, copied_tensor = snapshot(array), snapshot(tensor)
    array[0, 0], tensor[0] = 99, 99

    assert copied_array.dtype == np.uint16 and copied_array[0, 0] == 0
    assert (
        copied_tensor.dtype == torch.float16 and copied_tensor.device == tensor.device
    )
    assert copied_tensor[0] == 0


def test_image_data_keeps_identity_provenance_and_metadata():
    image = ImageData.from_tensor(
        torch.zeros((3, 4, 5), dtype=torch.uint8), image_id="frame-7"
    )

    copied = snapshot(image)
    image.tensor_image[0, 0, 0] = 255

    assert copied is not image
    assert copied.image_id == "frame-7"
    assert copied.parent == image.parent and copied.root == image.root
    assert copied.video_metadata == image.video_metadata
    assert copied.tensor_image[0, 0, 0] == 0
    assert copied.tensor_image.data_ptr() != image.tensor_image.data_ptr()


def test_containers_are_rebuilt_and_aliases_inside_one_event_stay_shared():
    shared = np.zeros(3)
    value = {"a": [shared, shared], "b": (1, "x"), "c": {1, 2}}

    copied = snapshot_fields({"one": value, "two": shared}, where="test").fields
    shared[0] = 5

    assert copied["one"]["a"][0] is copied["one"]["a"][1] is copied["two"]
    assert copied["two"][0] == 0
    assert copied["one"]["b"] is value["b"]
    assert copied["one"]["c"] == {1, 2} and copied["one"]["c"] is not value["c"]


def test_a_batch_keeps_indices_layout_and_metadata():
    metadata = EntryMetadata(sample={(0,): SampleContext(source_id="a")})
    batch = Batch([np.zeros(2), np.ones(2)], indices=[(0,), (3,)], metadata=metadata)

    copied = snapshot(batch)
    batch[0][0] = 7

    assert copied.indices == ((0,), (3,)) and copied.metadata is metadata
    assert copied[0][0] == 0


def test_mutable_dataclasses_are_rebuilt_frozen_unchanged_ones_kept():
    @dataclasses.dataclass
    class Box:
        values: list

    @dataclasses.dataclass(frozen=True)
    class Label:
        name: str

    box, label = Box([1]), Label("x")
    copied = snapshot_fields({"box": box, "label": label}, where="test").fields
    box.values.append(2)

    assert copied["box"] == Box([1]) and copied["label"] is label


def test_unknown_mutable_objects_are_rejected_not_borrowed():
    class Handle:
        pass

    with pytest.raises(EventEmissionError, match="has no supported ownership snapshot"):
        snapshot_fields({"handle": Handle()}, where="$steps.s event 'e'")


def test_hooks_and_registrations_provide_owned_copies():
    class Hooked:
        def __workflows_snapshot__(self):
            return "hooked copy"

    class Registered:
        pass

    class Derived(Registered):
        pass

    register_snapshot(Registered, lambda value: "registered copy")

    assert snapshot(Hooked()) == "hooked copy"
    assert snapshot(Derived()) == "registered copy"


# In engine runs -------------------------------------------------------------------


def test_async_handler_reads_the_value_at_emit_despite_later_mutation():
    probe = Probe()
    gate = probe.hold(0.0)
    frame = np.zeros(4, dtype=np.uint8)
    probe.payloads[0.0] = frame
    probe.after_emit = lambda value, payload: (
        payload.fill(9) if payload is not None else None
    )
    plan = reacting(
        active_definition(), handler(depth=2, bindings=BOTH, workflow=PAYLOAD_HANDLER)
    )
    run = start(plan, probe, {"cam_a": [0]})

    probe.wait_for(lambda: probe.values("after") == [0.0])
    assert frame.tolist() == [9, 9, 9, 9]
    gate.set()
    assert run.wait(WAIT)

    ((_, _, _, received),) = probe.seen
    assert received.tolist() == [0, 0, 0, 0] and received.dtype == np.uint8
    counters = run.reaction_counters[NOTIFY]
    assert (counters.snapshots, counters.snapshot_bytes) == (1, 4)
    assert counters.snapshot_unknown_sizes == 0


def test_sync_handler_receives_the_emitted_object_itself():
    probe = Probe()
    frame = np.zeros(2)
    plan = reacting(
        PASSIVE, handler(mode="sync", bindings=BOTH, workflow=PAYLOAD_HANDLER)
    )
    session = plan.create_session(resources={"probe": probe})
    probe.payloads[1.0] = frame

    session.run({"value": 1.0})

    assert probe.seen[0][3] is frame


class Tracked:
    """A payload whose snapshots and originals can be watched with weakrefs."""

    copies: List[weakref.ref] = []

    def __workflows_snapshot__(self):
        copy = Tracked()
        Tracked.copies.append(weakref.ref(copy))
        return copy


def test_unbound_fields_are_not_retained_and_dropped_events_are_released_at_once():
    Tracked.copies = []
    probe = Probe()
    gate = probe.hold(0.0)
    unbound = Tracked()
    unbound_ref = weakref.ref(unbound)
    probe.payloads.update({0.0: unbound, 1.0: Tracked(), 2.0: Tracked()})
    leaky = handler(depth=1, overflow="leaky", bindings=BOTH, workflow=PAYLOAD_HANDLER)
    value_only = handler("tally", depth=4)
    plan = reacting(active_definition(), leaky, value_only)
    feeds = {"cam_a": [0, probe.started(0.0), 1, 2]}
    run = start(plan, probe, feeds)

    probe.wait_for(lambda: probe.values("after") == [0.0, 1.0, 2.0])
    del probe.payloads[0.0], unbound
    # "tally" binds only the value: the original payload is gone, no copy kept.
    assert unbound_ref() is None
    # E1 was dropped: its copy is gone without a collector run.
    assert Tracked.copies[1]() is None
    assert run.reaction_counters[NOTIFY].pending == 1
    gate.set()
    assert run.wait(WAIT)

    probe.seen.clear()
    assert all(ref() is None for ref in Tracked.copies)


def test_futures_in_bound_fields_resolve_before_the_handler_and_failures_fail_the_step():  # noqa: E501
    probe = Probe()
    ready, failed = Future(), Future()
    ready.set_result(np.ones(2))
    failed.set_exception(RuntimeError("decode failed"))
    plan = reacting(
        PASSIVE, handler(mode="sync", bindings=BOTH, workflow=PAYLOAD_HANDLER)
    )
    session = plan.create_session(resources={"probe": probe})
    probe.payloads.update({1.0: ready, 2.0: failed})

    session.run({"value": 1.0})
    with pytest.raises(StepExecutionError, match="a future in a bound field failed"):
        session.run({"value": 2.0})

    assert probe.seen[0][3].tolist() == [1.0, 1.0]
    assert probe.values("react") == [1.0]


class Tagged:
    """A hashable mutable set member with an ownership hook."""

    def __init__(self, tags: List[str]) -> None:
        self.tags = tags

    def __workflows_snapshot__(self) -> "Tagged":
        return Tagged(list(self.tags))


def test_object_arrays_and_set_members_are_snapshotted_member_by_member():
    inner = []
    objects = np.empty((1, 2), dtype=object)
    objects[0, 0], objects[0, 1] = inner, "text"
    numbers = np.arange(3.0)
    member = Tagged(["a"])
    snapshot = snapshot_fields(
        {"objects": objects, "numbers": numbers, "members": {member}}, where="test"
    )
    inner.append("late")
    numbers[0] = -1.0
    member.tags.append("late")

    copied = snapshot.fields
    assert copied["objects"].dtype == object and copied["objects"].shape == (1, 2)
    assert copied["objects"][0, 0] == [] and copied["objects"][0, 1] == "text"
    np.testing.assert_array_equal(copied["numbers"], np.arange(3.0))
    (copied_member,) = copied["members"]
    assert copied_member is not member and copied_member.tags == ["a"]


def test_a_structured_array_with_object_fields_is_rejected():
    records = np.zeros(2, dtype=[("label", object), ("score", float)])
    with pytest.raises(EventEmissionError, match="structured array"):
        snapshot_fields({"records": records}, where="test")
