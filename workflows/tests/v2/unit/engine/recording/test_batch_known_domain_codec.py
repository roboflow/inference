"""Recording codec: a batch keeps its known child domain (decision 016).

A ``Group`` with filtered children has ``known_indices`` wider than
``indices``. The batch node then adds ``"known"``; every other batch is
written exactly as before, and older recordings decode with the default
domain (``known_indices == indices``).
"""

import json
from pathlib import Path
from typing import Any

import pytest
from roboflow_workflows.v2.data.batch import (
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    WorkflowsBuffer,
)
from roboflow_workflows.v2.engine.execution.outputs import GroupResult
from roboflow_workflows.v2.engine.plan import PulseKey
from roboflow_workflows.v2.engine.recording import (
    CodecRegistry,
    RecordedEntrySchema,
    RecordedGroupSchema,
    RecordingCorruptError,
    RecordingSchema,
    RecordingWriter,
    open_recording,
)

from tests.v2.support.recording.codecs import MemoryBlobs

REGISTRY = CodecRegistry()


def encode(value: Any) -> Any:
    node = REGISTRY.encoder(MemoryBlobs(), location="g.f").encode(value)

    return node


def decode(node: Any) -> Any:
    text = json.dumps(node, allow_nan=False)
    value = REGISTRY.decoder(MemoryBlobs(), location="g.f").decode(json.loads(text))

    return value


def test_a_partial_group_keeps_its_survivors_and_known_domain() -> None:
    group = Batch(
        [2.0, 3.0],
        indices=[(1, 2), (1, 3)],
        known_indices=[(1, 0), (1, 1), (1, 2), (1, 3)],
        parent_index=(1,),
    )

    node = encode({"group": group})
    decoded = decode(node)["group"]

    assert node["dict"]["group"]["batch"]["known"] == [[1, 0], [1, 1], [1, 2], [1, 3]]
    assert decoded == group
    assert decoded.known_indices == group.known_indices
    assert decoded.parent_index == (1,)


def test_an_all_filtered_group_stays_distinct_from_a_true_empty_group() -> None:
    all_filtered = Batch([], indices=[], known_indices=[(0,), (1,)])
    true_empty = Batch.empty()

    decoded_filtered, decoded_empty = decode(encode([all_filtered, true_empty]))

    assert len(decoded_filtered) == len(decoded_empty) == 0
    assert decoded_filtered.known_indices == ((0,), (1,))
    assert decoded_empty.known_indices == ()


def test_nested_batches_each_keep_their_own_domain() -> None:
    inner = Batch(
        ["b"], indices=[(0, 1)], known_indices=[(0, 0), (0, 1)], parent_index=(0,)
    )
    outer = Batch([inner, Batch.empty(parent_index=(2,))], indices=[(0,), (2,)])

    decoded = decode(encode(outer))

    assert decoded.known_indices == ((0,), (2,))
    assert decoded.content[0].known_indices == ((0, 0), (0, 1))
    assert decoded.content[1].known_indices == ()


def test_batches_with_the_default_domain_keep_the_historical_node() -> None:
    gapped = Batch(["a", "c"], indices=[(0, 0), (0, 2)], parent_index=(0,))
    outer = Batch([gapped, Batch.empty(parent_index=(2,))], indices=[(0,), (2,)])
    historical = {
        "batch": {
            "parent": [],
            "indices": [[0], [2]],
            "items": [
                {
                    "batch": {
                        "parent": [0],
                        "indices": [[0, 0], [0, 2]],
                        "items": ["a", "c"],
                    }
                },
                {"batch": {"parent": [2], "indices": [], "items": []}},
            ],
        }
    }

    node = encode(outer)
    decoded = decode(historical)

    assert json.dumps(node) == json.dumps(historical)
    assert decoded == outer
    assert decoded.known_indices == decoded.indices
    assert decoded.content[0].known_indices == ((0, 0), (0, 2))


@pytest.mark.parametrize(
    ("known", "message"),
    [
        ({"0": [0]}, r"^g\.f\.known: expected a JSON array"),
        ([[0], "1"], r"^g\.f\.known: an index must be a list of non-negative"),
        ([[0], [-1]], r"^g\.f\.known: an index must be a list of non-negative"),
        ([[0]], r"^g\.f: invalid batch: .*\(1,\) is not among known_indices"),
        ([[1], [0]], r"^g\.f: invalid batch: .*\(1,\) .* in the same order"),
        ([[0], [0], [1]], r"^g\.f: invalid batch: .*\(0,\) is duplicated"),
        ([[0], [1], [2, 0]], r"^g\.f: invalid batch: .*share one depth"),
    ],
)
def test_malformed_known_domains_are_corruption(known: Any, message: str) -> None:
    node = {
        "batch": {"parent": [], "indices": [[0], [1]], "items": [1, 2], "known": known}
    }

    with pytest.raises(RecordingCorruptError, match=message):
        decode(node)


def test_a_known_domain_outside_the_parent_is_corruption() -> None:
    node = {
        "batch": {
            "parent": [1],
            "indices": [[1, 0]],
            "items": [1],
            "known": [[1, 0], [2, 1]],
        }
    }

    with pytest.raises(RecordingCorruptError, match=r"does not extend parent index"):
        decode(node)


def test_unknown_batch_keys_are_still_corruption() -> None:
    node = {"batch": {"parent": [], "indices": [], "items": [], "extra": []}}

    with pytest.raises(RecordingCorruptError, match=r"optionally 'known'"):
        decode(node)


def test_the_store_keeps_a_grouped_entry_domain_when_attaching_views(
    tmp_path: Path,
) -> None:
    layout = EntryLayout((Axis(id="steps.split:parts", kind="dynamic_nesting"),))
    group = RecordedGroupSchema(
        name="parts",
        anchor_domain="video",
        entries=(RecordedEntrySchema("parts", "parts", None, ("*",), layout),),
    )
    value = Batch([2.0], indices=[(1,)], known_indices=[(0,), (1,), (2,)])
    outputs = WorkflowsBuffer(
        lineage_id="video",
        pulse_id=0,
        data={"parts": value},
        layout={"parts": layout},
        metadata={"parts": EntryMetadata()},
    )
    writer = RecordingWriter.create(
        tmp_path / "rec",
        schema=RecordingSchema([group]),
        codecs=REGISTRY,
        definition_digest="sha256:abc",
    )
    writer.write(
        GroupResult(
            group="parts",
            source="video",
            pulse=PulseKey(active_run_id="run-1", source="video", sequence=0),
            run_id="pulse-0",
            session_id="session-1",
            fields=(),
            outputs=outputs,
            selections={},
            statuses={"parts": "complete"},
            filtered_paths={"parts": ()},
            plan=None,
            causes=(),
        )
    )
    writer.finish("complete")

    recording = open_recording(tmp_path / "rec", codecs=REGISTRY)
    (chunk,) = recording.group("parts").iter_chunks()
    recorded = chunk.values["parts"]

    assert recorded == value
    assert recorded.known_indices == ((0,), (1,), (2,))
    assert recorded.layout == layout
