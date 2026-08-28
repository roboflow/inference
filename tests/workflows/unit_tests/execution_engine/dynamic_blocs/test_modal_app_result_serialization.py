"""Wire contract between the client and the Modal sandbox for batch-shaped data.

Two halves, both previously broken:

* **Inputs.** A block declaring ``batch_oriented_parameters`` receives a ``Batch``.
  ``Batch`` subclasses neither ``list`` nor ``dict``, so it used to fall through to
  the generic encoder and arrive as ``str(obj)`` — the block saw a repr, not data.
* **Results.** A ``BlockResult`` is a LIST whenever the block increases output
  dimensionality (offset-1) or is batch-oriented. The sandbox handed that to a
  dict-only serialiser, so those blocks failed on the return trip with
  ``AttributeError: 'list' object has no attribute 'items'``.

These drive the real ``Executor.execute_block`` end to end — user code is compiled
and run in-process — so they cover the shipped path, including the nested closures
that are otherwise unreachable. ``modal_app_with_fake_modal`` lives in conftest.
"""

import asyncio
import json

import msgpack
import numpy as np

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    serialize_for_modal_remote_execution,
    serialize_inputs_for_msgpack,
)


class _FakeRequest:
    """Minimal stand-in for starlette's Request: body() + headers.get()."""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()
        self.headers = {}

    async def body(self) -> bytes:
        return self._body


def _run_block(module, code: str, inputs_json: str = "{}") -> dict:
    executor = module.Executor.__new__(module.Executor)
    executor._code_namespaces = {}
    executor._shared_globals = {}
    request = _FakeRequest(
        {
            "code_str": code,
            "imports": [],
            "run_function_name": "run",
            "inputs_json": inputs_json,
        }
    )
    return asyncio.run(executor.execute_block(request))


def _image(seed: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"crop_{seed}"),
        numpy_image=np.full((4, 4, 3), seed, dtype=np.uint8),
    )


# --------------------------------------------------------------------------- #
# inputs: a Batch must arrive at user code as a Batch
# --------------------------------------------------------------------------- #


def test_batch_of_images_arrives_in_user_code_as_a_batch(
    modal_app_with_fake_modal,
) -> None:
    """The assertion this fix exists for: the block sees a real ``Batch``."""
    batch = Batch(content=[_image(1), _image(2)], indices=[(0,), (1,)])
    inputs_json = serialize_for_modal_remote_execution({"image": batch})

    response = _run_block(
        modal_app_with_fake_modal,
        "def run(image):\n"
        "    return {\n"
        "        'kind': type(image).__name__,\n"
        "        'n': len(image),\n"
        "        'idx': [list(i) for i in image.indices],\n"
        "        'shapes': [i.numpy_image.shape for i in image],\n"
        "    }",
        inputs_json=inputs_json,
    )

    assert response["success"] is True, response.get("error")
    result = json.loads(response["result"])
    assert result["kind"] == "Batch"
    assert result["n"] == 2
    assert result["idx"] == [[0], [1]]
    assert result["shapes"] == [[4, 4, 3], [4, 4, 3]]


def test_batch_of_scalars_arrives_in_user_code_as_a_batch(
    modal_app_with_fake_modal,
) -> None:
    """Non-image batch parameters regressed identically (e.g. per-crop fractions)."""
    batch = Batch(content=[0.357, 0.642], indices=[(0,), (1,)])
    inputs_json = serialize_for_modal_remote_execution({"box_top_frac": batch})

    response = _run_block(
        modal_app_with_fake_modal,
        "def run(box_top_frac):\n"
        "    return {\n"
        "        'kind': type(box_top_frac).__name__,\n"
        "        'values': list(box_top_frac),\n"
        "    }",
        inputs_json=inputs_json,
    )

    assert response["success"] is True, response.get("error")
    result = json.loads(response["result"])
    assert result["kind"] == "Batch"
    assert result["values"] == [0.357, 0.642]


def _through_msgpack(payload: dict) -> dict:
    """Pack and unpack for real.

    Without this the serialiser's output is handed straight back to the decoder
    in-process, so an unserialised ``Batch`` would survive by object identity and
    the test would pass even with the packing arm removed. Going through bytes
    forces the wire format to actually carry the data.
    """
    return msgpack.unpackb(msgpack.packb(payload, use_bin_type=True), raw=False)


def test_msgpack_input_round_trip_rebuilds_a_batch(
    modal_app_with_fake_modal,
) -> None:
    """The websocket arm: silently reintroduces the bug if left unfixed."""
    batch = Batch(content=[_image(3), _image(4)], indices=[(0,), (1,)])

    wire = _through_msgpack(serialize_inputs_for_msgpack({"image": batch}))
    decoded = modal_app_with_fake_modal.Executor._deserialize_msgpack_inputs(wire)

    rebuilt = decoded["image"]
    assert isinstance(rebuilt, Batch)
    assert len(rebuilt) == 2
    # DynamicBatchIndex is `tuple`; Batch.remove_by_indices matches against a
    # Set[tuple], so lists here would silently never match.
    assert rebuilt.indices == [(0,), (1,)]
    assert all(isinstance(index, tuple) for index in rebuilt.indices)
    assert [i.numpy_image.shape for i in rebuilt] == [(4, 4, 3), (4, 4, 3)]


def test_batch_round_trip_preserves_empty_indices(
    modal_app_with_fake_modal,
) -> None:
    """An empty `indices` list must not collapse to None."""
    wire = _through_msgpack(
        serialize_inputs_for_msgpack({"image": Batch(content=[], indices=[])})
    )
    decoded = modal_app_with_fake_modal.Executor._deserialize_msgpack_inputs(wire)

    rebuilt = decoded["image"]
    assert isinstance(rebuilt, Batch)
    assert rebuilt.indices == []


# --------------------------------------------------------------------------- #
# results: list-shaped BlockResults must serialise
# --------------------------------------------------------------------------- #


def test_execute_block_serialises_list_shaped_result(
    modal_app_with_fake_modal,
) -> None:
    """offset-1 / batch-oriented blocks return one entry per element."""
    response = _run_block(
        modal_app_with_fake_modal,
        "def run():\n"
        "    return [{'measurement': {'w': 1}}, {'measurement': {'w': 2}}]",
    )

    assert response["success"] is True, response.get("error")
    assert json.loads(response["result"]) == [
        {"measurement": {"w": 1}},
        {"measurement": {"w": 2}},
    ]


def test_execute_block_list_result_survives_nested_values(
    modal_app_with_fake_modal,
) -> None:
    response = _run_block(
        modal_app_with_fake_modal,
        "def run():\n    return [{'v': [1, 2]}, {'v': []}]",
    )

    assert response["success"] is True, response.get("error")
    assert json.loads(response["result"]) == [{"v": [1, 2]}, {"v": []}]


def test_execute_block_dict_shaped_result_is_unchanged(
    modal_app_with_fake_modal,
) -> None:
    """The ordinary (offset-0) contract must keep byte-identical behaviour."""
    response = _run_block(
        modal_app_with_fake_modal,
        "def run():\n    return {'measurement': {'w': 1}}",
    )

    assert response["success"] is True, response.get("error")
    assert json.loads(response["result"]) == {"measurement": {"w": 1}}
