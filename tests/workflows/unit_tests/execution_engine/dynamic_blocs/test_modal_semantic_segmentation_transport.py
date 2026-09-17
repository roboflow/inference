"""Semantic-segmentation predictions across the Modal custom-block boundary.

Semantic segmentation blocks emit ``sv.Detections`` with ``mask=None``, COCO RLE
masks in ``data["rle_mask"]`` and (historically) no ``image_dimensions``. The
generic serialiser dropped the RLE and produced ``image: {width: None,
height: None}``; ``sv.Detections.from_inference`` then raised inside the
websocket handler, outside the user-code guard, and the socket closed with no
response ("WebSocket connection to Modal endpoint lost after the request was
sent"). ``modal_app_with_fake_modal`` lives in conftest.
"""

import asyncio
import json
from uuid import uuid4

import msgpack
import numpy as np
import supervision as sv
from fastapi.testclient import TestClient
from pycocotools import mask as mask_utils

from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections_for_transport,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    RLE_MASK_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    _deserialize_msgpack_result,
    serialize_for_modal_remote_execution,
    serialize_inputs_for_msgpack,
)

from .conftest import build_ws_app

H, W = 12, 16

COVERAGE_CODE = """
def run(self, predictions):
    rles = predictions.data.get("rle_mask")
    rle = rles[list(predictions.data["class_name"]).index("unharvested")]
    total = int(rle["size"][0]) * int(rle["size"][1])
    foreground = int(predictions.mask[1].sum()) if predictions.mask is not None else -1
    return {"total": total, "foreground": foreground, "n": len(predictions)}
"""


def _semantic_segmentation_detections(with_image_dims: bool = False) -> sv.Detections:
    label = np.zeros((H, W), dtype=np.uint8)
    label[2:8, 3:11] = 1  # 48 px of class 1 ("unharvested")
    xyxy, cids, names, rles = [], [], [], []
    for cid, name in [(0, "harvested"), (1, "unharvested")]:
        binary = label == cid
        rows = np.where(np.any(binary, axis=1))[0]
        cols = np.where(np.any(binary, axis=0))[0]
        xyxy.append([cols[0], rows[0], cols[-1], rows[-1]])
        cids.append(cid)
        names.append(name)
        rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    data = {
        "class_name": np.array(names),
        DETECTION_ID_KEY: np.array([str(uuid4()) for _ in cids]),
        RLE_MASK_KEY_IN_SV_DETECTIONS: np.array(rles, dtype=object),
    }
    if with_image_dims:
        data[IMAGE_DIMENSIONS_KEY] = np.array([[H, W]] * len(cids))
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float64),
        mask=None,
        class_id=np.array(cids),
        confidence=np.ones(len(cids), dtype=np.float32),
        data=data,
    )


def _through_msgpack(payload: dict) -> dict:
    return msgpack.unpackb(msgpack.packb(payload, use_bin_type=True), raw=False)


def _ws_executor(module):
    executor = module.Executor.__new__(module.Executor)
    executor._code_namespaces = {}
    executor._shared_globals = {}
    return executor


# --------------------------------------------------------------------------- #
# serializer: RLE kept, image size recovered from it
# --------------------------------------------------------------------------- #


def test_transport_serializer_keeps_rle_and_recovers_image_size() -> None:
    serialised = serialise_sv_detections_for_transport(
        _semantic_segmentation_detections(with_image_dims=False)
    )

    assert serialised["image"] == {"width": W, "height": H}
    assert [p["rle_mask"]["size"] for p in serialised["predictions"]] == [[H, W]] * 2
    assert all("points" not in p for p in serialised["predictions"])


def test_transport_serializer_falls_back_to_plain_detections() -> None:
    detections = sv.Detections(
        xyxy=np.array([[1.0, 2.0, 5.0, 6.0]]),
        class_id=np.array([0]),
        confidence=np.array([0.9], dtype=np.float32),
        data={
            "class_name": np.array(["a"]),
            DETECTION_ID_KEY: np.array(["d1"]),
            IMAGE_DIMENSIONS_KEY: np.array([[H, W]]),
        },
    )

    serialised = serialise_sv_detections_for_transport(detections)

    assert serialised["image"] == {"width": W, "height": H}
    assert "rle_mask" not in serialised["predictions"][0]


# --------------------------------------------------------------------------- #
# websocket transport: inputs round trip with the RLE, and user code sees it
# --------------------------------------------------------------------------- #


def test_ws_inputs_round_trip_semantic_segmentation_without_image_dims(
    modal_app_with_fake_modal,
) -> None:
    wire = _through_msgpack(
        serialize_inputs_for_msgpack(
            {"predictions": _semantic_segmentation_detections(with_image_dims=False)}
        )
    )

    decoded = modal_app_with_fake_modal.Executor._deserialize_msgpack_inputs(wire)

    rebuilt = decoded["predictions"]
    assert isinstance(rebuilt, sv.Detections)
    assert len(rebuilt) == 2
    assert list(rebuilt.data["class_name"]) == ["harvested", "unharvested"]
    assert rebuilt.data[RLE_MASK_KEY_IN_SV_DETECTIONS][1]["size"] == [H, W]
    assert (rebuilt.data[IMAGE_DIMENSIONS_KEY] == [[H, W], [H, W]]).all()


def test_ws_user_code_receives_rle_masks(modal_app_with_fake_modal) -> None:
    module = modal_app_with_fake_modal
    wire = _through_msgpack(
        serialize_inputs_for_msgpack(
            {"predictions": _semantic_segmentation_detections()}
        )
    )
    inputs = module.Executor._deserialize_msgpack_inputs(wire)

    response = module.Executor._run_user_code_ws(
        _ws_executor(module), COVERAGE_CODE, [], "run", inputs
    )

    assert response["success"], response
    assert response["result"] == {"total": H * W, "foreground": 48, "n": 2}


def test_ws_result_round_trip_semantic_segmentation(
    modal_app_with_fake_modal,
) -> None:
    encoded = modal_app_with_fake_modal.Executor._serialize_msgpack_result(
        {"predictions": _semantic_segmentation_detections()}
    )

    decoded = _deserialize_msgpack_result(_through_msgpack(encoded))

    rebuilt = decoded["predictions"]
    assert len(rebuilt) == 2
    assert rebuilt.data[RLE_MASK_KEY_IN_SV_DETECTIONS][1]["size"] == [H, W]


# --------------------------------------------------------------------------- #
# websocket handler: a deserialization failure is an error frame, not a hangup
# --------------------------------------------------------------------------- #


def test_ws_handler_reports_input_deserialization_failure_instead_of_closing(
    modal_app_with_fake_modal,
) -> None:
    """The failure mode behind the original bug: detections the server cannot
    rebuild must come back as an error frame, and the socket must survive."""
    module = modal_app_with_fake_modal
    _, ws_app = build_ws_app(module, module.Executor._run_user_code_ws)
    # `predictions` is tagged as detections but lacks the required keys.
    broken_frame = msgpack.packb(
        {
            "code_str": "def run(self, predictions):\n    return {'ok': True}\n",
            "imports": [],
            "run_function_name": "run",
            "inputs": {"predictions": {"_type": "sv_detections", "image": None}},
            "code_hash": "",
            "workflow_context": {},
        },
        use_bin_type=True,
    )

    with TestClient(ws_app).websocket_connect("/ws") as websocket:
        websocket.send_bytes(broken_frame)
        first = msgpack.unpackb(websocket.receive_bytes(), raw=False)
        # The socket must survive the failure: a good frame still gets served.
        websocket.send_bytes(
            msgpack.packb(
                {
                    "code_str": "def run(self, a):\n    return {'a': a}\n",
                    "imports": [],
                    "run_function_name": "run",
                    "inputs": {"a": 1},
                    "code_hash": "",
                    "workflow_context": {},
                },
                use_bin_type=True,
            )
        )
        second = msgpack.unpackb(websocket.receive_bytes(), raw=False)

    assert first["success"] is False
    assert first["server_error"] is True
    assert "could not decode this request's inputs" in first["error"]
    assert second["success"] is True
    assert second["result"] == {"a": 1}


# --------------------------------------------------------------------------- #
# HTTP transport: same predictions must survive the JSON path too
# --------------------------------------------------------------------------- #


def test_http_inputs_round_trip_semantic_segmentation(
    modal_app_with_fake_modal,
) -> None:
    module = modal_app_with_fake_modal
    executor = _ws_executor(module)

    class _FakeRequest:
        def __init__(self, payload: dict):
            self._body = json.dumps(payload).encode()
            self.headers = {}

        async def body(self) -> bytes:
            return self._body

    inputs_json = serialize_for_modal_remote_execution(
        {"predictions": _semantic_segmentation_detections()}
    )
    request = _FakeRequest(
        {
            "code_str": COVERAGE_CODE,
            "imports": [],
            "run_function_name": "run",
            "inputs_json": inputs_json,
        }
    )

    response = asyncio.run(executor.execute_block(request))

    assert response["success"], response
    # The HTTP transport ships the result as a JSON string.
    assert json.loads(response["result"]) == {"total": H * W, "foreground": 48, "n": 2}
