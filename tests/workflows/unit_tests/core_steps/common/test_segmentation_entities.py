"""The segmentation prediction classes live in workflows; the server re-exports them.

`Point`, `InstanceSegmentationBasePrediction`, `InstanceSegmentationPrediction`,
`InstanceSegmentationRLEPrediction` (server: `responses/inference.py`) and
`Sam2SegmentationPrediction` (server: `responses/sam2.py`) are ONE class object
per name. The blocks build them from remote responses as well as from their own
arithmetic, so they keep pydantic coercion and validation - the interactive
parser's string confidence becomes a float, a malformed mask is a
`ValidationError` (round-6 defect 1). The five response-level DTOs
(`InferenceResponseImage`, `InferenceResponse`, `CvInferenceResponse`,
`WithVisualizationResponse`, `InstanceSegmentationInferenceResponse`) also
live in workflows and the server re-exports them; the workflow blocks build
them from locals and dump them with
`model_dump(by_alias=True, exclude_none=True)`.
"""

import ast
import base64
from pathlib import Path
from typing import List, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.entities.responses import inference as server_inference
from inference.core.entities.responses import sam2 as server_sam2
from inference.core.workflows.core_steps.common import (
    inference_response_entities as local_response,
)
from inference.core.workflows.core_steps.common import segmentation_entities as local
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive import (
    v1 as interactive_v1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)

# tests/workflows/unit_tests/core_steps/common/<file> -> parents[5] is the repo root
ENTITIES_MODULE = (
    Path(__file__).resolve().parents[5]
    / "workflows/roboflow_workflows/core_steps/common/segmentation_entities.py"
)
RESPONSE_ENTITIES_MODULE = (
    Path(__file__).resolve().parents[5]
    / "workflows/roboflow_workflows/core_steps/common/inference_response_entities.py"
)
_POINTS = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
PAIRS = [
    (local.Point, server_inference.Point),
    (
        local.InstanceSegmentationBasePrediction,
        server_inference.InstanceSegmentationBasePrediction,
    ),
    (
        local.InstanceSegmentationPrediction,
        server_inference.InstanceSegmentationPrediction,
    ),
    (
        local.InstanceSegmentationRLEPrediction,
        server_inference.InstanceSegmentationRLEPrediction,
    ),
    (local.Sam2SegmentationPrediction, server_sam2.Sam2SegmentationPrediction),
    (local_response.InferenceResponseImage, server_inference.InferenceResponseImage),
    (local_response.InferenceResponse, server_inference.InferenceResponse),
    (local_response.CvInferenceResponse, server_inference.CvInferenceResponse),
    (
        local_response.WithVisualizationResponse,
        server_inference.WithVisualizationResponse,
    ),
    (
        local_response.InstanceSegmentationInferenceResponse,
        server_inference.InstanceSegmentationInferenceResponse,
    ),
]


@pytest.mark.parametrize(
    "local_cls,server_cls", PAIRS, ids=[pair[1].__name__ for pair in PAIRS]
)
def test_server_module_re_exports_the_workflows_class(local_cls, server_cls) -> None:
    assert local_cls is server_cls


def test_server_classes_still_derive_from_and_hold_the_workflows_classes() -> None:
    assert issubclass(server_inference.Point3D, local.Point)
    assert issubclass(server_inference.Keypoint, local.Point)
    assert issubclass(
        server_inference.ObjectDetectionInferenceResponse,
        local_response.CvInferenceResponse,
    )
    assert issubclass(
        server_inference.ObjectDetectionInferenceResponse,
        local_response.WithVisualizationResponse,
    )
    assert (
        server_inference.InstanceSegmentationInferenceResponse.model_fields[
            "predictions"
        ].annotation
        == List[
            Union[
                local.InstanceSegmentationPrediction,
                local.InstanceSegmentationRLEPrediction,
            ]
        ]
    )
    assert (
        server_sam2.Sam2SegmentationResponse.model_fields["predictions"].annotation
        == List[local.Sam2SegmentationPrediction]
    )
    assert interactive_v1.Sam2SegmentationPrediction is local.Sam2SegmentationPrediction


def test_entities_module_imports_nothing_from_the_server() -> None:
    assert ENTITIES_MODULE.is_file(), ENTITIES_MODULE
    tree = ast.parse(ENTITIES_MODULE.read_text(encoding="utf-8"))
    imported = sorted(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported == ["pydantic", "typing", "uuid"], imported


def test_response_entities_module_imports_nothing_from_the_server() -> None:
    assert RESPONSE_ENTITIES_MODULE.is_file(), RESPONSE_ENTITIES_MODULE
    tree = ast.parse(RESPONSE_ENTITIES_MODULE.read_text(encoding="utf-8"))
    imported = sorted(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported == [
        "pydantic",
        "roboflow_workflows.core_steps.common.segmentation_entities",
        "typing",
    ], imported


# --- the remote parser, through the function and through BOTH remote branches ---

PARSER_TABLE = [
    (
        "str confidence",
        {"predictions": [{"masks": [[[0, 0], [4, 0], [4, 4]]], "confidence": "0.9"}]},
        ("ok", [(0.9, "float", [[[0, 0], [4, 0], [4, 4]]])]),
    ),
    (
        "int confidence",
        {"predictions": [{"masks": [[[0, 0], [4, 0], [4, 4]]], "confidence": 1}]},
        ("ok", [(1.0, "float", [[[0, 0], [4, 0], [4, 4]]])]),
    ),
    (
        "missing confidence",
        {"predictions": [{"masks": [[[0, 0], [4, 0], [4, 4]]]}]},
        ("ok", [(0.0, "float", [[[0, 0], [4, 0], [4, 4]]])]),
    ),
    (
        "missing masks",
        {"predictions": [{"confidence": 0.5}]},
        ("ok", [(0.5, "float", [])]),
    ),
    (
        "rle masks",
        {
            "predictions": [
                {"masks": {"size": [2, 2], "counts": "abc"}, "confidence": 0.5}
            ]
        },
        ("ok", [(0.5, "float", {"size": [2, 2], "counts": "abc"})]),
    ),
    (
        "flat list response",
        [{"masks": [[[0, 0], [4, 0], [4, 4]]], "confidence": "0.25"}],
        ("ok", [(0.25, "float", [[[0, 0], [4, 0], [4, 4]]])]),
    ),
    (
        "masks not a list",
        {"predictions": [{"masks": "oops", "confidence": 0.9}]},
        (
            "error",
            "2 validation errors for Sam2SegmentationPrediction",
            [
                (("masks", "list[list[list[int]]]"), "list_type"),
                (("masks", "dict[str,any]"), "dict_type"),
            ],
        ),
    ),
    (
        "non-numeric points",
        {"predictions": [{"masks": [[["a", "b"]]], "confidence": 0.9}]},
        (
            "error",
            "3 validation errors for Sam2SegmentationPrediction",
            [
                (("masks", "list[list[list[int]]]", 0, 0, 0), "int_parsing"),
                (("masks", "list[list[list[int]]]", 0, 0, 1), "int_parsing"),
                (("masks", "dict[str,any]"), "dict_type"),
            ],
        ),
    ),
    (
        "non-numeric confidence",
        {"predictions": [{"masks": [[[0, 0], [4, 0], [4, 4]]], "confidence": "high"}]},
        (
            "error",
            "1 validation error for Sam2SegmentationPrediction",
            [(("confidence",), "float_parsing")],
        ),
    ),
]


def _outcome(build):
    try:
        value = build()
    except ValidationError as error:
        return (
            "error",
            str(error).splitlines()[0],
            [(e["loc"], e["type"]) for e in error.errors()],
        )
    return ("ok", [(p.confidence, type(p.confidence).__name__, p.masks) for p in value])


@pytest.mark.parametrize(
    "payload,expected", [c[1:] for c in PARSER_TABLE], ids=[c[0] for c in PARSER_TABLE]
)
def test_parse_segmentation_predictions_coerces_and_rejects_as_before_the_move(
    payload, expected
) -> None:
    assert (
        _outcome(lambda: interactive_v1._parse_segmentation_predictions(payload))
        == expected
    )


def _one_image_batch():
    return Batch(
        content=[
            WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id="p"),
                numpy_image=np.zeros((20, 10, 3), dtype=np.uint8),
            )
        ],
        indices=[(0,)],
    )


def _run_remote_branch(branch, payload):
    """`sdk` = run_remotely (InferenceHTTPClient.sam3_visual_segment, v1.py:411);
    `proxy` = run_via_request (requests.post, v1.py:492)."""
    if branch == "sdk":
        with patch.object(interactive_v1, "InferenceHTTPClient") as client_cls:
            client_cls.return_value.sam3_visual_segment.return_value = payload
            block = interactive_v1.SegmentAnything3InteractiveBlockV1(
                model_manager=MagicMock(),
                api_key="k",
                step_execution_mode=StepExecutionMode.REMOTE,
            )
            return block.run_remotely(
                images=_one_image_batch(),
                points=[{"x": 1, "y": 1}],
                boxes=None,
                threshold=0.5,
                multimask_output=True,
            )
    response = MagicMock()
    response.json.return_value = payload
    with patch.object(interactive_v1.requests, "post", return_value=response):
        block = interactive_v1.SegmentAnything3InteractiveBlockV1(
            model_manager=MagicMock(),
            api_key="k",
            step_execution_mode=StepExecutionMode.LOCAL,
        )
        return block.run_via_request(
            images=_one_image_batch(),
            points=[{"x": 1, "y": 1}],
            boxes=None,
            threshold=0.5,
            multimask_output=True,
        )


@pytest.mark.parametrize("branch", ["sdk", "proxy"])
def test_remote_branch_coerces_a_string_confidence_into_a_detection(branch) -> None:
    """A dataclass carrier left "0.9" a str and `prediction.confidence < threshold`
    raised TypeError in the SAM2 converter (round-6 defect 1)."""
    result = _run_remote_branch(
        branch,
        {"predictions": [{"masks": [[[0, 0], [4, 0], [4, 4]]], "confidence": "0.9"}]},
    )
    detections = result[0]["predictions"]
    assert len(detections) == 1
    assert detections.confidence.tolist() == [pytest.approx(0.9)]


@pytest.mark.parametrize("branch", ["sdk", "proxy"])
def test_remote_branch_rejects_malformed_masks_with_the_pydantic_error(branch) -> None:
    with pytest.raises(ValidationError) as error:
        _run_remote_branch(
            branch, {"predictions": [{"masks": "oops", "confidence": 0.9}]}
        )
    assert str(error.value).startswith(
        "2 validation errors for Sam2SegmentationPrediction\nmasks.list[list[list[int]]]\n"
    )


# --- the moved prediction classes: frozen coercion / rejection ---

CLASS_TABLE = [
    (
        "Point str coords",
        local.Point,
        {"x": "1", "y": "2.5"},
        ("ok", {"x": 1.0, "y": 2.5}),
    ),
    ("Point missing y", local.Point, {"x": 1}, ("error", [(("y",), "missing")])),
    (
        "prediction alias + str confidence",
        local.InstanceSegmentationPrediction,
        {
            "x": 1,
            "y": 2,
            "width": 3,
            "height": 4,
            "confidence": "0.5",
            "class": "cat",
            "class_id": 0,
            "points": [{"x": 0, "y": 0}],
            "detection_id": "d",
        },
        (
            "ok",
            {
                "x": 1.0,
                "y": 2.0,
                "width": 3.0,
                "height": 4.0,
                "confidence": 0.5,
                "class": "cat",
                "class_id": 0,
                "detection_id": "d",
                "points": [{"x": 0.0, "y": 0.0}],
                "mask_format": "polygon",
            },
        ),
    ),
    (
        "prediction missing points",
        local.InstanceSegmentationPrediction,
        {
            "x": 1,
            "y": 2,
            "width": 3,
            "height": 4,
            "confidence": 0.5,
            "class": "cat",
            "class_id": 0,
        },
        ("error", [(("points",), "missing")]),
    ),
    (
        "rle prediction rle not a dict",
        local.InstanceSegmentationRLEPrediction,
        {
            "x": 1,
            "y": 2,
            "width": 3,
            "height": 4,
            "confidence": 0.5,
            "class": "cat",
            "class_id": 0,
            "rle": "x",
        },
        ("error", [(("rle",), "dict_type")]),
    ),
]


@pytest.mark.parametrize(
    "cls,kwargs,expected", [c[1:] for c in CLASS_TABLE], ids=[c[0] for c in CLASS_TABLE]
)
def test_moved_prediction_classes_validate_as_before_the_move(
    cls, kwargs, expected
) -> None:
    try:
        outcome = ("ok", cls(**kwargs).model_dump(by_alias=True, exclude_none=True))
    except ValidationError as error:
        outcome = ("error", [(e["loc"], e["type"]) for e in error.errors()])
    assert outcome == expected


# --- response-level Pydantic models: dump, optional fields, serialization ---


def _prediction():
    return local.InstanceSegmentationPrediction(
        **{
            "x": 5.0,
            "y": 5.0,
            "width": 10.0,
            "height": 10.0,
            "confidence": 0.9,
            "class": "cat",
            "class_id": 3,
            "points": [local.Point(x=px, y=py) for px, py in _POINTS],
            "detection_id": "fixed",
        }
    )


def _rle_prediction():
    return local.InstanceSegmentationRLEPrediction(
        **{
            "x": 5.0,
            "y": 5.0,
            "width": 10.0,
            "height": 10.0,
            "confidence": 0.9,
            "class": "cat",
            "class_id": 3,
            "rle": {"size": [10, 10], "counts": "abc"},
            "detection_id": "fixed",
        }
    )


def test_image_response_dump_uses_by_alias_exclude_none() -> None:
    dumped = local_response.InferenceResponseImage(width=640, height=480).model_dump(
        by_alias=True, exclude_none=True
    )
    assert dumped == {"width": 640, "height": 480}


@pytest.mark.parametrize(
    "predictions",
    [[], [_prediction()], [_rle_prediction()], [_prediction(), _rle_prediction()]],
    ids=["empty", "polygon", "rle", "mixed"],
)
def test_response_dump_omits_optional_fields_when_none(predictions) -> None:
    dumped = local_response.InstanceSegmentationInferenceResponse(
        image=local_response.InferenceResponseImage(width=640, height=480),
        predictions=predictions,
    ).model_dump(by_alias=True, exclude_none=True)
    assert dumped["image"] == {"width": 640, "height": 480}
    assert "inference_id" not in dumped
    assert "frame_id" not in dumped
    assert "time" not in dumped
    assert "visualization" not in dumped
    assert len(dumped["predictions"]) == len(predictions)


def test_response_dump_carries_optionals_when_set() -> None:
    dumped = local_response.InstanceSegmentationInferenceResponse(
        image=local_response.InferenceResponseImage(width=1, height=1),
        predictions=[_prediction()],
        inference_id="i",
        frame_id=3,
        time=0.5,
    ).model_dump(by_alias=True, exclude_none=True)
    assert dumped["inference_id"] == "i"
    assert dumped["frame_id"] == 3
    assert dumped["time"] == 0.5


def test_response_visualization_is_base64_encoded_on_json_dump() -> None:
    payload = b"\x00\x01\x02\x03"
    response = local_response.InstanceSegmentationInferenceResponse(
        image=local_response.InferenceResponseImage(width=1, height=1),
        predictions=[],
        visualization=payload,
    )
    # model_dump() keeps the raw bytes; the JSON serializer encodes them.
    assert response.model_dump()["visualization"] == payload
    encoded = response.model_dump_json()
    import json

    assert json.loads(encoded)["visualization"] == base64.b64encode(payload).decode(
        "utf-8"
    )


def test_cv_response_accepts_list_of_images() -> None:
    """CvInferenceResponse.image is Union[List[InferenceResponseImage], InferenceResponseImage].
    The list form is exercised by multi-image responses in the server code."""
    response = local_response.InstanceSegmentationInferenceResponse(
        image=[
            local_response.InferenceResponseImage(width=10, height=20),
            local_response.InferenceResponseImage(width=30, height=40),
        ],
        predictions=[],
    )
    dumped = response.model_dump(by_alias=True, exclude_none=True)
    assert dumped["image"] == [
        {"width": 10, "height": 20},
        {"width": 30, "height": 40},
    ]
