"""``v2/detections_filter@v1`` and ``v2/detections_transformation@v1`` on every carrier.

The scene is described in ``scene.py``. Cars are rows 0 and 2.
"""

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.query import QueryRuntimeError
from roboflow_workflows.v2.blocks.transforms import (
    DetectionsFilterBlock,
    DetectionsTransformationBlock,
)
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.rows import PredictionRows

from . import scene

CLASS_IN = {
    "type": "DetectionsFilter",
    "filter_operation": {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {
                    "type": "DynamicOperand",
                    "operations": [
                        {"type": "ExtractDetectionProperty", "property_name": "class_name"}
                    ],
                },
                "comparator": {"type": "in (Sequence)"},
                "right_operand": {"type": "DynamicOperand", "operand_name": "classes"},
            }
        ],
    },
}  # fmt: skip
SORT_DOWN = {"type": "SortDetections", "mode": "confidence", "ascending": False}
CARS = {"classes": ["car"]}


def run_chain(block_class, predictions, operations, parameters=None):
    params = {
        "predictions": "$inputs.predictions",
        "operations": operations,
        "operations_parameters": {name: "$inputs." + name for name in parameters or {}},
    }
    validated = block_class.Params.model_validate(params)
    result = block_class().run(
        predictions=predictions,
        operations=validated.operations,
        operations_parameters=dict(parameters or {}),
    )
    return result["predictions"]


@pytest.mark.parametrize(
    "make",
    [scene.boxes, scene.dense, scene.rle, scene.view, scene.key_points, scene.rows],
)
def test_filter_keeps_carrier_and_row_identity(make) -> None:
    # given
    predictions = make()

    # when
    kept = run_chain(DetectionsFilterBlock, predictions, [CLASS_IN], CARS)

    # then
    assert type(kept) is type(predictions)
    assert scene.row_ids(kept) == [0, 2]


def test_filter_gathers_dense_masks_with_their_rows() -> None:
    # when
    kept = run_chain(DetectionsFilterBlock, scene.dense(), [CLASS_IN], CARS)

    # then
    assert kept.mask.nonzero().tolist() == [[0, 0, 0], [1, 2, 2]]


def test_filter_gathers_rle_masks_with_their_rows() -> None:
    # when
    kept = run_chain(DetectionsFilterBlock, scene.rle(), [CLASS_IN], CARS)

    # then
    assert kept.mask.masks == [b"r0", b"r2"]


def test_view_rows_keep_compact_scores_and_are_never_upscaled() -> None:
    # given
    view = scene.view()

    # when
    kept = run_chain(DetectionsTransformationBlock, view, [CLASS_IN, SORT_DOWN], CARS)

    # then
    assert isinstance(kept, SegmentationView)
    assert torch.equal(kept.scores, view.scores[[0, 2]])
    assert kept.materialization_count == 0
    assert not kept.is_full_res_materialized()


def test_transformation_sort_moves_keypoints_and_columns_with_rows() -> None:
    # when
    sorted_points = run_chain(
        DetectionsTransformationBlock, scene.key_points(), [SORT_DOWN]
    )
    sorted_rows = run_chain(DetectionsTransformationBlock, scene.rows(), [SORT_DOWN])

    # then
    assert scene.row_ids(sorted_points) == [0, 2, 1, 3]
    assert sorted_points[0].xy[:, 0, 0].tolist() == [0.0, 8.0, 4.0, 12.0]
    assert isinstance(sorted_rows, PredictionRows)
    assert sorted_rows.columns["tracker_id"].tolist() == [7, 9, 8, 10]


def test_transformation_rename_shift_and_parent_pick_keep_rows() -> None:
    # given
    operations = [
        {
            "type": "DetectionsRename",
            "class_map": {"car": "vehicle"},
            "strict": False,
        },
        {"type": "DetectionsShift", "shift_x": 10, "shift_y": 0},
        {"type": "PickDetectionsByParentClass", "parent_class": "bus"},
    ]

    # when
    picked = run_chain(DetectionsTransformationBlock, scene.dense(), operations)

    # then
    assert scene.row_ids(picked) == [3, 1]
    assert picked.xyxy[:, 0].tolist() == [110.0, 120.0]
    assert picked.image_metadata["class_names"][picked.class_id[1].item()] == "person"


def test_empty_predictions_stay_empty() -> None:
    # when
    kept = run_chain(DetectionsFilterBlock, scene.empty(), [CLASS_IN, SORT_DOWN], CARS)

    # then
    assert kept.xyxy.shape == (0, 4)
    assert scene.row_ids(kept) == []


def test_literal_parameter_values_and_parameter_reuse() -> None:
    # given
    params = DetectionsFilterBlock.Params.model_validate(
        {
            "predictions": "$inputs.predictions",
            "operations": [CLASS_IN],
            "operations_parameters": {"classes": ["bus"]},
        }
    )
    block = DetectionsFilterBlock()

    # when
    first = block.run(
        predictions=scene.boxes(),
        operations=params.operations,
        operations_parameters={"classes": ["bus"]},
    )
    second = block.run(
        predictions=scene.boxes(),
        operations=params.operations,
        operations_parameters={"classes": ["person"]},
    )

    # then
    assert scene.row_ids(first["predictions"]) == [3]
    assert scene.row_ids(second["predictions"]) == [1]


@pytest.mark.parametrize(
    "operations, parameters, message",
    [
        ([CLASS_IN], {}, "operand `classes` is not declared"),
        ([CLASS_IN], {"classes": 1, "_": 2}, "reserved for the detection"),
        (
            [{"type": "DetectionsToDictionary"}],
            {},
            "DetectionsToDictionary does not return detections",
        ),
        ([{"type": "SortDetections", "mode": "colour"}], {}, "mode"),
        ({"type": "SortDetections"}, {}, "list"),
    ],
)
def test_invalid_chains_fail_when_parameters_are_validated(
    operations, parameters, message
) -> None:
    # when
    with pytest.raises(ValidationError) as error:
        DetectionsTransformationBlock.Params.model_validate(
            {
                "predictions": "$inputs.predictions",
                "operations": operations,
                "operations_parameters": parameters,
            }
        )

    # then
    assert message in str(error.value)


def test_runtime_failure_names_the_step_and_detection() -> None:
    # given
    greater_than_limit = {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operations": [
                            {"type": "ExtractDetectionProperty", "property_name": "confidence"}
                        ],
                    },
                    "comparator": {"type": "(Number) >"},
                    "right_operand": {"type": "DynamicOperand", "operand_name": "limit"},
                }
            ],
        },
    }  # fmt: skip

    # when
    with pytest.raises(QueryRuntimeError) as error:
        run_chain(
            DetectionsFilterBlock, scene.boxes(), [greater_than_limit], {"limit": "x"}
        )

    # then
    assert "$.operations[0]" in str(error.value)
    assert "detection 0" in str(error.value)
