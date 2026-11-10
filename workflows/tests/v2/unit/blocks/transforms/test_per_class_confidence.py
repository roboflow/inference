"""``roboflow_core/per_class_confidence_filter@v1``: thresholds, names, carriers and errors.

Scene (``scene.py``): car 0.875, person 0.5, car 0.875, bus 0.375.
"""

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.transforms import PerClassConfidenceFilterBlock
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.rows import PredictionRows

from . import scene


def keep(predictions, class_thresholds, default_threshold=0.3):
    result = PerClassConfidenceFilterBlock().run(
        predictions=predictions,
        class_thresholds=class_thresholds,
        default_threshold=default_threshold,
    )
    return result["predictions"]


def test_listed_classes_use_their_threshold_and_others_the_default() -> None:
    # when
    kept = keep(scene.boxes(), {"car": 0.9, "person": 0.5}, default_threshold=0.3)

    # then
    assert scene.row_ids(kept) == [1, 3]


def test_confidence_equal_to_the_threshold_is_kept() -> None:
    # when
    kept = keep(scene.boxes(), {"car": 0.875}, default_threshold=1.0)

    # then
    assert scene.row_ids(kept) == [0, 2]


def test_all_kept_or_empty_returns_the_input_object() -> None:
    # given
    predictions = scene.boxes()
    empty = scene.empty()

    # when / then
    assert keep(predictions, {}, default_threshold=0.0) is predictions
    assert keep(empty, {"car": 1.0}) is empty


def test_per_row_class_name_wins_over_the_class_map() -> None:
    # given: row 1 calls itself a pedestrian; the map says person
    predictions = scene.with_row_classes([None, "pedestrian", None, None])

    # when
    kept = keep(predictions, {"person": 0.0, "pedestrian": 0.9}, default_threshold=0.0)

    # then
    assert scene.row_ids(kept) == [0, 2, 3]


def test_rows_without_a_name_use_the_default_not_a_none_key() -> None:
    # given
    predictions = scene.with_row_classes([None, None, None, None])
    predictions.image_metadata.pop("class_names")

    # when
    kept = keep(predictions, {"None": 0.0}, default_threshold=0.6)

    # then
    assert scene.row_ids(kept) == [0, 2]


@pytest.mark.parametrize("make", [scene.dense, scene.rle, scene.key_points, scene.rows])
def test_carriers_keep_type_and_aligned_fields(make) -> None:
    # when
    kept = keep(make(), {"car": 0.9})

    # then
    assert type(kept) is type(make())
    assert scene.row_ids(kept) == [1, 3]


def test_keypoints_and_columns_follow_rows() -> None:
    # when
    points = keep(scene.key_points(), {"car": 0.9})
    rows = keep(scene.rows(), {"car": 0.9})

    # then
    assert points[0].xy[:, 0, 0].tolist() == [4.0, 12.0]
    assert isinstance(rows, PredictionRows)
    assert rows.columns["tracker_id"].tolist() == [8, 10]


def test_view_is_filtered_without_upscaling() -> None:
    # given
    view = scene.view()

    # when
    kept = keep(view, {"car": 0.9})

    # then
    assert isinstance(kept, SegmentationView)
    assert torch.equal(kept.scores, view.scores[[1, 3]])
    assert kept.materialization_count == 0


def test_keypoints_without_boxes_fail_clearly() -> None:
    # when
    with pytest.raises(ValueError) as error:
        keep(scene.key_points(with_boxes=False), {"car": 0.9})

    # then
    assert "per_class_confidence_filter" in str(error.value)
    assert "no bounding-box Detections" in str(error.value)


@pytest.mark.parametrize(
    "thresholds, default, message",
    [
        ({"car": 1.5}, 0.3, "class_thresholds['car'] must be a number in [0, 1]"),
        ({"car": True}, 0.3, "class_thresholds['car'] must be a number in [0, 1]"),
        ({"car": float("nan")}, 0.3, "must be a number in [0, 1]"),
        ({"car": 0.5}, -0.1, "default_threshold must be a number in [0, 1]"),
        (["car"], 0.3, "must be a dictionary"),
    ],
)
def test_invalid_runtime_thresholds_fail_clearly(thresholds, default, message) -> None:
    # when
    with pytest.raises(ValueError) as error:
        keep(scene.boxes(), thresholds, default_threshold=default)

    # then
    assert message in str(error.value)


@pytest.mark.parametrize(
    "params, message",
    [
        ({"class_thresholds": {"car": 2}}, "must be a number in [0, 1]"),
        ({"class_thresholds": {}, "default_threshold": 5}, "default_threshold"),
        ({"class_thresholds": {"car": "high"}}, "class_thresholds"),
    ],
)
def test_invalid_literal_thresholds_fail_validation(params, message) -> None:
    # when
    with pytest.raises(ValidationError) as error:
        PerClassConfidenceFilterBlock.Params.model_validate(
            {"predictions": "$inputs.predictions", **params}
        )

    # then
    assert message in str(error.value)


def test_selector_thresholds_pass_validation() -> None:
    # when
    params = PerClassConfidenceFilterBlock.Params.model_validate(
        {
            "predictions": "$inputs.predictions",
            "class_thresholds": "$inputs.thresholds",
            "default_threshold": "$inputs.default",
        }
    )

    # then
    assert params.class_thresholds == "$inputs.thresholds"
