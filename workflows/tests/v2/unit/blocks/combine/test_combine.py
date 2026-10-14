"""``roboflow_core/detections_combine@v1`` against the V1 numpy block and analytic values."""

import dataclasses

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.transformations.detections_combine.v1 import (
    DetectionsCombineBlockV1,
)
from roboflow_workflows.v2.blocks.combine import combine_predictions
from roboflow_workflows.v2.blocks.query.detection_ops import derived_detection_id
from roboflow_workflows.v2.blocks.query.detections import DetectionTable
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import FrameMapping
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import row_record

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

from .scene import (
    PEOPLE,
    SIZE_HW,
    STRIDE,
    VEHICLES,
    as_sv,
    box_masks,
    boxes,
    dense,
    frame_metadata,
    grid_geometry,
    rle,
    view,
)


def people_and_vehicles():
    return boxes(PEOPLE, namespace="people"), boxes(VEHICLES, namespace="vehicles")


def static_crop_view(model=VEHICLES, *, namespace="vehicles", crop_x=4):
    """A view on the scene's grid shape whose model saw a static crop.

    The crop starts at ``crop_x`` and is right-padded back to the scene's
    network input, so the grid shape matches ``view()`` while the geometry does
    not. Upscaling stays exactly 2x, so the right masks fill each box.
    """
    height, width = SIZE_HW
    geometry = dataclasses.replace(
        grid_geometry(),
        padding_ltrb=(0, 0, crop_x, 0),
        crop_size_hw=(height, width - crop_x),
        static_crop_xywh=(crop_x, 0, width - crop_x, height),
    )
    detections = boxes(model, namespace=namespace)
    crop = box_masks(detections)[:, :, crop_x:].float().unsqueeze(1)
    cells = torch.nn.functional.max_pool2d(crop, STRIDE).squeeze(1)
    padded = torch.nn.functional.pad(cells, (0, crop_x // STRIDE))
    return SegmentationView(
        detections=detections,
        scores=padded * 16.0 - 8.0,
        score_type="logits",
        threshold=0.0,
        geometry=geometry,
    )


def test_rows_values_and_class_names_match_v1_numpy_block() -> None:
    # given
    people, vehicles = people_and_vehicles()
    expected = DetectionsCombineBlockV1().run(
        prediction_one=as_sv(people), prediction_two=as_sv(vehicles)
    )["predictions"]

    # when
    combined = combine_predictions(people, vehicles)

    # then
    np.testing.assert_array_equal(combined.xyxy.numpy(), expected.xyxy)
    np.testing.assert_array_equal(combined.confidence.numpy(), expected.confidence)
    assert combined.class_id.tolist() == expected.class_id.tolist() == [0, 0, 0, 1, 0]
    names = DetectionTable(combined).class_names(require_all=True)
    assert names == expected.data["class_name"].tolist()
    assert names == ["person", "person", "car", "bus", "car"]
    assert combined.image_metadata["class_names"] == {0: "person", 1: "bus"}
    OBJECT_DETECTION_PREDICTION_KIND.check(combined)


def test_different_results_get_a_fresh_namespace_without_generated_ids() -> None:
    # given
    people, vehicles = people_and_vehicles()
    picked = select_predictions(vehicles, indices=[2, 1])

    # when
    combined = combine_predictions(people, picked)

    # then
    record = row_record(combined)
    assert record.namespace not in ("people", "vehicles")
    assert (record.rows, record.ids) == (4, None)
    assert combined.bboxes_metadata == [{}, {}, {"class": "car"}, {}]
    exported = [derived_detection_id(record.namespace, i) for i in range(4)]
    assert derived_detection_id("people", 0) not in exported
    assert people.bboxes_metadata is None and picked.bboxes_metadata is None


def test_rows_of_one_result_keep_its_namespace_and_ids() -> None:
    # given
    vehicles = boxes(VEHICLES, namespace="vehicles")
    confident = select_predictions(vehicles, indices=[1, 0])
    cars = select_predictions(vehicles, indices=[0, 2])

    # when
    combined = combine_predictions(confident, cars)

    # then
    record = row_record(combined)
    assert record.namespace == "vehicles"
    assert record.host_ids() == [1, 0, 0, 2]
    assert combined.bboxes_metadata is None
    assert combined.image_metadata["class_names"] == VEHICLES["class_names"]


def test_implicit_and_explicit_ids_of_one_result_are_concatenated() -> None:
    # given
    vehicles = boxes(VEHICLES, namespace="vehicles")
    last = select_predictions(vehicles, indices=[2])

    # when
    combined = combine_predictions(vehicles, last)

    # then
    assert row_record(combined).host_ids() == [0, 1, 2, 2]


def test_explicit_row_metadata_is_kept_and_missing_rows_padded() -> None:
    # given
    people, vehicles = people_and_vehicles()
    people = dataclasses.replace(
        people,
        bboxes_metadata=[{"detection_id": "p0", "tag": "a"}, {"tag": "b"}],
    )

    # when
    combined = combine_predictions(people, vehicles)

    # then
    rows = combined.bboxes_metadata
    assert [row.get("tag") for row in rows] == ["a", "b", None, None, None]
    assert rows[0]["detection_id"] == "p0"
    assert all("detection_id" not in row for row in rows[1:])
    assert people.bboxes_metadata == [{"detection_id": "p0", "tag": "a"}, {"tag": "b"}]


def test_a_row_class_name_wins_over_the_conflict_fill() -> None:
    # given
    people, vehicles = people_and_vehicles()
    vehicles = dataclasses.replace(
        vehicles, bboxes_metadata=[{"class": "taxi"}, {}, {}]
    )

    # when
    combined = combine_predictions(people, vehicles)

    # then
    names = DetectionTable(combined).class_names(require_all=True)
    assert names == ["person", "person", "taxi", "bus", "car"]


def test_class_maps_with_json_string_keys_still_clash() -> None:
    # given
    people, vehicles = people_and_vehicles()
    vehicles = dataclasses.replace(
        vehicles,
        image_metadata={
            **vehicles.image_metadata,
            "class_names": {"0": "car", "1": "bus"},
        },
    )

    # when
    combined = combine_predictions(people, vehicles)

    # then
    names = DetectionTable(combined).class_names(require_all=True)
    assert names == ["person", "person", "car", "bus", "car"]
    assert combined.image_metadata["class_names"] == {0: "person", 1: "bus"}


def test_compatible_class_maps_need_no_row_names() -> None:
    # given
    first = boxes(VEHICLES, namespace="a", rows=[0])
    second = boxes(VEHICLES, namespace="b", rows=[1])

    # when
    combined = combine_predictions(first, second)

    # then
    assert combined.bboxes_metadata is None
    assert DetectionTable(combined).class_names(require_all=True) == ["car", "bus"]


@pytest.mark.parametrize("empty_first", [True, False])
def test_an_empty_input_returns_the_other_unchanged(empty_first: bool) -> None:
    # given
    people = boxes(PEOPLE, namespace="people")
    empty = dense(VEHICLES, namespace=None)
    empty = select_predictions(empty, indices=[])
    inputs = (empty, people) if empty_first else (people, empty)
    v1 = DetectionsCombineBlockV1().run(
        prediction_one=as_sv(inputs[0]), prediction_two=as_sv(inputs[1])
    )["predictions"]

    # when
    combined = combine_predictions(*inputs)

    # then
    assert combined is people
    np.testing.assert_array_equal(v1.xyxy, people.xyxy.numpy())


def test_two_empty_inputs_return_prediction_one() -> None:
    # given
    first = boxes(PEOPLE, rows=[])
    second = boxes(VEHICLES, namespace="vehicles", rows=[])

    # when
    combined = combine_predictions(first, second)

    # then
    assert combined is first


def test_different_images_are_rejected() -> None:
    # given
    people = boxes(PEOPLE, namespace="people")
    elsewhere = boxes(VEHICLES, namespace="vehicles", image_id="other-image")

    # when / then
    with pytest.raises(ValueError, match="different images"):
        combine_predictions(people, elsewhere)


def test_a_crop_of_the_same_image_is_rejected_with_a_remedy() -> None:
    # given
    people = boxes(PEOPLE, namespace="people")
    in_street = FrameMapping("street", SIZE_HW, offset_xy=(10.0, 10.0))
    crop = FrameRef(
        image_id="crop-1", size_hw=(20, 30), parent=in_street, root=in_street
    )
    vehicles = boxes(VEHICLES, namespace="vehicles")
    in_crop = dataclasses.replace(
        vehicles,
        image_metadata={**vehicles.image_metadata, **crop.prediction_metadata()},
    )

    # when / then
    with pytest.raises(ValueError) as error:
        combine_predictions(people, in_crop)
    message = str(error.value)
    assert "different frames" in message
    assert "workflow image 'street'" in message
    assert "Run both model steps on the same image or frame" in message
    assert "prediction_to_frame" in message


def test_an_unknown_frame_is_rejected() -> None:
    # given
    people = boxes(PEOPLE, namespace="people")
    bare = dataclasses.replace(
        boxes(VEHICLES, namespace=None), image_metadata={"class_names": {0: "car"}}
    )

    # when / then
    with pytest.raises(ValueError, match="prediction_two has no frame metadata"):
        combine_predictions(people, bare)


def test_different_devices_are_rejected() -> None:
    # given
    people = boxes(PEOPLE, namespace=None)
    vehicles = boxes(VEHICLES, namespace=None)
    vehicles = dataclasses.replace(vehicles, xyxy=vehicles.xyxy.to("meta"))

    # when / then
    with pytest.raises(ValueError) as error:
        combine_predictions(people, vehicles)
    message = str(error.value)
    assert "prediction_one is on cpu and prediction_two on meta" in message
    assert "set model_package_id on the model steps" in message


def test_boxes_and_masks_cannot_mix_as_in_v1() -> None:
    # given
    people = boxes(PEOPLE, namespace="people")
    vehicles = dense(VEHICLES, namespace="vehicles")
    with pytest.raises(ValueError, match="All or none of the 'mask'"):
        DetectionsCombineBlockV1().run(
            prediction_one=as_sv(people),
            prediction_two=as_sv(vehicles, with_masks=True),
        )

    # when / then
    with pytest.raises(ValueError, match="all or none of the predictions must carry"):
        combine_predictions(people, vehicles)
    with pytest.raises(ValueError, match="all or none of the predictions must carry"):
        combine_predictions(view(VEHICLES, namespace="vehicles"), people)


def test_dense_masks_are_concatenated_as_in_v1() -> None:
    # given
    people, vehicles = dense(PEOPLE), dense(VEHICLES, namespace="vehicles")
    expected = DetectionsCombineBlockV1().run(
        prediction_one=as_sv(people, with_masks=True),
        prediction_two=as_sv(vehicles, with_masks=True),
    )["predictions"]

    # when
    combined = combine_predictions(people, vehicles)

    # then
    assert isinstance(combined, InstanceDetections)
    np.testing.assert_array_equal(combined.mask.numpy(), expected.mask)
    assert row_record(combined).rows == 5
    INSTANCE_SEGMENTATION_PREDICTION_KIND.check(combined)


def test_dense_masks_of_different_sizes_are_rejected() -> None:
    # given
    vehicles = dense(VEHICLES, namespace="vehicles")
    vehicles = dataclasses.replace(vehicles, mask=vehicles.mask[:, :, :50])

    # when / then
    with pytest.raises(ValueError, match="mask sizes differ"):
        combine_predictions(dense(PEOPLE), vehicles)


def test_rle_masks_stay_encoded() -> None:
    # given
    people, vehicles = rle(PEOPLE), rle(VEHICLES, namespace="vehicles")

    # when
    combined = combine_predictions(people, vehicles)

    # then
    assert isinstance(combined.mask, InstancesRLEMasks)
    assert combined.mask.masks == people.mask.masks + vehicles.mask.masks
    assert tuple(combined.mask.image_size) == SIZE_HW
    INSTANCE_SEGMENTATION_PREDICTION_KIND.check(combined)


def test_views_on_one_grid_stay_compact() -> None:
    # given
    people, vehicles = view(PEOPLE), view(VEHICLES, namespace="vehicles")

    # when
    combined = combine_predictions(people, vehicles)

    # then
    assert isinstance(combined, SegmentationView)
    assert combined.scores.shape == (5, 20, 30)
    assert torch.equal(combined.scores[2:], vehicles.scores)
    assert people.materialization_count == vehicles.materialization_count == 0
    assert combined.materialization_count == 0
    assert DetectionTable(combined).class_names(require_all=True) == [
        "person",
        "person",
        "car",
        "bus",
        "car",
    ]
    assert torch.equal(combined.full_res().mask, box_masks(combined.detections))


def test_dense_fallback_views_combine_to_dense_masks() -> None:
    # given: v2/instance_segmentation@v1 emits only dense-fallback views today
    people = SegmentationView.from_dense(dense(PEOPLE))
    vehicles = SegmentationView.from_dense(dense(VEHICLES, namespace="vehicles"))
    assert people.is_dense_fallback and vehicles.is_dense_fallback

    # when
    combined = combine_predictions(people, vehicles)

    # then
    assert isinstance(combined, InstanceDetections)
    assert torch.equal(combined.mask, box_masks(combined))
    assert torch.equal(combined.mask[:2], people.full_res().mask)
    assert torch.equal(combined.mask[2:], vehicles.full_res().mask)
    assert people.materialization_count == vehicles.materialization_count == 0
    assert row_record(combined).namespace not in {"people", "vehicles"}
    INSTANCE_SEGMENTATION_PREDICTION_KIND.check(combined)


def test_views_with_different_thresholds_materialize_at_reference_fidelity() -> None:
    # given
    people = view(PEOPLE)
    vehicles = view(VEHICLES, namespace="vehicles", threshold=1.0)

    # when
    combined = combine_predictions(people, vehicles)

    # then
    assert isinstance(combined, InstanceDetections)
    expected = torch.cat([people.full_res().mask, vehicles.full_res().mask])
    assert torch.equal(combined.mask, expected)


def test_equal_grid_shapes_with_different_geometry_keep_each_inputs_masks() -> None:
    # given: same grid shape, but the vehicles model saw a static crop, so its
    # cells lie 4 px further right in the image
    people, vehicles = view(PEOPLE), static_crop_view(VEHICLES)
    assert people.scores.shape[1:] == vehicles.scores.shape[1:]
    assert people.geometry != vehicles.geometry

    # when
    combined = combine_predictions(people, vehicles)

    # then: each row is upscaled with its own input's geometry
    assert isinstance(combined, InstanceDetections)
    assert torch.equal(combined.mask, box_masks(combined))
    assert torch.equal(combined.mask[:2], people.full_res().mask)
    assert torch.equal(combined.mask[2:], vehicles.full_res().mask)
    INSTANCE_SEGMENTATION_PREDICTION_KIND.check(combined)


@pytest.mark.parametrize(
    "first, second",
    [(view, dense), (dense, rle), (rle, view)],
    ids=["view+dense", "dense+rle", "rle+view"],
)
def test_mixed_mask_representations_become_dense(first, second) -> None:
    # given
    people, vehicles = first(PEOPLE), second(VEHICLES, namespace="vehicles")

    # when
    combined = combine_predictions(people, vehicles)

    # then
    assert isinstance(combined, InstanceDetections)
    assert isinstance(combined.mask, torch.Tensor)
    assert torch.equal(combined.mask, box_masks(combined))
    assert row_record(combined).rows == 5
    INSTANCE_SEGMENTATION_PREDICTION_KIND.check(combined)


def test_frame_from_legacy_keys_only_is_accepted() -> None:
    # given
    legacy = {
        key: value
        for key, value in frame_metadata().items()
        if key != "workflows_v2_frame"
    }
    people = dataclasses.replace(
        boxes(PEOPLE, namespace=None),
        image_metadata={**legacy, "class_names": {0: "person"}},
    )

    # when
    combined = combine_predictions(people, boxes(VEHICLES, namespace="vehicles"))

    # then
    assert isinstance(combined, Detections)
    assert len(combined) == 5
