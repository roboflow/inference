"""Unit tests for SAM 3 Interactive block (point / box prompting via SAM3 PVS)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.segmentation_entities import (
    Sam2SegmentationPrediction,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3_interactive import (
    v1 as sam3_interactive_v1,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3_interactive import (
    v1_tensor as sam3_interactive_v1_tensor,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3_interactive.v1 import (
    BlockManifest,
    SegmentAnything3InteractiveBlockV1,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3_interactive.v1_tensor import (
    BlockManifest as TensorBlockManifest,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3_interactive.v1_tensor import (
    SegmentAnything3InteractiveBlockV1 as TensorSegmentAnything3InteractiveBlockV1,
)
from roboflow_workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    PREDICTION_TYPE_KEY,
    RLE_MASK_KEY_IN_SV_DETECTIONS,
)
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

from inference_models.models.base.object_detection import Detections
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask


@pytest.fixture
def mock_model_manager():
    mock = MagicMock()
    mock_prediction = MagicMock()
    mock_prediction.masks = [[[0, 0], [100, 0], [100, 100], [0, 100]]]
    mock_prediction.confidence = 0.95
    mock.run_sam2_segmentation.return_value = [MagicMock(predictions=[mock_prediction])]
    return mock


@pytest.fixture
def mock_workflow_image_data():
    start_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=start_image,
    )


def _example_detections() -> sv.Detections:
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    detections["class_name"] = np.array(["object"])
    detections["detection_id"] = np.array(["det_1"])
    return detections


def test_manifest_parsing_with_literal_points() -> None:
    data = {
        "type": "roboflow_core/sam3_interactive@v1",
        "name": "sam3_interactive",
        "images": "$inputs.image",
        "points": [
            {"x": 320, "y": 240, "positive": True},
            {"x": 100, "y": 100, "positive": False},
        ],
    }
    result = BlockManifest.model_validate(data)
    assert result.type == "roboflow_core/sam3_interactive@v1"
    assert len(result.points) == 2


def test_manifest_parsing_with_points_selector() -> None:
    data = {
        "type": "roboflow_core/sam3_interactive@v1",
        "name": "sam3_interactive",
        "images": "$inputs.image",
        "points": "$inputs.points",
        "boxes": "$steps.detection.predictions",
    }
    result = BlockManifest.model_validate(data)
    assert result.points == "$inputs.points"
    assert result.boxes == "$steps.detection.predictions"


def test_manifest_parsing_with_sequence_points() -> None:
    data = {
        "type": "roboflow_core/sam3_interactive@v1",
        "name": "sam3_interactive",
        "images": "$inputs.image",
        "points": [[320, 240], [100, 100, False]],
    }
    result = BlockManifest.model_validate(data)
    assert len(result.points) == 2


def test_manifest_parsing_rejects_invalid_points() -> None:
    data = {
        "type": "roboflow_core/sam3_interactive@v1",
        "name": "sam3_interactive",
        "images": "$inputs.image",
        "points": [{"x": 320}],
    }
    with pytest.raises(Exception):
        _ = BlockManifest.model_validate(data)


def test_run_raises_when_no_prompts_given(
    mock_model_manager, mock_workflow_image_data
) -> None:
    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    with pytest.raises(ValueError):
        _ = block.run(
            images=[mock_workflow_image_data],
            points=None,
            boxes=None,
            threshold=0.0,
            multimask_output=True,
        )


def test_run_locally_with_point_prompts(
    mock_model_manager, mock_workflow_image_data
) -> None:
    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        points=[
            {"x": 320, "y": 240, "positive": True},
            {"x": 100, "y": 100, "positive": False},
        ],
        boxes=None,
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    assert "predictions" in result[0]
    mock_model_manager.add_model.assert_called_once()
    prompts = mock_model_manager.run_sam2_segmentation.call_args.kwargs["prompts"]
    assert len(prompts) == 1
    assert len(prompts[0]["points"]) == 2
    assert prompts[0]["points"][0]["x"] == 320
    assert prompts[0]["points"][0]["positive"] is True
    assert prompts[0]["points"][1]["positive"] is False
    predictions = result[0]["predictions"]
    assert predictions.class_id.tolist() == [-1]
    assert predictions["class_name"].tolist() == ["foreground"]


def test_numpy_boxes_and_points_keep_class_zero_distinct_from_point_prompt(
    mock_model_manager, mock_workflow_image_data
) -> None:
    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        points=[[320, 240]],
        boxes=[_example_detections()],
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    # box and point prompts cannot be mixed in a single SAM prompt batch,
    # so the block issues one request per prompt group
    assert mock_model_manager.run_sam2_segmentation.call_count == 2
    box_prompts = mock_model_manager.run_sam2_segmentation.call_args_list[0].kwargs[
        "prompts"
    ]
    assert len(box_prompts) == 1
    assert "box" in box_prompts[0]
    assert box_prompts[0]["box"]["x"] == 30  # box centre of [10, 10, 50, 50]
    assert "points" not in box_prompts[0]
    point_prompts = mock_model_manager.run_sam2_segmentation.call_args_list[1].kwargs[
        "prompts"
    ]
    assert len(point_prompts) == 1
    assert "box" not in point_prompts[0]
    assert point_prompts[0]["points"][0]["x"] == 320
    # masks from both requests are merged into a single output
    predictions = result[0]["predictions"]
    assert predictions.class_id.tolist() == [0, -1]
    assert list(predictions["class_name"]) == ["object", "foreground"]


def test_tensor_boxes_and_points_keep_class_zero_distinct_from_point_prompt(
    mock_workflow_image_data,
) -> None:
    model_manager = MagicMock()
    model_prediction = MagicMock()
    model_prediction.masks = torch.ones((1, 480, 640), dtype=torch.float32)
    model_prediction.scores = torch.tensor([0.95], dtype=torch.float32)
    model_manager.run_tensor_native_inference.side_effect = [
        [model_prediction],
        [model_prediction],
    ]
    block = TensorSegmentAnything3InteractiveBlockV1(
        model_manager=model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    boxes = Detections(
        xyxy=torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
        class_id=torch.tensor([0], dtype=torch.int64),
        confidence=torch.tensor([0.9], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "object"}},
        bboxes_metadata=[{"detection_id": "det_1"}],
    )

    result = block.run_locally(
        images=[mock_workflow_image_data],
        points=[[320, 240]],
        boxes=[boxes],
        threshold=0.0,
        multimask_output=False,
    )

    predictions = result[0]["predictions"]
    assert predictions.class_id.tolist() == [0, -1]
    assert [metadata["class"] for metadata in predictions.bboxes_metadata] == [
        "object",
        "foreground",
    ]
    assert predictions.image_metadata[CLASS_NAMES_KEY] == {
        0: "object",
        -1: "foreground",
    }


def test_run_locally_with_empty_detections_and_no_points(
    mock_model_manager, mock_workflow_image_data
) -> None:
    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        points=None,
        boxes=[sv.Detections.empty()],
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    assert len(result[0]["predictions"]) == 0
    mock_model_manager.run_sam2_segmentation.assert_not_called()


@patch(
    "roboflow_workflows.core_steps.models.foundation.segment_anything3_interactive.v1.InferenceHTTPClient"
)
def test_run_remotely_calls_sam3_visual_segment(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
) -> None:
    mock_client = MagicMock()
    mock_client.sam3_visual_segment.return_value = {
        "predictions": [
            {
                "confidence": 0.95,
                "masks": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
            }
        ],
        "time": 0.1,
    }
    mock_client_cls.return_value = mock_client

    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        points=[{"x": 320, "y": 240, "positive": True}],
        boxes=None,
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    assert "predictions" in result[0]
    mock_client.sam3_visual_segment.assert_called_once()
    call_kwargs = mock_client.sam3_visual_segment.call_args.kwargs
    assert call_kwargs["prompts"] == [
        {"points": [{"x": 320.0, "y": 240.0, "positive": True}]}
    ]


@patch(
    "roboflow_workflows.core_steps.models.foundation.segment_anything3_interactive.v1.InferenceHTTPClient"
)
def test_run_remotely_with_boxes(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
) -> None:
    mock_client = MagicMock()
    mock_client.sam3_visual_segment.return_value = {
        "predictions": [
            {
                "confidence": 0.95,
                "masks": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
            }
        ],
        "time": 0.1,
    }
    mock_client_cls.return_value = mock_client

    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        points=None,
        boxes=[_example_detections()],
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    call_kwargs = mock_client.sam3_visual_segment.call_args.kwargs
    assert call_kwargs["prompts"] == [
        {"box": {"x": 30.0, "y": 30.0, "width": 40.0, "height": 40.0}}
    ]
    # class name of the box prompt should be forwarded to predicted masks
    predictions = result[0]["predictions"]
    assert list(predictions["class_name"]) == ["object"]


def _two_region_mask() -> np.ndarray:
    # one object whose mask has two separate regions
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:150, 100:200] = True
    mask[300:350, 400:500] = True
    return mask


# outlines of the two regions of `_two_region_mask()`, as a polygon response
TWO_REGION_POLYGONS = [
    [[100, 100], [199, 100], [199, 149], [100, 149]],
    [[400, 300], [499, 300], [499, 349], [400, 349]],
]


def _coco_rle(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _model_manager_returning(*predictions: Sam2SegmentationPrediction) -> MagicMock:
    model_manager = MagicMock()
    model_manager.run_sam2_segmentation.return_value = [
        MagicMock(predictions=list(predictions))
    ]
    return model_manager


def _local_block(model_manager: MagicMock) -> SegmentAnything3InteractiveBlockV1:
    return SegmentAnything3InteractiveBlockV1(
        model_manager=model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )


@pytest.mark.parametrize("manifest_class", [BlockManifest, TensorBlockManifest])
def test_manifest_output_format_defaults_to_polygons(manifest_class) -> None:
    result = manifest_class.model_validate(
        {
            "type": "roboflow_core/sam3_interactive@v1",
            "name": "sam3_interactive",
            "images": "$inputs.image",
            "points": [[320, 240]],
        }
    )

    assert result.output_format == "polygons"
    assert [kind.name for kind in result.get_actual_outputs()[0].kind] == [
        "instance_segmentation_prediction"
    ]


@pytest.mark.parametrize("manifest_class", [BlockManifest, TensorBlockManifest])
def test_manifest_with_rle_output_declares_the_rle_kind_first(manifest_class) -> None:
    result = manifest_class.model_validate(
        {
            "type": "roboflow_core/sam3_interactive@v1",
            "name": "sam3_interactive",
            "images": "$inputs.image",
            "points": [[320, 240]],
            "output_format": "rle",
        }
    )

    assert [kind.name for kind in result.get_actual_outputs()[0].kind] == [
        "rle_instance_segmentation_prediction",
        "instance_segmentation_prediction",
    ]
    assert {kind.name for kind in manifest_class.describe_outputs()[0].kind} == {
        "instance_segmentation_prediction",
        "rle_instance_segmentation_prediction",
    }


def test_manifest_rejects_unknown_output_format() -> None:
    with pytest.raises(Exception):
        _ = BlockManifest.model_validate(
            {
                "type": "roboflow_core/sam3_interactive@v1",
                "name": "sam3_interactive",
                "images": "$inputs.image",
                "points": [[320, 240]],
                "output_format": "mask",
            }
        )


def test_run_locally_with_polygons_does_not_request_a_mask_format(
    mock_model_manager, mock_workflow_image_data
) -> None:
    block = _local_block(model_manager=mock_model_manager)

    result = block.run(
        images=[mock_workflow_image_data],
        points=[[320, 240]],
        boxes=None,
        threshold=0.0,
        multimask_output=True,
    )

    assert "format" not in mock_model_manager.run_sam2_segmentation.call_args.kwargs
    predictions = result[0]["predictions"]
    assert RLE_MASK_KEY_IN_SV_DETECTIONS not in predictions.data
    assert predictions[PREDICTION_TYPE_KEY].tolist() == ["instance-segmentation"]


def test_run_locally_with_rle_output_keeps_the_model_mask(
    mock_workflow_image_data,
) -> None:
    mask = _two_region_mask()
    rle = _coco_rle(mask)
    model_manager = _model_manager_returning(
        Sam2SegmentationPrediction(masks=rle, confidence=0.95, format="rle")
    )
    block = _local_block(model_manager=model_manager)

    result = block.run(
        images=[mock_workflow_image_data],
        points=None,
        boxes=[_example_detections()],
        threshold=0.0,
        multimask_output=False,
        output_format="rle",
    )

    assert model_manager.run_sam2_segmentation.call_args.kwargs["format"] == "rle"
    predictions = result[0]["predictions"]
    # one detection for the box prompt, although its mask has two regions
    assert len(predictions) == 1
    assert predictions[RLE_MASK_KEY_IN_SV_DETECTIONS][0] == rle
    assert np.array_equal(predictions.mask[0], mask)
    assert predictions.xyxy.tolist() == [[100.0, 100.0, 500.0, 350.0]]
    assert predictions.confidence.tolist() == pytest.approx([0.95])
    assert predictions.class_id.tolist() == [0]
    assert predictions["class_name"].tolist() == ["object"]
    assert predictions[PREDICTION_TYPE_KEY].tolist() == ["rle-instance-segmentation"]


def test_run_locally_with_rle_output_and_point_prompt_uses_the_synthetic_class(
    mock_workflow_image_data,
) -> None:
    model_manager = _model_manager_returning(
        Sam2SegmentationPrediction(
            masks=_coco_rle(_two_region_mask()), confidence=0.95, format="rle"
        )
    )
    block = _local_block(model_manager=model_manager)

    result = block.run(
        images=[mock_workflow_image_data],
        points=[[150, 125]],
        boxes=None,
        threshold=0.0,
        multimask_output=True,
        output_format="rle",
    )

    predictions = result[0]["predictions"]
    assert predictions.class_id.tolist() == [-1]
    assert predictions["class_name"].tolist() == ["foreground"]


def test_rle_output_drops_masks_below_threshold_and_empty_masks(
    mock_workflow_image_data,
) -> None:
    mask = _two_region_mask()
    model_manager = _model_manager_returning(
        Sam2SegmentationPrediction(masks=_coco_rle(mask), confidence=0.9, format="rle"),
        Sam2SegmentationPrediction(masks=_coco_rle(mask), confidence=0.2, format="rle"),
        Sam2SegmentationPrediction(
            masks=_coco_rle(np.zeros_like(mask)), confidence=0.9, format="rle"
        ),
    )
    boxes = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50], [60, 60, 90, 90], [100, 100, 150, 150]]),
        confidence=np.array([0.9, 0.9, 0.9]),
        class_id=np.array([0, 1, 2]),
    )
    boxes["class_name"] = np.array(["kept", "low_confidence", "empty"])
    boxes["detection_id"] = np.array(["a", "b", "c"])
    block = _local_block(model_manager=model_manager)

    result = block.run(
        images=[mock_workflow_image_data],
        points=None,
        boxes=[boxes],
        threshold=0.5,
        multimask_output=False,
        output_format="rle",
    )

    assert result[0]["predictions"]["class_name"].tolist() == ["kept"]


def test_rle_output_with_no_prompts_for_an_image_is_empty(
    mock_model_manager, mock_workflow_image_data
) -> None:
    block = _local_block(model_manager=mock_model_manager)

    result = block.run(
        images=[mock_workflow_image_data],
        points=None,
        boxes=[sv.Detections.empty()],
        threshold=0.0,
        multimask_output=True,
        output_format="rle",
    )

    assert len(result[0]["predictions"]) == 0
    mock_model_manager.run_sam2_segmentation.assert_not_called()


def test_rle_output_rasterises_polygons_from_a_server_that_ignores_the_format(
    mock_workflow_image_data,
) -> None:
    model_manager = _model_manager_returning(
        Sam2SegmentationPrediction(masks=TWO_REGION_POLYGONS, confidence=0.95)
    )
    block = _local_block(model_manager=model_manager)

    result = block.run(
        images=[mock_workflow_image_data],
        points=None,
        boxes=[_example_detections()],
        threshold=0.0,
        multimask_output=False,
        output_format="rle",
    )

    predictions = result[0]["predictions"]
    assert len(predictions) == 1
    rle = predictions[RLE_MASK_KEY_IN_SV_DETECTIONS][0]
    assert np.array_equal(mask_utils.decode(rle).astype(bool), _two_region_mask())


@pytest.mark.parametrize(
    "output_format, mask_input_format", [("polygons", "json"), ("rle", "rle")]
)
@patch(
    "roboflow_workflows.core_steps.models.foundation.segment_anything3_interactive.v1.InferenceHTTPClient"
)
def test_run_remotely_requests_the_mask_format(
    mock_client_cls,
    output_format,
    mask_input_format,
    mock_model_manager,
    mock_workflow_image_data,
) -> None:
    mock_client = MagicMock()
    mock_client.sam3_visual_segment.return_value = {"predictions": [], "time": 0.1}
    mock_client_cls.return_value = mock_client
    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    _ = block.run(
        images=[mock_workflow_image_data],
        points=[[320, 240]],
        boxes=None,
        threshold=0.0,
        multimask_output=True,
        output_format=output_format,
    )

    call_kwargs = mock_client.sam3_visual_segment.call_args.kwargs
    assert call_kwargs["mask_input_format"] == mask_input_format


@patch(
    "roboflow_workflows.core_steps.models.foundation.segment_anything3_interactive.v1.InferenceHTTPClient"
)
def test_run_remotely_with_rle_output_keeps_the_server_rle(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
) -> None:
    rle = _coco_rle(_two_region_mask())
    mock_client = MagicMock()
    mock_client.sam3_visual_segment.return_value = {
        "predictions": [{"confidence": 0.95, "masks": rle, "format": "rle"}],
        "time": 0.1,
    }
    mock_client_cls.return_value = mock_client
    block = SegmentAnything3InteractiveBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        points=None,
        boxes=[_example_detections()],
        threshold=0.0,
        multimask_output=False,
        output_format="rle",
    )

    predictions = result[0]["predictions"]
    assert predictions[RLE_MASK_KEY_IN_SV_DETECTIONS].tolist() == [rle]
    assert predictions["class_name"].tolist() == ["object"]


def _proxy_response(masks) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"predictions": [{"confidence": 0.95, "masks": masks}]}
    return response


def test_run_via_request_with_polygons_sends_no_format(
    mock_model_manager, mock_workflow_image_data
) -> None:
    block = _local_block(model_manager=mock_model_manager)

    with patch.object(sam3_interactive_v1, "requests") as requests_mock:
        requests_mock.post.return_value = _proxy_response(masks=TWO_REGION_POLYGONS)
        result = block.run_via_request(
            images=[mock_workflow_image_data],
            points=[[150, 125]],
            boxes=None,
            threshold=0.0,
            multimask_output=True,
        )

    assert "format" not in requests_mock.post.call_args.kwargs["json"]
    # one prediction per polygon
    assert len(result[0]["predictions"]) == 2


def test_run_via_request_with_rle_output_asks_the_proxy_for_rle(
    mock_model_manager, mock_workflow_image_data
) -> None:
    rle = _coco_rle(_two_region_mask())
    block = _local_block(model_manager=mock_model_manager)

    with patch.object(sam3_interactive_v1, "requests") as requests_mock:
        requests_mock.post.return_value = _proxy_response(masks=rle)
        result = block.run_via_request(
            images=[mock_workflow_image_data],
            points=[[150, 125]],
            boxes=None,
            threshold=0.0,
            multimask_output=True,
            output_format="rle",
        )

    assert requests_mock.post.call_args.kwargs["json"]["format"] == "rle"
    assert result[0]["predictions"][RLE_MASK_KEY_IN_SV_DETECTIONS].tolist() == [rle]


def _workflow_with_sam3_interactive(output_format: str) -> dict:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/sam3_interactive@v1",
                "name": "sam3",
                "images": "$inputs.image",
                "points": [{"x": 150, "y": 125, "positive": True}],
                "output_format": output_format,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.sam3.predictions",
            }
        ],
    }


def _engine_model_manager(mask: np.ndarray) -> MagicMock:
    """Answers both representations: the numpy block (RLE only when it asks for
    RLE) and the tensor-native twin."""
    model_manager = MagicMock()

    def run_sam2_segmentation(**kwargs):
        masks = (
            _coco_rle(mask) if kwargs.get("format") == "rle" else TWO_REGION_POLYGONS
        )
        prediction = Sam2SegmentationPrediction(masks=masks, confidence=0.95)
        return [MagicMock(predictions=[prediction])]

    model_manager.run_sam2_segmentation.side_effect = run_sam2_segmentation
    tensor_prediction = MagicMock()
    tensor_prediction.masks = torch.from_numpy(
        np.where(mask, 1.0, -1.0)[None].astype(np.float32)
    )
    tensor_prediction.scores = torch.tensor([0.95])
    model_manager.run_tensor_native_inference.return_value = [tensor_prediction]
    return model_manager


def _run_workflow(output_format: str, mask: np.ndarray) -> dict:
    execution_engine = ExecutionEngine.init(
        workflow_definition=_workflow_with_sam3_interactive(
            output_format=output_format
        ),
        init_parameters={
            "workflows_core.model_manager": _engine_model_manager(mask=mask),
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=1,
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = execution_engine.run(
        runtime_parameters={"image": [image]}, serialize_results=True
    )

    return result[0]["predictions"]


def test_workflow_output_carries_the_rle_mask_when_rle_is_selected() -> None:
    mask = _two_region_mask()

    serialized = _run_workflow(output_format="rle", mask=mask)

    (prediction,) = serialized["predictions"]
    assert "points" not in prediction
    assert np.array_equal(mask_utils.decode(prediction["rle_mask"]).astype(bool), mask)


def test_workflow_output_carries_polygons_by_default() -> None:
    serialized = _run_workflow(output_format="polygons", mask=_two_region_mask())

    assert len(serialized["predictions"]) > 0
    for prediction in serialized["predictions"]:
        assert "rle_mask" not in prediction
        assert len(prediction["points"]) >= 3


def test_workflow_rle_output_drops_empty_masks() -> None:
    serialized = _run_workflow(
        output_format="rle", mask=np.zeros((480, 640), dtype=bool)
    )

    assert serialized["predictions"] == []


def _tensor_box_prompt() -> Detections:
    return Detections(
        xyxy=torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
        class_id=torch.tensor([0], dtype=torch.int64),
        confidence=torch.tensor([0.9], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "object"}},
        bboxes_metadata=[{"detection_id": "det_1"}],
    )


def _run_tensor_via_polygon_proxy(
    workflow_image: WorkflowImageData, output_format: str
):
    block = TensorSegmentAnything3InteractiveBlockV1(
        model_manager=MagicMock(),
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )
    response = MagicMock()
    response.json.return_value = {
        "predictions": [{"confidence": 0.95, "masks": TWO_REGION_POLYGONS}]
    }
    with patch.object(sam3_interactive_v1_tensor, "requests") as requests_mock:
        requests_mock.post.return_value = response
        result = block.run_via_request(
            images=[workflow_image],
            points=None,
            boxes=[_tensor_box_prompt()],
            threshold=0.0,
            multimask_output=False,
            output_format=output_format,
        )
    return result[0]["predictions"]


def test_tensor_rle_output_rasterises_a_polygon_answer_into_one_mask_per_prompt(
    mock_workflow_image_data,
) -> None:
    predictions = _run_tensor_via_polygon_proxy(
        workflow_image=mock_workflow_image_data, output_format="rle"
    )

    assert predictions.xyxy.shape[0] == 1
    assert np.array_equal(
        coco_rle_masks_to_numpy_mask(predictions.mask)[0], _two_region_mask()
    )
    assert [metadata["detection_id"] for metadata in predictions.bboxes_metadata] == [
        "det_1"
    ]


def test_tensor_polygon_output_keeps_one_instance_per_polygon(
    mock_workflow_image_data,
) -> None:
    predictions = _run_tensor_via_polygon_proxy(
        workflow_image=mock_workflow_image_data, output_format="polygons"
    )

    assert predictions.xyxy.shape[0] == 2


@pytest.mark.parametrize(
    "output_format, expected_instances", [("rle", 0), ("polygons", 1)]
)
def test_tensor_empty_masks_are_dropped_for_rle_output(
    mock_workflow_image_data, output_format, expected_instances
) -> None:
    model_manager = MagicMock()
    model_prediction = MagicMock()
    model_prediction.masks = torch.full((1, 480, 640), -1.0)
    model_prediction.scores = torch.tensor([0.95])
    model_manager.run_tensor_native_inference.return_value = [model_prediction]
    block = TensorSegmentAnything3InteractiveBlockV1(
        model_manager=model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        points=[[150, 125]],
        boxes=None,
        threshold=0.0,
        multimask_output=True,
        output_format=output_format,
    )

    # the polygon output keeps the empty instance and its serializer drops it
    assert result[0]["predictions"].xyxy.shape[0] == expected_instances
