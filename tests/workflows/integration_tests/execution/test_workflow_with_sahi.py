import base64

import cv2
import numpy as np
import supervision as sv

from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.env import USE_INFERENCE_MODELS, WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

SAHI_WORKFLOW = {
    "version": "1.0.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "overlap_filtering_strategy"},
    ],
    "steps": [
        {
            "type": "roboflow_core/image_slicer@v1",
            "name": "image_slicer",
            "image": "$inputs.image",
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detection",
            "image": "$steps.image_slicer.slices",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/detections_stitch@v1",
            "name": "stitch",
            "reference_image": "$inputs.image",
            "predictions": "$steps.detection.predictions",
            "overlap_filtering_strategy": "$inputs.overlap_filtering_strategy",
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bbox_visualiser",
            "predictions": "$steps.stitch.predictions",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.stitch.predictions",
            "coordinates_system": "own",
        },
        {
            "type": "JsonField",
            "name": "visualisation",
            "selector": "$steps.bbox_visualiser.image",
        },
    ],
}


def test_sahi_workflow_with_none_as_filtering_strategy(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test we check how all blocks that form SAHI technique behave.
    Blocks involved in tests:
    - "roboflow_core/image_slicer@v1" from inference.core.workflows.core_steps.transformations.image_slicer.v1
    - "roboflow_core/detections_stitch@v1", from inference.core.workflows.core_steps.fusion.detections_stitch.v1

    This scenario covers usage of SAHI when overlapping predictions are not post-processed.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SAHI_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, crowd_image],
            "overlap_filtering_strategy": "none",
        }
    )

    # then
    if not USE_INFERENCE_MODELS:
        assert np.allclose(
            result[0]["predictions"].xyxy,
            np.array(
                [
                    [113, 479, 343, 639],
                    [425, 493, 583, 615],
                    [381, 520, 407, 538],
                    [512, 493, 583, 613],
                    [775, 388, 1151, 640],
                    [775, 388, 1151, 640],
                    [1025, 390, 1664, 640],
                    [1536, 506, 1717, 640],
                    [113, 512, 345, 660],
                    [424, 512, 583, 614],
                    [381, 520, 407, 538],
                    [512, 512, 582, 613],
                    [768, 512, 1152, 976],
                    [765, 513, 1152, 980],
                    [1024, 512, 1661, 954],
                    [1537, 512, 1749, 947],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for first image to be exactly as measured during test creation"
        assert np.allclose(
            result[1]["predictions"].xyxy,
            np.array(
                [
                    [180, 273, 244, 383],
                    [271, 266, 328, 383],
                    [552, 259, 598, 365],
                    [113, 269, 145, 347],
                    [416, 258, 457, 365],
                    [521, 257, 555, 360],
                    [387, 264, 414, 342],
                    [158, 267, 183, 349],
                    [324, 256, 345, 320],
                    [341, 261, 362, 338],
                    [247, 251, 262, 284],
                    [239, 251, 249, 282],
                    [552, 260, 598, 366],
                    [523, 257, 557, 362],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for second image to be exactly as measured during test creation"
    else:
        assert np.allclose(
            result[0]["predictions"].xyxy,
            np.array(
                [
                    [113, 479, 343, 639],
                    [425, 493, 583, 615],
                    [381, 520, 407, 538],
                    [326, 519, 356, 537],
                    [512, 493, 583, 613],
                    [775, 388, 1151, 640],
                    [775, 388, 1151, 640],
                    [1025, 390, 1665, 640],
                    [1536, 506, 1717, 640],
                    [113, 512, 345, 660],
                    [424, 512, 583, 614],
                    [381, 520, 407, 538],
                    [111, 519, 139, 537],
                    [325, 519, 356, 536],
                    [512, 512, 582, 613],
                    [768, 509, 1152, 976],
                    [765, 513, 1152, 980],
                    [1023, 511, 1661, 954],
                    [1537, 512, 1749, 947],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for first image to be exactly as measured during test creation"
        assert np.allclose(
            result[1]["predictions"].xyxy,
            np.array(
                [
                    [180, 273, 244, 384],
                    [271, 267, 328, 384],
                    [552, 260, 598, 365],
                    [113, 270, 145, 348],
                    [416, 259, 457, 365],
                    [521, 257, 555, 360],
                    [387, 264, 414, 342],
                    [158, 268, 183, 350],
                    [324, 257, 345, 321],
                    [341, 262, 362, 338],
                    [247, 251, 262, 285],
                    [240, 251, 250, 282],
                    [552, 260, 598, 366],
                    [523, 257, 557, 362],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for second image to be exactly as measured during test creation"


SAHI_WORKFLOW_SLICER_V2 = {
    "version": "1.0.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "overlap_filtering_strategy"},
        {"type": "WorkflowParameter", "name": "slice_width", "default_value": 128},
        {"type": "WorkflowParameter", "name": "slice_height", "default_value": 128},
        {"type": "WorkflowParameter", "name": "slice_overlap", "default_value": 0.1},
    ],
    "steps": [
        {
            "type": "roboflow_core/image_slicer@v2",
            "name": "image_slicer",
            "image": "$inputs.image",
            "slice_width": "$inputs.slice_width",
            "slice_height": "$inputs.slice_height",
            "slice_overlap": "$inputs.slice_overlap",
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detection",
            "image": "$steps.image_slicer.slices",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/detections_stitch@v1",
            "name": "stitch",
            "reference_image": "$inputs.image",
            "predictions": "$steps.detection.predictions",
            "overlap_filtering_strategy": "$inputs.overlap_filtering_strategy",
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bbox_visualiser",
            "predictions": "$steps.stitch.predictions",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.stitch.predictions",
            "coordinates_system": "own",
        },
        {
            "type": "JsonField",
            "name": "slices",
            "selector": "$steps.image_slicer.slices",
        },
        {
            "type": "JsonField",
            "name": "visualisation",
            "selector": "$steps.bbox_visualiser.image",
        },
    ],
}


@add_to_workflows_gallery(
    category="Advanced inference techniques",
    use_case_title="SAHI in workflows - object detection",
    use_case_description="""
This example illustrates usage of [SAHI](https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/) 
technique in workflows.

Workflows implementation requires three blocks:

- Image Slicer - which runs a sliding window over image and for each image prepares batch of crops 

- detection model block (in our scenario Roboflow Object Detection model) - which is responsible 
for making predictions on each crop

- Detections stitch - which combines partial predictions for each slice of the image into a single prediction
    """,
    workflow_definition=SAHI_WORKFLOW,
    workflow_name_in_app="sahi-detection",
)
def test_sahi_workflow_with_slicer_v2(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test we check how all blocks that form SAHI technique behave.
    Blocks involved in tests:
    - "roboflow_core/image_slicer@v2" from inference.core.workflows.core_steps.transformations.image_slicer.v2
    - "roboflow_core/detections_stitch@v1", from inference.core.workflows.core_steps.fusion.detections_stitch.v1

    This scenario covers usage of SAHI when overlapping predictions are not post-processed.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SAHI_WORKFLOW_SLICER_V2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "overlap_filtering_strategy": "nms",
        }
    )

    # then
    assert np.allclose(
        result[0]["predictions"].xyxy,
        np.array(
            [
                [103, 103, 113, 124],
                [182, 272, 231, 334],
                [114, 270, 144, 334],
                [271, 267, 329, 334],
                [226, 288, 246, 329],
                [240, 251, 251, 283],
                [249, 251, 261, 284],
                [388, 264, 413, 334],
                [309, 265, 318, 297],
                [359, 260, 374, 291],
                [323, 257, 345, 318],
                [342, 260, 361, 321],
                [415, 259, 457, 334],
                [552, 260, 597, 334],
                [522, 257, 557, 334],
                [158, 297, 181, 348],
            ]
        ),
        atol=2,
    ), "Expected boxes for first image to be exactly as measured during test creation"


def test_sahi_workflow_with_nms_as_filtering_strategy(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test we check how all blocks that form SAHI technique behave.
    Blocks involved in tests:
    - "roboflow_core/image_slicer@v1" from inference.core.workflows.core_steps.transformations.image_slicer.v1
    - "roboflow_core/detections_stitch@v1", from inference.core.workflows.core_steps.fusion.detections_stitch.v1

    This scenario covers usage of SAHI when overlapping predictions are post-processed with NMS
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SAHI_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, crowd_image],
            "overlap_filtering_strategy": "nms",
        }
    )

    # then
    if not USE_INFERENCE_MODELS:
        assert np.allclose(
            result[0]["predictions"].xyxy,
            np.array(
                [
                    [113, 479, 343, 639],
                    [381, 520, 407, 538],
                    [775, 388, 1151, 640],
                    [775, 388, 1151, 640],
                    [1025, 390, 1664, 640],
                    [1536, 506, 1717, 640],
                    [512, 512, 582, 613],
                    [768, 512, 1152, 976],
                    [765, 513, 1152, 980],
                    [1024, 512, 1661, 954],
                    [1537, 512, 1749, 947],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for first image to be exactly as measured during test creation"
        assert np.allclose(
            result[1]["predictions"].xyxy,
            np.array(
                [
                    [180, 273, 244, 383],
                    [271, 266, 328, 383],
                    [113, 269, 145, 347],
                    [416, 258, 457, 365],
                    [387, 264, 414, 342],
                    [158, 267, 183, 349],
                    [324, 256, 345, 320],
                    [341, 261, 362, 338],
                    [247, 251, 262, 284],
                    [239, 251, 249, 282],
                    [552, 260, 598, 366],
                    [523, 257, 557, 362],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for second image to be exactly as measured during test creation"
    else:
        assert np.allclose(
            result[0]["predictions"].xyxy,
            np.array(
                [
                    [113, 479, 343, 639],
                    [381, 520, 407, 538],
                    [326, 519, 356, 537],
                    [775, 388, 1151, 640],
                    [775, 388, 1151, 640],
                    [1025, 390, 1665, 640],
                    [1536, 506, 1717, 640],
                    [111, 519, 139, 537],
                    [512, 512, 582, 613],
                    [768, 509, 1152, 976],
                    [765, 513, 1152, 980],
                    [1023, 511, 1661, 954],
                    [1537, 512, 1749, 947],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for first image to be exactly as measured during test creation"
        assert np.allclose(
            result[1]["predictions"].xyxy,
            np.array(
                [
                    [180, 273, 244, 384],
                    [271, 267, 328, 384],
                    [113, 270, 145, 348],
                    [416, 259, 457, 365],
                    [387, 264, 414, 342],
                    [158, 268, 183, 350],
                    [324, 257, 345, 321],
                    [341, 262, 362, 338],
                    [247, 251, 262, 285],
                    [240, 251, 250, 282],
                    [552, 260, 598, 366],
                    [523, 257, 557, 362],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for second image to be exactly as measured during test creation"


def test_sahi_workflow_with_nmm_as_filtering_strategy(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test we check how all blocks that form SAHI technique behave.
    Blocks involved in tests:
    - "roboflow_core/image_slicer@v1" from inference.core.workflows.core_steps.transformations.image_slicer.v1
    - "roboflow_core/detections_stitch@v1", from inference.core.workflows.core_steps.fusion.detections_stitch.v1

    This scenario covers usage of SAHI when overlapping predictions are post-processed with NMM
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SAHI_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, crowd_image],
            "overlap_filtering_strategy": "nmm",
        }
    )

    # then
    if not USE_INFERENCE_MODELS:
        assert np.allclose(
            result[0]["predictions"].xyxy,
            np.array(
                [
                    [113, 479, 345, 660],
                    [1025, 390, 1664, 640],
                    [424, 493, 583, 615],
                    [381, 520, 407, 538],
                    [1537, 512, 1749, 947],
                    [1024, 512, 1661, 954],
                    [775, 388, 1151, 640],
                    [768, 512, 1152, 976],
                    [1536, 506, 1717, 640],
                    [775, 388, 1151, 640],
                    [765, 513, 1152, 980],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for first image to be exactly as measured during test creation"
        assert np.allclose(
            result[1]["predictions"].xyxy,
            np.array(
                [
                    [552, 259, 598, 366],
                    [180, 273, 244, 383],
                    [271, 266, 328, 383],
                    [113, 269, 145, 347],
                    [521, 257, 557, 362],
                    [416, 258, 457, 365],
                    [387, 264, 414, 342],
                    [158, 267, 183, 349],
                    [324, 256, 345, 320],
                    [341, 261, 362, 338],
                    [247, 251, 262, 284],
                    [239, 251, 249, 282],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for second image to be exactly as measured during test creation"
    else:
        assert np.allclose(
            result[0]["predictions"].xyxy,
            np.array(
                [
                    [113, 479, 345, 660],
                    [1025, 390, 1665, 640],
                    [424, 493, 583, 615],
                    [381, 520, 407, 538],
                    [1537, 512, 1749, 947],
                    [1023, 511, 1661, 954],
                    [775, 388, 1151, 640],
                    [325, 519, 356, 537],
                    [111, 519, 139, 537],
                    [768, 509, 1152, 976],
                    [1536, 506, 1717, 640],
                    [775, 388, 1151, 640],
                    [765, 513, 1152, 980],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for first image to be exactly as measured during test creation"
        assert np.allclose(
            result[1]["predictions"].xyxy,
            np.array(
                [
                    [552, 260, 598, 366],
                    [180, 273, 244, 384],
                    [271, 267, 328, 384],
                    [113, 270, 145, 348],
                    [521, 257, 557, 362],
                    [416, 259, 457, 365],
                    [387, 264, 414, 342],
                    [158, 268, 183, 350],
                    [324, 257, 345, 321],
                    [341, 262, 362, 338],
                    [247, 251, 262, 285],
                    [240, 251, 250, 282],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for second image to be exactly as measured during test creation"


def test_sahi_workflow_with_serialization(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SAHI_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, crowd_image],
            "overlap_filtering_strategy": "nms",
        },
        serialize_results=True,
    )

    # then
    result_1 = sv.Detections.from_inference(result[0]["predictions"])
    result_2 = sv.Detections.from_inference(result[1]["predictions"])
    if not USE_INFERENCE_MODELS:
        assert (
            len(result_1) == 11
        ), "Expected to deserialize 1st image detections properly"
        assert (
            len(result_2) == 12
        ), "Expected to deserialize 2nd image detections properly"
    else:
        assert (
            len(result_1) == 13
        ), "Expected to deserialize 1st image detections properly"
        assert (
            len(result_2) == 12
        ), "Expected to deserialize 2nd image detections properly"
    decoded_image_bytes = base64.b64decode(result[0]["visualisation"]["value"])
    decoded_image = cv2.imdecode(
        np.frombuffer(decoded_image_bytes, np.uint8), cv2.IMREAD_COLOR
    )
    assert (
        decoded_image.shape == license_plate_image.shape
    ), "Expected to deserialize result image properly"


def test_sahi_workflow_provides_the_same_result_as_sahi_applied_directly(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test we check how all blocks that form SAHI technique behave.
    Blocks involved in tests:
    - "roboflow_core/image_slicer@v1" from inference.core.workflows.core_steps.transformations.image_slicer.v1
    - "roboflow_core/detections_stitch@v1", from inference.core.workflows.core_steps.fusion.detections_stitch.v1

    This scenario covers checking if sv.InferenceSlicer gives the same results as
    SAHI technique in workflows.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SAHI_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    model_manager.add_model(
        model_id="yolov8n-640",
        api_key=None,
    )
    model = model_manager.models()["yolov8n-640"]

    def slicer_callback(image_slice: np.ndarray):
        inference_image = {"type": "numpy_object", "value": image_slice}
        request = ObjectDetectionInferenceRequest(
            api_key=None,
            model_id="yolov8n-640",
            image=[inference_image],
        )

        predictions = model.infer_from_request(request)[0]
        detections = sv.Detections.from_inference(predictions)
        return detections

    try:
        slicer = sv.InferenceSlicer(
            callback=slicer_callback,
            slice_wh=(640, 640),
            overlap_wh=(0.2, 0.2),
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            iou_threshold=0.3,
        )
    except ValueError:
        slicer = sv.InferenceSlicer(
            callback=slicer_callback,
            slice_wh=(640, 640),
            overlap_ratio_wh=(0.2, 0.2),
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            iou_threshold=0.3,
        )

    # when
    detections_obtained_directly = slicer(crowd_image)
    workflow_result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image],
            "overlap_filtering_strategy": "nms",
        }
    )

    detections_obtained_directly_xyxy = detections_obtained_directly.xyxy.copy()
    detections_obtained_directly_xyxy.sort(axis=0)
    workflow_result_xyxy = workflow_result[0]["predictions"].xyxy.copy()
    workflow_result_xyxy.sort(axis=0)
    # then
    assert np.allclose(
        detections_obtained_directly_xyxy,
        workflow_result_xyxy,
        atol=2,
    ), "Expected bounding boxes to be the same for workflow SAHI and direct SAHI"
    detections_obtained_directly_confidence = (
        detections_obtained_directly.confidence.copy()
    )
    detections_obtained_directly_confidence.sort()
    workflow_result_confidence = workflow_result[0]["predictions"].confidence.copy()
    workflow_result_confidence.sort()
    assert np.allclose(
        detections_obtained_directly_confidence,
        workflow_result_confidence,
        atol=1e-1,
    ), "Expected confidences to be the same for workflow SAHI and direct SAHI"
    detections_obtained_directly_class_id = detections_obtained_directly.class_id.copy()
    detections_obtained_directly_class_id.sort(axis=0)
    workflow_result_class_id = workflow_result[0]["predictions"].class_id.copy()
    workflow_result_class_id.sort(axis=0)
    assert np.all(
        detections_obtained_directly_class_id == workflow_result_class_id
    ), "Expected class ids to be the same for workflow SAHI and direct SAHI"


SAHI_WORKFLOW_FOR_SEGMENTATION = {
    "version": "1.0.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "overlap_filtering_strategy"},
    ],
    "steps": [
        {
            "type": "roboflow_core/image_slicer@v1",
            "name": "image_slicer",
            "image": "$inputs.image",
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "detection",
            "image": "$steps.image_slicer.slices",
            "model_id": "yolov8n-seg-640",
        },
        {
            "type": "roboflow_core/detections_stitch@v1",
            "name": "stitch",
            "reference_image": "$inputs.image",
            "predictions": "$steps.detection.predictions",
            "overlap_filtering_strategy": "$inputs.overlap_filtering_strategy",
        },
        {
            "type": "roboflow_core/mask_visualization@v1",
            "name": "mask_visualiser",
            "predictions": "$steps.stitch.predictions",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.stitch.predictions",
            "coordinates_system": "own",
        },
        {
            "type": "JsonField",
            "name": "visualisation",
            "selector": "$steps.mask_visualiser.image",
        },
    ],
}


@add_to_workflows_gallery(
    category="Advanced inference techniques",
    use_case_title="SAHI in workflows - instance segmentation",
    use_case_description="""
This example illustrates usage of [SAHI](https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/) 
technique in workflows.

Workflows implementation requires three blocks:

- Image Slicer - which runs a sliding window over image and for each image prepares batch of crops 

- detection model block (in our scenario Roboflow Instance Segmentation model) - which is responsible 
for making predictions on each crop

- Detections stitch - which combines partial predictions for each slice of the image into a single prediction
    """,
    workflow_definition=SAHI_WORKFLOW,
    workflow_name_in_app="sahi-segmentation",
)
def test_sahi_workflow_for_segmentation_with_nms_as_filtering_strategy(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    """
    In this test we check how all blocks that form SAHI technique behave.
    Blocks involved in tests:
    - "roboflow_core/image_slicer@v1" from inference.core.workflows.core_steps.transformations.image_slicer.v1
    - "roboflow_core/detections_stitch@v1", from inference.core.workflows.core_steps.fusion.detections_stitch.v1

    This scenario covers usage of SAHI when overlapping predictions are post-processed with NMS in context
    of instance segmentation model.
    """
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SAHI_WORKFLOW_FOR_SEGMENTATION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [crowd_image],
            "overlap_filtering_strategy": "nms",
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    if not USE_INFERENCE_MODELS:
        assert np.allclose(
            result[0]["predictions"].xyxy,
            np.array(
                [
                    [553, 259, 598, 365],
                    [181, 272, 243, 385],
                    [271, 266, 329, 384],
                    [158, 268, 184, 349],
                    [113, 269, 144, 347],
                    [415, 258, 458, 365],
                    [386, 263, 415, 342],
                    [143, 264, 164, 329],
                    [239, 250, 249, 282],
                    [248, 250, 261, 284],
                    [323, 256, 346, 319],
                    [342, 260, 361, 335],
                    [522, 258, 557, 361],
                    [525, 274, 550, 318],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for first image to be exactly as measured during test creation"
    else:
        assert np.allclose(
            result[0]["predictions"].xyxy,
            np.array(
                [
                    [553, 260, 598, 366],
                    [181, 272, 243, 385],
                    [271, 267, 329, 385],
                    [158, 268, 184, 349],
                    [113, 270, 144, 348],
                    [415, 259, 458, 365],
                    [386, 263, 415, 342],
                    [143, 264, 164, 329],
                    [239, 250, 249, 283],
                    [248, 251, 261, 284],
                    [323, 257, 346, 320],
                    [342, 261, 361, 335],
                    [227, 288, 245, 328],
                    [522, 258, 557, 361],
                    [525, 274, 550, 318],
                ]
            ),
            atol=1e-1,
        ), "Expected boxes for first image to be exactly as measured during test creation"
