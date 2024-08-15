import matplotlib.pyplot as plt
import numpy as np
import supervision as sv

from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

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
            "type": "roboflow_core/roboflow_object_detection_model@v1",
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

    slicer = sv.InferenceSlicer(
        callback=slicer_callback,
        slice_wh=(640, 640),
        overlap_ratio_wh=(0.2, 0.2),
        overlap_filter_strategy=sv.OverlapFilter.NON_MAX_SUPPRESSION,
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

    # then
    assert np.allclose(
        detections_obtained_directly.xyxy,
        workflow_result[0]["predictions"].xyxy,
        atol=1e-1,
    ), "Expected bounding boxes to be the same for workflow SAHI and direct SAHI"
    assert np.allclose(
        detections_obtained_directly.confidence,
        workflow_result[0]["predictions"].confidence,
        atol=1e-4,
    ), "Expected confidences to be the same for workflow SAHI and direct SAHI"
    assert np.all(
        detections_obtained_directly.class_id
        == workflow_result[0]["predictions"].class_id
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
            "type": "roboflow_core/roboflow_object_detection_model@v1",
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
