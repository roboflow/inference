import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
from av import VideoFrame

from inference.core import logger
from inference.core.env import DEBUG_WEBRTC_PROCESSING_LATENCY
from inference.core.interfaces.camera.entities import VideoFrame as InferenceVideoFrame
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

logging.getLogger("aiortc").setLevel(logging.WARNING)


def process_frame(
    frame: VideoFrame,
    frame_id: int,
    inference_pipeline: InferencePipeline,
    stream_output: Optional[str] = None,
    render_output: bool = True,
    include_errors_on_frame: bool = True,
    declared_fps: float = 30.0,
) -> Tuple[
    Dict[str, Union[WorkflowImageData, Any]],
    Optional[VideoFrame],
    List[str],
    Optional[str],
]:
    np_image = frame.to_ndarray(format="bgr24")
    workflow_output: Dict[str, Union[WorkflowImageData, Any]] = {}
    errors = []
    detected_output = None

    try:
        video_frame = InferenceVideoFrame(
            image=np_image,
            frame_id=frame_id,
            frame_timestamp=datetime.datetime.now(),
            comes_from_video_file=False,
            fps=declared_fps,
            measured_fps=declared_fps,
        )
        workflow_output = inference_pipeline._on_video_frame([video_frame])[0]
    except Exception as e:
        logger.exception("Error in workflow processing")
        errors.append(str(e))

    # If render_output is False, return early without rendering (data-only mode)
    if not render_output:
        return workflow_output, None, errors, None

    # Extract visual output for rendering
    result_np_image: Optional[np.ndarray] = None
    try:
        result_np_image = get_frame_from_workflow_output(
            workflow_output=workflow_output,
            frame_output_key=stream_output,
        )
        if result_np_image is None:
            for k in workflow_output.keys():
                result_np_image = get_frame_from_workflow_output(
                    workflow_output=workflow_output,
                    frame_output_key=k,
                )
                if result_np_image is not None:
                    detected_output = k
                    if stream_output is not None and stream_output != "":
                        errors.append(
                            f"'{stream_output}' not found in workflow outputs, using '{k}' instead"
                        )
                    break
        if result_np_image is None:
            errors.append("Visualisation blocks were not executed")
            errors.append("or workflow was not configured to output visuals.")
            errors.append("Please try to adjust the scene so models detect objects")
            errors.append("or stop preview, update workflow and try again.")
            result_np_image = np_image
    except Exception as e:
        logger.exception("Error extracting visual output")
        result_np_image = np_image
        errors.append(str(e))

    if include_errors_on_frame and errors:
        result_np_image = overlay_text_on_np_frame(
            frame=result_np_image,
            text=errors,
        )

    return (
        workflow_output,
        VideoFrame.from_ndarray(result_np_image, format="bgr24"),
        errors,
        detected_output,
    )


def overlay_text_on_np_frame(frame: np.ndarray, text: List[str]):
    for i, l in enumerate(text):
        frame = cv.putText(
            frame,
            l,
            (10, 20 + 30 * i),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return frame


def get_frame_from_workflow_output(
    workflow_output: Dict[str, Union[WorkflowImageData, Any]], frame_output_key: str
) -> Optional[np.ndarray]:
    latency: Optional[datetime.timedelta] = None
    np_image: Optional[np.ndarray] = None

    step_output = workflow_output.get(frame_output_key)
    if isinstance(step_output, WorkflowImageData):
        if (
            DEBUG_WEBRTC_PROCESSING_LATENCY
            and step_output.video_metadata
            and step_output.video_metadata.frame_timestamp is not None
        ):
            latency = (
                datetime.datetime.now() - step_output.video_metadata.frame_timestamp
            )
        np_image = step_output.numpy_image
    elif isinstance(step_output, dict):
        for frame_output in step_output.values():
            if isinstance(frame_output, WorkflowImageData):
                if (
                    DEBUG_WEBRTC_PROCESSING_LATENCY
                    and frame_output.video_metadata
                    and frame_output.video_metadata.frame_timestamp is not None
                ):
                    latency = (
                        datetime.datetime.now()
                        - frame_output.video_metadata.frame_timestamp
                    )
                np_image = frame_output.numpy_image

    # logger.warning since inference pipeline is noisy on INFO level
    if DEBUG_WEBRTC_PROCESSING_LATENCY and latency is not None:
        logger.warning("Processing latency: %ss", latency.total_seconds())

    return np_image
