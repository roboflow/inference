import ctypes
import datetime
import logging
import struct
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
from av import VideoFrame

from inference.core import logger
from inference.core.env import DEBUG_WEBRTC_PROCESSING_LATENCY
from inference.core.interfaces.camera.entities import VideoFrame as InferenceVideoFrame
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.webrtc_worker.entities import VIDEO_FILE_HEADER_SIZE
from inference.core.utils.roboflow import get_model_id_chunks
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.models.aliases import resolve_roboflow_model_alias
from inference.usage_tracking.collector import usage_collector

logging.getLogger("aiortc").setLevel(logging.WARNING)


def detect_image_output(
    workflow_output: Dict[str, Union[WorkflowImageData, Any]],
) -> Optional[str]:
    """Detect the first available image output field in workflow output."""
    for output_name in workflow_output.keys():
        if (
            get_frame_from_workflow_output(
                workflow_output=workflow_output,
                frame_output_key=output_name,
            )
            is not None
        ):
            return output_name
    return None


def process_frame(
    frame: VideoFrame,
    frame_id: int,
    inference_pipeline: InferencePipeline,
    stream_output: Optional[str] = None,
    render_output: bool = True,
    include_errors_on_frame: bool = True,
) -> Tuple[
    Dict[str, Union[WorkflowImageData, Any]],
    Optional[VideoFrame],
    List[str],
]:
    np_image = frame.to_ndarray(format="bgr24")
    workflow_output: Dict[str, Union[WorkflowImageData, Any]] = {}
    errors = []

    try:
        video_frame = InferenceVideoFrame(
            image=np_image,
            frame_id=frame_id,
            frame_timestamp=datetime.datetime.now(),
            comes_from_video_file=False,
            fps=30,  # placeholder
            measured_fps=30,  # placeholder
        )
        workflow_output = inference_pipeline._on_video_frame([video_frame])[0]
    except Exception as e:
        logger.exception("Error in workflow processing")
        errors.append(str(e))

    if not render_output:
        return workflow_output, None, errors

    if stream_output is None:
        errors.append("stream_output is required when render_output=True")
        return (
            workflow_output,
            VideoFrame.from_ndarray(np_image, format="bgr24"),
            errors,
        )

    result_np_image: Optional[np.ndarray] = None
    try:
        result_np_image = get_frame_from_workflow_output(
            workflow_output=workflow_output,
            frame_output_key=stream_output,
        )
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


def workflow_contains_instant_model(workflow_specification: Dict[str, Any]):
    for step in workflow_specification["steps"]:
        step_type = step["type"]
        if "roboflow_core/roboflow_object_detection_model" in step_type:
            if "model_id" not in step:
                continue
            model_id = step["model_id"]
            model_id = resolve_roboflow_model_alias(model_id=model_id)
            _, version_id = get_model_id_chunks(model_id=model_id)
            if version_id is None:
                return True
    return False


def workflow_contains_preloaded_model(
    workflow_specification: Dict[str, Any], preload_models: List[str]
):
    preload_models = set(preload_models)
    for step in workflow_specification["steps"]:
        if "model_id" not in step:
            continue
        model_id = step["model_id"]
        resolved_model_id = resolve_roboflow_model_alias(model_id=model_id)
        if model_id in preload_models or resolved_model_id in preload_models:
            return True
    return False


# Video File Upload Protocol
# Header: [chunk_index:u32][total_chunks:u32][payload]
def parse_video_file_chunk(message: bytes) -> Tuple[int, int, bytes]:
    """Parse video file chunk message.

    Returns: (chunk_index, total_chunks, payload)
    """
    if len(message) < VIDEO_FILE_HEADER_SIZE:
        raise ValueError(f"Message too short: {len(message)} bytes")
    chunk_index, total_chunks = struct.unpack("<II", message[:8])
    return chunk_index, total_chunks, message[8:]


def warmup_cuda(
    max_retries: int = 10,
    retry_delay: float = 0.5,
):
    cu = ctypes.CDLL("libcuda.so.1")

    for attempt in range(max_retries):
        rc = cu.cuInit(0)

        if rc == 0:
            break
        else:
            if attempt < max_retries - 1:
                logger.warning(
                    "cuInit failed on attempt %s/%s with code %s, retrying...",
                    attempt + 1,
                    max_retries,
                    rc,
                )
                time.sleep(retry_delay)
    else:
        raise RuntimeError(f"CUDA initialization failed after {max_retries} attempts")

    logger.info("CUDA initialization succeeded")


def is_over_quota(api_key: str) -> bool:
    api_key_plan_details = usage_collector._plan_details.get_api_key_plan(
        api_key=api_key
    )
    is_over_quota = api_key_plan_details.get(
        usage_collector._plan_details._over_quota_col_name
    )
    return is_over_quota


def get_video_rotation(filepath: str) -> int:
    """Detect video rotation from metadata (displaymatrix or rotate tag).

    Args:
        filepath: Path to the video file

    Returns:
        Rotation in degrees (-90, 0, 90, 180, 270) or 0 if not found.
        Negative values indicate counter-clockwise rotation.
    """
    import json
    import os
    import subprocess

    logger.info("Detecting video rotation for: %s", filepath)

    # Check if file exists
    if not os.path.exists(filepath):
        logger.warning("Video file does not exist: %s", filepath)
        return 0

    logger.info("Video file exists, size: %d bytes", os.path.getsize(filepath))

    try:
        # Use -show_streams which is compatible with all ffprobe versions
        # This returns full stream info including tags and side_data_list
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-of",
                "json",
                filepath,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        logger.info(
            "ffprobe result: returncode=%s, stdout_len=%s, stderr=%s",
            result.returncode,
            len(result.stdout) if result.stdout else 0,
            result.stderr[:500] if result.stderr else "(empty)",
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                stream = streams[0]
                # Check displaymatrix side_data first
                for sd in stream.get("side_data_list", []):
                    if "rotation" in sd:
                        rotation = int(sd["rotation"])
                        logger.info(
                            "Video rotation detected: %d° (from displaymatrix)",
                            rotation,
                        )
                        return rotation
                # Fall back to rotate tag in stream tags
                rotate_str = stream.get("tags", {}).get("rotate", "0")
                rotation = int(rotate_str)
                if rotation != 0:
                    logger.info(
                        "Video rotation detected: %d° (from rotate tag)", rotation
                    )
                    return rotation
                logger.info("No rotation metadata found in stream")
            else:
                logger.info("ffprobe returned no streams for rotation detection")
        else:
            logger.warning(
                "ffprobe failed with returncode %s: %s",
                result.returncode,
                result.stderr,
            )
    except FileNotFoundError:
        logger.warning("ffprobe not available - binary not found in PATH")
    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timed out after 5 seconds")
    except Exception as e:
        logger.warning("ffprobe rotation detection failed: %s", e)

    logger.info("No rotation detected, returning 0")
    return 0


def get_cv2_rotation_code(rotation: int) -> Optional[int]:
    """Get OpenCV rotation code to correct a given rotation.

    Args:
        rotation: Rotation angle in degrees from metadata

    Returns:
        cv2 rotation constant or None if no correction needed
    """
    # The displaymatrix rotation indicates how the video is rotated.
    # To correct it, we apply the OPPOSITE rotation.
    if rotation in (-90, 270):
        return cv.ROTATE_90_CLOCKWISE
    elif rotation in (90, -270):
        return cv.ROTATE_90_COUNTERCLOCKWISE
    elif rotation in (180, -180):
        return cv.ROTATE_180
    return None


def rotate_video_frame(frame: VideoFrame, rotation_code: int) -> VideoFrame:
    """Apply rotation to a video frame using OpenCV.

    Args:
        frame: Input VideoFrame
        rotation_code: cv2 rotation constant (ROTATE_90_CLOCKWISE, etc.)

    Returns:
        Rotated VideoFrame
    """
    img = frame.to_ndarray(format="bgr24")
    img = cv.rotate(img, rotation_code)
    return VideoFrame.from_ndarray(img, format="bgr24")
