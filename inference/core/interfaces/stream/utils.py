import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from typing import Callable, Dict, List, Optional, TypeVar, Union

import numpy as np

from inference.core.env import DEFAULT_BUFFER_SIZE, ENABLE_WORKFLOWS_PROFILING
from inference.core.interfaces.camera.entities import (
    StatusUpdate,
    VideoFrame,
    VideoSourceIdentifier,
)
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
    VideoSource,
)
from inference.core.workflows.execution_engine.profiling.core import WorkflowsProfiler

T = TypeVar("T")


def prepare_video_sources(
    video_reference: Union[VideoSourceIdentifier, List[VideoSourceIdentifier]],
    video_source_properties: Optional[
        Union[Dict[str, float], List[Optional[Dict[str, float]]]]
    ],
    status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]],
    source_buffer_filling_strategy: Optional[BufferFillingStrategy],
    source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy],
    desired_source_fps: Optional[Union[float, int]] = None,
    decoding_buffer_size: int = DEFAULT_BUFFER_SIZE,
    allow_tensor_frames: bool = False,
) -> List[VideoSource]:
    video_reference = wrap_in_list(element=video_reference)
    if len(video_reference) < 1:
        raise ValueError(
            "Cannot initialise `InferencePipeline` with empty `video_reference`"
        )
    video_source_properties = wrap_in_list(element=video_source_properties)
    video_source_properties = broadcast_elements(
        elements=video_source_properties,
        desired_length=len(video_reference),
        error_description="Cannot apply `video_source_properties` to video sources due to missmatch in "
        "number of entries in properties configuration.",
    )
    return initialise_video_sources(
        video_reference=video_reference,
        video_source_properties=video_source_properties,
        status_update_handlers=status_update_handlers,
        source_buffer_filling_strategy=source_buffer_filling_strategy,
        source_buffer_consumption_strategy=source_buffer_consumption_strategy,
        desired_source_fps=desired_source_fps,
        decoding_buffer_size=decoding_buffer_size,
        allow_tensor_frames=allow_tensor_frames,
    )


def wrap_in_list(element: Union[T, List[T]]) -> List[T]:
    if not issubclass(type(element), list):
        element = [element]
    return element


def broadcast_elements(
    elements: List[T],
    desired_length: int,
    error_description: str,
) -> List[T]:
    if len(elements) == desired_length:
        return elements
    if len(elements) != 1:
        raise ValueError(error_description)
    return elements * desired_length


def initialise_video_sources(
    video_reference: List[VideoSourceIdentifier],
    video_source_properties: List[Optional[Dict[str, float]]],
    status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]],
    source_buffer_filling_strategy: Optional[BufferFillingStrategy],
    source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy],
    desired_source_fps: Optional[Union[float, int]] = None,
    decoding_buffer_size: int = DEFAULT_BUFFER_SIZE,
    allow_tensor_frames: bool = False,
) -> List[VideoSource]:
    if isinstance(source_buffer_filling_strategy, str):
        source_buffer_filling_strategy = BufferFillingStrategy(
            source_buffer_filling_strategy
        )
    if isinstance(source_buffer_consumption_strategy, str):
        source_buffer_consumption_strategy = BufferConsumptionStrategy(
            source_buffer_consumption_strategy
        )
    return [
        VideoSource.init(
            video_reference=reference,
            status_update_handlers=status_update_handlers,
            buffer_filling_strategy=source_buffer_filling_strategy,
            buffer_consumption_strategy=source_buffer_consumption_strategy,
            video_source_properties=source_properties,
            source_id=i,
            desired_fps=desired_source_fps,
            buffer_size=decoding_buffer_size,
            allow_tensor_frames=allow_tensor_frames,
        )
        for i, (reference, source_properties) in enumerate(
            zip(video_reference, video_source_properties)
        )
    ]


def materialise_video_frames_for_sink(
    video_frames: Union[VideoFrame, List[Optional[VideoFrame]]],
) -> Union[VideoFrame, List[Optional[VideoFrame]]]:
    if isinstance(video_frames, list):
        return [
            materialise_video_frame_for_sink(video_frame)
            for video_frame in video_frames
        ]
    return materialise_video_frame_for_sink(video_frames)


def materialise_video_frame_for_sink(
    video_frame: Optional[VideoFrame],
) -> Optional[VideoFrame]:
    """Convert a tensor video frame to a host BGR ``np.ndarray`` frame.

    Under ENABLE_TENSOR_DATA_REPRESENTATION the pipeline hands sinks the
    original on-device tensor frame — nothing materialises in dispatch. Sinks
    that actually consume pixels on CPU (visualisation, active learning, ...)
    call this at their own boundary and pay the device-to-host copy only when
    the pixels are genuinely needed. A numpy frame passes through untouched.
    """
    if video_frame is None or isinstance(video_frame.image, np.ndarray):
        return video_frame

    tensor_image = video_frame.image.detach().cpu()
    if tensor_image.ndim == 2:
        numpy_image = tensor_image.contiguous().numpy()
    elif tensor_image.ndim == 3 and tensor_image.shape[0] == 1:
        numpy_image = tensor_image[0].contiguous().numpy()
    elif tensor_image.ndim == 3 and tensor_image.shape[0] in {3, 4}:
        numpy_image = tensor_image.permute(1, 2, 0).contiguous().numpy()
        channel_order = [2, 1, 0]
        if tensor_image.shape[0] == 4:
            channel_order.append(3)
        numpy_image = np.ascontiguousarray(numpy_image[..., channel_order])
    else:
        raise ValueError("Tensor video frames must use HW, 1CHW, 3CHW, or 4CHW layout")
    return replace(video_frame, image=numpy_image)


def on_pipeline_end(
    thread_pool_executor: ThreadPoolExecutor,
    cancel_thread_pool_tasks_on_exit: bool,
    profiler: WorkflowsProfiler,
    profiling_directory: str,
) -> None:
    if ENABLE_WORKFLOWS_PROFILING:
        save_workflows_profiler_trace(
            directory=profiling_directory,
            profiler_trace=profiler.export_trace(),
        )
    try:
        thread_pool_executor.shutdown(cancel_futures=cancel_thread_pool_tasks_on_exit)
    except TypeError:
        # we must support Python 3.8 which do not support `cancel_futures`
        thread_pool_executor.shutdown()


def save_workflows_profiler_trace(
    directory: str,
    profiler_trace: List[dict],
) -> None:
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    formatted_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    track_path = os.path.join(
        directory, f"inference_pipeline_workflow_execution_tack_{formatted_time}.json"
    )
    with open(track_path, "w") as f:
        json.dump(profiler_trace, f)
