from typing import Callable, Dict, List, Optional, TypeVar, Union

from inference.core import logger
from inference.core.interfaces.camera.entities import StatusUpdate
from inference.core.interfaces.camera.utils import FPSLimiterStrategy
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
    VideoSource,
)

T = TypeVar("T")


def prepare_video_sources(
    video_reference: Union[str, int, List[Union[str, int]]],
    video_source_properties: Optional[
        Union[Dict[str, float], List[Optional[Dict[str, float]]]]
    ],
    status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]],
    source_buffer_filling_strategy: Optional[BufferFillingStrategy],
    source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy],
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
    if len(elements) <= 1:
        raise ValueError(error_description)
    return elements * desired_length


def initialise_video_sources(
    video_reference: List[Union[str, int]],
    video_source_properties: List[Optional[Dict[str, float]]],
    status_update_handlers: Optional[List[Callable[[StatusUpdate], None]]],
    source_buffer_filling_strategy: Optional[BufferFillingStrategy],
    source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy],
) -> List[VideoSource]:
    return [
        VideoSource.init(
            video_reference=reference,
            status_update_handlers=status_update_handlers,
            buffer_filling_strategy=source_buffer_filling_strategy,
            buffer_consumption_strategy=source_buffer_consumption_strategy,
            video_source_properties=source_properties,
            source_id=i,
        )
        for i, (reference, source_properties) in enumerate(
            zip(video_reference, video_source_properties)
        )
    ]


def negotiate_rate_limiter_strategy(
    video_sources: List[VideoSource],
    max_fps: Optional[float],
) -> FPSLimiterStrategy:
    source_types_statuses = {
        s.describe_source().source_properties.is_file for s in video_sources
    }
    if len(source_types_statuses) == 2 and max_fps is not None:
        logger.warning(
            f"`InferencePipeline` started with FPS limit rate. Detected both files and video streams as video sources. "
            f"Rate limiter cannot satisfy both - choosing `FPSLimiterStrategy.DROP` which may drop file sources frames "
            f"that would not happen if only video files are submitted into processing."
        )
        return FPSLimiterStrategy.DROP
    if True in source_types_statuses:
        return FPSLimiterStrategy.WAIT
    return FPSLimiterStrategy.DROP
