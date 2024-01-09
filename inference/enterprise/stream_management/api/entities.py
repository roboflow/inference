from typing import List, Optional, Union

from pydantic import BaseModel, Field

from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)


class UDPSinkConfiguration(BaseModel):
    type: str = Field(
        description="Type identifier field. Must be `udp_sink`", default="udp_sink"
    )
    host: str = Field(description="Host of UDP sink.")
    port: int = Field(description="Port of UDP sink.")


class ObjectDetectionModelConfiguration(BaseModel):
    type: str = Field(
        description="Type identifier field. Must be `object-detection`",
        default="object-detection",
    )
    class_agnostic_nms: Optional[bool] = Field(
        description="Flag to decide if class agnostic NMS to be applied. If not given, default or InferencePipeline host env will be used.",
        default=None,
    )
    confidence: Optional[float] = Field(
        description="Confidence threshold for predictions. If not given, default or InferencePipeline host env will be used.",
        default=None,
    )
    iou_threshold: Optional[float] = Field(
        description="IoU threshold of post-processing. If not given, default or InferencePipeline host env will be used.",
        default=None,
    )
    max_candidates: Optional[int] = Field(
        description="Max candidates in post-processing. If not given, default or InferencePipeline host env will be used.",
        default=None,
    )
    max_detections: Optional[int] = Field(
        description="Max detections in post-processing. If not given, default or InferencePipeline host env will be used.",
        default=None,
    )


class PipelineInitialisationRequest(BaseModel):
    model_id: str = Field(description="Roboflow model id")
    video_reference: Union[str, int] = Field(
        description="Reference to video source - either stream, video file or device. It must be accessible from the host running inference stream"
    )
    sink_configuration: UDPSinkConfiguration = Field(
        description="Configuration of the sink."
    )
    api_key: Optional[str] = Field(description="Roboflow API key", default=None)
    max_fps: Optional[Union[float, int]] = Field(
        description="Limit of FPS in video processing.", default=None
    )
    source_buffer_filling_strategy: Optional[str] = Field(
        description=f"`source_buffer_filling_strategy` parameter of Inference Pipeline (see docs). One of {[e.value for e in BufferFillingStrategy]}",
        default=None,
    )
    source_buffer_consumption_strategy: Optional[str] = Field(
        description=f"`source_buffer_consumption_strategy` parameter of Inference Pipeline (see docs). One of {[e.value for e in BufferConsumptionStrategy]}",
        default=None,
    )
    model_configuration: ObjectDetectionModelConfiguration = Field(
        description="Configuration of the model",
        default_factory=ObjectDetectionModelConfiguration,
    )
    active_learning_enabled: Optional[bool] = Field(
        description="Flag to decide if Active Learning middleware should be enabled. If not given - env variable `ACTIVE_LEARNING_ENABLED` will be used (with default `True`).",
        default=None,
    )


class CommandContext(BaseModel):
    request_id: Optional[str] = Field(
        description="Server-side request ID", default=None
    )
    pipeline_id: Optional[str] = Field(
        description="Identifier of pipeline connected to operation", default=None
    )


class CommandResponse(BaseModel):
    status: str = Field(description="Operation status")
    context: CommandContext = Field(description="Context of the command.")


class InferencePipelineStatusResponse(CommandResponse):
    report: dict


class ListPipelinesResponse(CommandResponse):
    pipelines: List[str] = Field(description="List IDs of active pipelines")
