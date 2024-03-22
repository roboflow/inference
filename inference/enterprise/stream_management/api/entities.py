from typing import Dict, List, Optional, Union

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
    video_reference: Union[str, int, List[Union[str, int]]] = Field(
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
    video_source_properties: Optional[
        Union[Dict[str, float], List[Optional[Dict[str, float]]]]
    ] = Field(
        description="Optional source properties to set up the video source, corresponding to cv2 VideoCapture properties cv2.CAP_PROP_*. If not given, defaults for the video source will be used.",
        examples=[
            {
                "frame_width": 1920,
                "frame_height": 1080,
                "fps": 30.0,
            }
        ],
        default={},
    )
    active_learning_target_dataset: Optional[str] = Field(
        default=None,
        examples=["my_dataset"],
        description="Parameter to be used when Active Learning data registration should happen against different dataset than the one pointed by model_id",
    )
    batch_collection_timeout: Optional[float] = Field(
        default=None,
        examples=[0.1],
        description="Parameter that is important if `video_reference` points multiple video sources. In that case - it dictates how long process of grabbing frames from multiple sources can wait for collection of full batch. See `InferencePipeline` docs for more details.",
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
