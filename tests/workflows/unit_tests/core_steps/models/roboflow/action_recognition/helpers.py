"""Shared action recognition model test inputs and recording backend."""

from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import torch

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference_models import ActionRecognitionModel
from inference_models import (
    ActionRecognitionPrediction as ModelActionRecognitionPrediction,
)
from inference_models import VideoSampling


class _FakeActionRecognitionModel(ActionRecognitionModel):
    # A plain class attribute overrides the ABC property so tests can set
    # per-instance temporal contracts.
    video_sampling = VideoSampling()

    def __init__(
        self,
        responses: Optional[
            List[Union[List[ModelActionRecognitionPrediction], Exception]]
        ] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.responses = list(responses or [])
        self.calls = []
        self._class_names = class_names

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        raise NotImplementedError

    def infer(
        self,
        frames,
        class_names=None,
        fps=None,
        **kwargs,
    ):
        recorded_frames = [
            frame.clone() if isinstance(frame, torch.Tensor) else frame.copy()
            for frame in frames
        ]
        self.calls.append(
            {
                "frames": recorded_frames,
                "class_names": (list(class_names) if class_names is not None else None),
                "fps": fps,
                "duration_seconds": kwargs.get("duration_seconds"),
                "confidence": kwargs.get("confidence"),
            }
        )
        if not self.responses:
            return []
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _model_segment(
    class_name: str,
    start_frame_idx: int = 0,
    end_frame_idx: int = 0,
) -> ModelActionRecognitionPrediction:
    return ModelActionRecognitionPrediction(
        start_frame_idx=start_frame_idx,
        end_frame_idx=end_frame_idx,
        class_name=class_name,
    )


def _make_frame(
    frame_number: int,
    video_id: str = "stream-0",
    fps: Optional[float] = 4.0,
    measured_fps: Optional[float] = None,
    frame_timestamp: datetime = datetime(2024, 1, 1),
    bgr_color: Optional[List[int]] = None,
    tensor_rgb_color: Optional[List[int]] = None,
    comes_from_video_file: Optional[bool] = None,
    total_frames: Optional[int] = None,
) -> WorkflowImageData:
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        fps=fps,
        measured_fps=measured_fps,
        frame_timestamp=frame_timestamp,
        comes_from_video_file=comes_from_video_file,
        total_frames=total_frames,
    )
    image_kwargs = {}
    if tensor_rgb_color is not None:
        image_kwargs["tensor_image"] = (
            torch.tensor(tensor_rgb_color, dtype=torch.uint8)
            .view(3, 1, 1)
            .expand(3, 2, 2)
            .clone()
        )
    else:
        color = bgr_color or [frame_number % 255, 10, 20]
        image_kwargs["numpy_image"] = np.full((2, 2, 3), color, dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"{video_id}:{frame_number}"),
        workflow_root_ancestor_metadata=ImageParentMetadata(
            parent_id=f"{video_id}:root"
        ),
        video_metadata=metadata,
        **image_kwargs,
    )


def _run(
    block,
    frame: WorkflowImageData,
    class_filter=("walk", "run"),
    window_seconds=1.0,
    stride_seconds=0.5,
    sample_fps=2.0,
    min_frames=1,
    requires_regular_sampling=False,
    max_sample_lateness_seconds=0.0,
    **block_kwargs,
):
    # The temporal contract travels with the model.
    if block._model is not None:
        block._model.video_sampling = VideoSampling(
            window_seconds=window_seconds,
            sample_fps=sample_fps,
            min_frames=min_frames,
            requires_regular_sampling=requires_regular_sampling,
            max_sample_lateness_seconds=max_sample_lateness_seconds,
        )
    return block.run(
        images=[frame],
        class_filter=list(class_filter) if class_filter is not None else None,
        model_id="cosmos-3-edge",
        stride_seconds=stride_seconds,
        **block_kwargs,
    )[0]


def _timeline_entry(
    start_frame_idx: int,
    end_frame_idx: int,
    class_name: str,
) -> ActionRecognitionPrediction:
    return ActionRecognitionPrediction(
        start_frame_idx=start_frame_idx,
        end_frame_idx=end_frame_idx,
        class_name=class_name,
        class_id=0,
    )


def make_block(
    block_type,
    responses: Optional[
        List[Union[List[ModelActionRecognitionPrediction], Exception]]
    ] = None,
    model_class_names: Optional[List[str]] = None,
):
    block = block_type(
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    model = _FakeActionRecognitionModel(
        responses=responses,
        class_names=model_class_names,
    )
    block._model = model
    block._current_model_id = "cosmos-3-edge"
    return block, model
