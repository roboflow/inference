from typing import Any, List, Optional

from inference.core.entities.requests.action_recognition import (
    ActionRecognitionInferenceRequest,
)
from inference.core.entities.responses.action_recognition import (
    ActionRecognitionInferenceResponse,
    ActionRecognitionSegment,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    DISABLED_INFERENCE_MODELS_BACKENDS,
    VALID_INFERENCE_MODELS_BACKENDS,
)
from inference.core.models.base import Model
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.video_utils import probe_video, read_frames, video_source_path
from inference_models import AutoModel
from inference_models.models.base.action_recognition import (
    ActionRecognitionModel,
    merge_segment,
    plan_windows,
)


class InferenceModelsActionRecognitionAdapter(Model):
    """Serves a clip to an action recognition model, one window at a time.

    The model declares how a clip is cut and sampled, so a caller never states
    a window length or a frame rate. Windows tile from the start of the clip
    and the trailing remainder is dropped, which is how training validates.
    """

    def __init__(self, model_id: str, api_key: str = None, **kwargs):
        super().__init__()
        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
        self.api_key = api_key if api_key else API_KEY
        self.task_type = "action-recognition"
        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        loaded_model = AutoModel.from_pretrained(
            model_id_or_path=_weights_id(model_id=model_id),
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )
        self._model: ActionRecognitionModel = _as_action_recognition_model(
            model=loaded_model, model_id=model_id
        )

    def infer_from_request(
        self, request: ActionRecognitionInferenceRequest
    ) -> ActionRecognitionInferenceResponse:
        sampling = self._model.video_sampling
        class_filter = request.class_filter or None
        id_vocabulary = self._model.class_names or class_filter or None
        with video_source_path(
            video_type=request.video.type, value=request.video.value
        ) as path:
            source_fps, frame_count = probe_video(path=path)
            windows = plan_windows(
                frame_count=frame_count,
                source_fps=source_fps,
                sampling=sampling,
            )
            timeline: List[ActionRecognitionSegment] = []
            for window in windows:
                frames = read_frames(
                    path=path,
                    frame_indices=window.frame_indices,
                    max_frame_side=sampling.max_frame_side,
                )
                if len(frames) < max(1, sampling.min_frames):
                    continue
                self._merge_window(
                    timeline=timeline,
                    window_frame_indices=window.frame_indices[: len(frames)],
                    segments=self._model.infer(
                        frames=frames,
                        class_names=class_filter,
                        fps=window.sample_fps,
                    ),
                    id_vocabulary=id_vocabulary,
                    source_fps=source_fps,
                    window_sample_fps=window.sample_fps,
                )
        timeline.sort(key=lambda entry: (entry.start_frame_idx, entry.class_id))
        return ActionRecognitionInferenceResponse(
            timeline=timeline,
            source_fps=source_fps,
            frame_count=frame_count,
            windows_classified=len(windows),
        )

    def _merge_window(
        self,
        timeline: List[ActionRecognitionSegment],
        window_frame_indices: Any,
        segments: List[Any],
        id_vocabulary: Optional[List[str]],
        source_fps: float,
        window_sample_fps: float,
    ) -> None:
        sample_count = len(window_frame_indices)
        if sample_count == 0:
            return
        # A window's segments index its own frames; the timeline counts the
        # clip's.
        stride = max(1.0, source_fps / window_sample_fps)
        for segment in segments:
            start_idx = min(sample_count - 1, max(0, int(segment.start_frame_idx)))
            end_idx = min(sample_count - 1, max(0, int(segment.end_frame_idx)))
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            merge_segment(
                timeline=timeline,
                segment=ActionRecognitionSegment(
                    start_frame_idx=window_frame_indices[start_idx],
                    end_frame_idx=window_frame_indices[end_idx],
                    class_name=segment.class_name,
                    class_id=(
                        id_vocabulary.index(segment.class_name)
                        if id_vocabulary is not None
                        and segment.class_name in id_vocabulary
                        else -1
                    ),
                ),
                stride=stride,
            )

    def preprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "Action recognition reads a clip through infer_from_request."
        )

    def predict(self, *args, **kwargs):
        raise NotImplementedError(
            "Action recognition reads a clip through infer_from_request."
        )

    def postprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "Action recognition reads a clip through infer_from_request."
        )

    def clear_cache(self, delete_from_disk: bool = True) -> None:
        pass


def _weights_id(model_id: str) -> str:
    """Strip the task suffix a hosted base carries under this task.

    The hosted reasoner serves more than one task, so it is addressed here as
    "cosmos-3-edge/action_recognition" while its weights answer to
    "cosmos-3-edge".
    """
    task_suffix = "/action_recognition"
    if model_id.endswith(task_suffix):
        return model_id[: -len(task_suffix)]
    return model_id


def _as_action_recognition_model(model: Any, model_id: str) -> ActionRecognitionModel:
    """Accept a model that already serves the task, or wrap a bare reasoner."""
    if isinstance(model, ActionRecognitionModel):
        return model
    from inference_models.models.cosmos3.cosmos3_action_recognition import (
        Cosmos3EdgeActionRecognition,
    )
    from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner

    if isinstance(model, Cosmos3EdgeReasoner):
        return Cosmos3EdgeActionRecognition(reasoner=model)
    raise ValueError(f"Model {model_id} does not support action recognition.")
