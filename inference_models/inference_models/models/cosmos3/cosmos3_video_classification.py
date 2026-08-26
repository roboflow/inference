import json
import re
from typing import Any, List, Optional, Union

import numpy as np
import torch

from inference_models.models.base.video_classification import (
    VideoClassificationModel,
    VideoSegmentClassification,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner

TEMPORAL_LOCALIZATION_PROMPT_TEMPLATE = (
    "Identify every temporal event in this video that matches the class vocabulary.\n"
    "Class vocabulary: {class_vocabulary}\n"
    "The clip contains {num_frames} frames sampled at {fps} fps.\n"
    "Return STRICT JSON array output using this schema:\n"
    '[{{"start": <frame index>, "end": <frame index>, '
    '"class": "<one of the vocabulary>"}}]\n'
    "Report start and end as frame indices between 0 and {max_frame_idx}, "
    "not timestamps. Events may overlap.\n"
    "Return [] when no event matches the class vocabulary."
)


def _parse_frame_index(value: Any) -> Optional[int]:
    """Return ``value`` when it is a plain JSON integer, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _parse_temporal_segments(
    text: str,
    class_names: Optional[List[str]],
    num_frames: int,
) -> List[VideoSegmentClassification]:
    """Parse temporal-localization output into frame-index ranges.

    An entry survives only when both boundaries are valid integer frame
    indices inside ``[0, num_frames - 1]``; inverted boundaries are
    swapped.
    """
    if not isinstance(text, str) or num_frames <= 0:
        return []
    decoder = json.JSONDecoder()
    entries = None
    for match in re.finditer(r"\[", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            entries = value
            break
    if entries is None:
        return []

    allowed_classes = (
        {str(class_name).strip() for class_name in class_names}
        if class_names is not None
        else None
    )
    max_frame_idx = num_frames - 1
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label_value = entry.get("class")
        if not isinstance(label_value, str):
            continue
        label = label_value.strip()
        if not label or (
            allowed_classes is not None and label not in allowed_classes
        ):
            continue
        start_frame_idx = _parse_frame_index(entry.get("start"))
        end_frame_idx = _parse_frame_index(entry.get("end"))
        if start_frame_idx is None or end_frame_idx is None:
            continue
        if not 0 <= start_frame_idx <= max_frame_idx:
            continue
        if not 0 <= end_frame_idx <= max_frame_idx:
            continue
        if start_frame_idx > end_frame_idx:
            start_frame_idx, end_frame_idx = end_frame_idx, start_frame_idx
        result.append(
            VideoSegmentClassification(
                start_frame_idx=start_frame_idx,
                end_frame_idx=end_frame_idx,
                class_name=label,
            )
        )
    return result


class Cosmos3VideoSegmentClassification(VideoClassificationModel):
    def __init__(self, reasoner: Cosmos3EdgeReasoner):
        self._reasoner = reasoner

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "Cosmos3VideoSegmentClassification":
        reasoner = Cosmos3EdgeReasoner.from_pretrained(model_name_or_path, **kwargs)
        return cls(reasoner=reasoner)

    def infer(
        self,
        frames: List[Union[np.ndarray, torch.Tensor]],
        class_names: Optional[List[str]] = None,
        fps: Optional[float] = None,
        **kwargs,
    ) -> List[VideoSegmentClassification]:
        if fps is None:
            raise ValueError("fps is required for temporal localization")
        normalized_frames = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().permute(1, 2, 0)
                if frame.dtype == torch.bfloat16:
                    frame = frame.float()
                frame = frame.numpy()
            normalized_frames.append(frame)
        class_vocabulary = [str(class_name).strip() for class_name in class_names or []]
        prompt = TEMPORAL_LOCALIZATION_PROMPT_TEMPLATE.format(
            class_vocabulary=json.dumps(class_vocabulary),
            num_frames=len(normalized_frames),
            fps=fps,
            max_frame_idx=max(len(normalized_frames) - 1, 0),
        )
        response = self._reasoner.prompt_video(
            frames=normalized_frames,
            prompt=prompt,
            input_color_format="rgb",
            **kwargs,
        )
        if isinstance(response, dict):
            response = response.get("answer", "")
        return _parse_temporal_segments(
            text=response,
            class_names=class_names,
            num_frames=len(normalized_frames),
        )
