import json
import re
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch

from inference_models.models.base.video_segment_classification import (
    VideoSegmentClassificationModel,
    VideoSegmentClassificationPrediction,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner

# Prompt format follows NVIDIA's Cosmos3 reasoner temporal-localization cookbook:
# https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md#temporal-localization
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
OPEN_VOCABULARY_TEMPORAL_LOCALIZATION_PROMPT_TEMPLATE = (
    "Identify notable temporal events in this video and label each with a short "
    "lowercase class phrase.\n"
    "The clip contains {num_frames} frames sampled at {fps} fps.\n"
    "Return STRICT JSON array output using this schema:\n"
    '[{{"start": <frame index>, "end": <frame index>, '
    '"class": "<short lowercase class phrase>"}}]\n'
    "Report start and end as frame indices between 0 and {max_frame_idx}, "
    "not timestamps. Events may overlap.\n"
    "Return [] when no notable temporal event occurs."
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
) -> List[VideoSegmentClassificationPrediction]:
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
            VideoSegmentClassificationPrediction(
                start_frame_idx=start_frame_idx,
                end_frame_idx=end_frame_idx,
                class_name=label,
            )
        )
    return result


class Cosmos3EdgeVideoSegmentClassification(VideoSegmentClassificationModel):
    def __init__(
        self,
        reasoner: Cosmos3EdgeReasoner,
        class_names: Optional[List[str]] = None,
    ):
        self._reasoner = reasoner
        self._class_names = class_names

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "Cosmos3EdgeVideoSegmentClassification":
        reasoner = Cosmos3EdgeReasoner.from_pretrained(model_name_or_path, **kwargs)
        class_names = None
        try:
            with open(Path(model_name_or_path) / "model_config.json") as config_file:
                model_config = json.load(config_file)
        except (FileNotFoundError, json.JSONDecodeError):
            model_config = None
        if isinstance(model_config, dict):
            configured_class_names = model_config.get("class_names")
            if isinstance(configured_class_names, list) and all(
                isinstance(class_name, str) for class_name in configured_class_names
            ):
                class_names = configured_class_names
        return cls(reasoner=reasoner, class_names=class_names)

    def infer(
        self,
        frames: List[Union[np.ndarray, torch.Tensor]],
        class_names: Optional[List[str]] = None,
        fps: Optional[float] = None,
        **kwargs,
    ) -> List[VideoSegmentClassificationPrediction]:
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
        vocabulary = class_names or self.class_names or None
        prompt_template = (
            TEMPORAL_LOCALIZATION_PROMPT_TEMPLATE
            if vocabulary is not None
            else OPEN_VOCABULARY_TEMPORAL_LOCALIZATION_PROMPT_TEMPLATE
        )
        prompt_parameters = {
            "num_frames": len(normalized_frames),
            "fps": fps,
            "max_frame_idx": max(len(normalized_frames) - 1, 0),
        }
        if vocabulary is not None:
            prompt_parameters["class_vocabulary"] = json.dumps(
                [str(class_name).strip() for class_name in vocabulary]
            )
        prompt = prompt_template.format(**prompt_parameters)
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
            class_names=vocabulary,
            num_frames=len(normalized_frames),
        )
