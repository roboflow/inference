import json
import math
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
# The checkpoint ignores vocabulary constraints layered onto its trained
# localization prompt (seven prompt variants tested — every one captioned
# freely or collapsed to a whole-clip summary). Vocabulary classification
# therefore runs as a second text-only call that maps each dense caption
# onto the vocabulary or "other".
VOCABULARY_MAPPING_PROMPT_TEMPLATE = (
    "Here are numbered captions of events from a video:\n{captions}\n\n"
    "Here is a list of action classes: {vocab}\n\n"
    "Match each caption to one action class. Use the class text verbatim. "
    'If a caption matches no class, use "other". If a caption describes '
    "several actions or summarizes the video, pick the single best class "
    'or "other". Do not overthink: one quick decision per caption.\n'
    "Return STRICT JSON, one entry per caption number: "
    '{{"1": "<class or other>", "2": "<class or other>", ...}}'
)
# Temporal localization answers carry a think block plus a JSON entry per
# event; the package default of 512 new tokens forces the model to compress
# the clip into one summary segment. 4096 matches the cookbook demo budget.
TEMPORAL_LOCALIZATION_MAX_NEW_TOKENS = 4096

# The open-vocabulary prompt is the cookbook's verbatim temporal-localization
# prompt. Variants that reworded it ("notable events", a "class" key, extra
# format constraints) collapsed the model's output to one whole-clip segment
# on the cookbook's own demo asset; only the trained phrasing densifies.
OPEN_VOCABULARY_TEMPORAL_LOCALIZATION_PROMPT = """List all action segments in the video.

Provide the result in json format with 'seconds' for time depiction for each event. Use keywords 'start', 'end' and 'caption' in the json output. Please list multiple events if applicable.

```json
[
{
  "start": t_start,
  "end": t_end,
  "caption": EVENT1
},
{
  "start": t_start,
  "end": t_end,
  "caption": EVENT2
},
...
]
```"""


def _parse_seconds(value: Any) -> Optional[float]:
    """Return a finite JSON number as seconds, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _parse_first_json_object(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_temporal_segments(
    text: str,
    class_names: Optional[List[str]],
    num_frames: int,
    fps: float,
) -> List[VideoSegmentClassificationPrediction]:
    """Parse second-based temporal output into frame-index ranges."""
    if (
        not isinstance(text, str)
        or num_frames <= 0
        or fps <= 0
        or not math.isfinite(fps)
    ):
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
    duration = num_frames / fps
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        # The open-vocabulary prompt keeps the cookbook's "caption" key;
        # the vocabulary prompt asks for "class".
        label_value = entry.get("class")
        if not isinstance(label_value, str):
            label_value = entry.get("caption")
        if not isinstance(label_value, str):
            continue
        label = label_value.strip()
        if not label or (
            allowed_classes is not None and label not in allowed_classes
        ):
            continue
        start_seconds = _parse_seconds(entry.get("start"))
        end_seconds = _parse_seconds(entry.get("end"))
        if start_seconds is None or end_seconds is None:
            continue
        if end_seconds < 0 or start_seconds > duration:
            continue
        start_frame_idx = math.floor(start_seconds * fps)
        end_frame_idx = math.ceil(end_seconds * fps)
        start_frame_idx = min(max(start_frame_idx, 0), max_frame_idx)
        end_frame_idx = min(max(end_frame_idx, 0), max_frame_idx)
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

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names

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
        kwargs.setdefault("max_new_tokens", TEMPORAL_LOCALIZATION_MAX_NEW_TOKENS)
        response = self._reasoner.prompt_video(
            frames=normalized_frames,
            prompt=OPEN_VOCABULARY_TEMPORAL_LOCALIZATION_PROMPT,
            input_color_format="rgb",
            video_fps=fps,
            **kwargs,
        )
        if isinstance(response, dict):
            response = response.get("answer", "")
        segments = _parse_temporal_segments(
            text=response,
            class_names=None,
            num_frames=len(normalized_frames),
            fps=fps,
        )
        if vocabulary is None or not segments:
            return segments
        return self._map_segments_to_vocabulary(
            segments=segments, vocabulary=vocabulary
        )

    def _map_segments_to_vocabulary(
        self,
        segments: List[VideoSegmentClassificationPrediction],
        vocabulary: List[str],
    ) -> List[VideoSegmentClassificationPrediction]:
        cleaned_vocabulary = [str(class_name).strip() for class_name in vocabulary]
        numbered_captions = {
            str(index + 1): segment.class_name
            for index, segment in enumerate(segments)
        }
        prompt = VOCABULARY_MAPPING_PROMPT_TEMPLATE.format(
            captions=json.dumps(numbered_captions, indent=1),
            vocab=json.dumps(cleaned_vocabulary),
        )
        answer = self._reasoner.prompt_text(
            prompt=prompt,
            max_new_tokens=TEMPORAL_LOCALIZATION_MAX_NEW_TOKENS,
        )
        if isinstance(answer, dict):
            answer = answer.get("answer", "")
        mapping = _parse_first_json_object(answer)
        if mapping is None:
            return []
        allowed_classes = set(cleaned_vocabulary)
        result = []
        for index, segment in enumerate(segments):
            label = mapping.get(str(index + 1))
            if not isinstance(label, str):
                continue
            label = label.strip()
            if label not in allowed_classes:
                continue
            result.append(
                VideoSegmentClassificationPrediction(
                    start_frame_idx=segment.start_frame_idx,
                    end_frame_idx=segment.end_frame_idx,
                    class_name=label,
                )
            )
        return result
