"""Streaming SAM2 video tracker backed by ``transformers.Sam2VideoModel``.

This is the HuggingFace-based alternative to the older
``sam2_rt.SAM2ForStream`` (which relies on Meta's ``sam2`` package and
its camera predictor).  It builds on ``HFStreamingVideoBase`` so the
same streaming-friendly interface can be reused by other HF video
trackers (e.g. a future SAM3 video port).

Both are registered and available concurrently:

- ``segment-anything-2-rt`` → ``SAM2ForStream`` (sam2 package; older)
- ``sam2video``             → ``SAM2Video`` (HF transformers; this file)
"""

from typing import Any, Tuple

from inference_models.errors import MissingDependencyError
from inference_models.models.common.hf_streaming_video import HFStreamingVideoBase


class SAM2Video(HFStreamingVideoBase):
    """SAM2 streaming tracker via the HuggingFace transformers port.

    SAM2 only accepts visual (box / point) prompts; text prompting is
    a SAM3 feature and is rejected at the base-class level.
    """

    _supports_text_prompts = False

    @classmethod
    def _resolve_transformers_classes(cls) -> Tuple[Any, Any]:
        try:
            from transformers import Sam2VideoModel, Sam2VideoProcessor
        except ImportError as import_error:
            raise MissingDependencyError(
                message=(
                    "Could not import Sam2VideoModel / Sam2VideoProcessor "
                    "from transformers.  Ensure a transformers version "
                    "that ships SAM2 video support is installed."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/"
                    "runtime-environment/#missingdependencyerror"
                ),
            ) from import_error
        return Sam2VideoModel, Sam2VideoProcessor
