"""Streaming SAM3 video tracker backed by ``transformers.Sam3VideoModel``.

The native ``sam3`` package's video predictor is session based and
requires a pre-existing MP4 / JPEG directory — it is not designed for
frame-by-frame streaming from memory.  HuggingFace's port exposes the
underlying model's streaming interface (``init_video_session`` +
per-frame ``model(inference_session=..., frame=...)``), which is what
``InferencePipeline``-style streaming needs.

SAM3 accepts both visual (box) prompts and open-vocabulary text
prompts.  Everything else — session management, the opaque
``state_dict`` contract, output unpacking — lives in the shared
``HFStreamingVideoBase``.
"""

from typing import Any, Tuple

from inference_models.errors import MissingDependencyError
from inference_models.models.common.hf_streaming_video import HFStreamingVideoBase


class SAM3Video(HFStreamingVideoBase):
    """SAM3 streaming tracker via the HuggingFace transformers port."""

    _supports_text_prompts = True

    @classmethod
    def _resolve_transformers_classes(cls) -> Tuple[Any, Any]:
        try:
            from transformers import Sam3VideoModel, Sam3VideoProcessor
        except ImportError as import_error:
            raise MissingDependencyError(
                message=(
                    "Could not import Sam3VideoModel / Sam3VideoProcessor "
                    "from transformers.  Ensure a transformers version "
                    "that ships SAM3 video support is installed."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/"
                    "runtime-environment/#missingdependencyerror"
                ),
            ) from import_error
        return Sam3VideoModel, Sam3VideoProcessor
