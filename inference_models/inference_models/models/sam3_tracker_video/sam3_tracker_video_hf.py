"""Streaming SAM3 visually prompted tracker backed by
``transformers.Sam3TrackerVideoModel``.

SAM3's video stack in transformers comes in two flavours:

- ``Sam3VideoModel`` / ``Sam3VideoProcessor`` — open-vocabulary
  *concept* tracking from text prompts (wrapped by
  ``sam3_video.SAM3Video``).
- ``Sam3TrackerVideoModel`` / ``Sam3TrackerVideoProcessor`` — the
  SAM2-style visually prompted (box / point) tracker, wrapped here.
  It is SAM3's drop-in upgrade for the ``sam2video`` use case: same
  prompt vocabulary and streaming contract, but with SAM3's larger
  perception-encoder backbone, which markedly improves identity
  retention on long videos and crowded scenes (see the SAM3 paper's
  VOS results vs SAM 2.1).

Both tracker families load from the same ``sam3video`` weights
package — the checkpoint contains the detector and tracker weights,
and each transformers class picks the subset it needs.

The processor surface is shape-identical to ``Sam2VideoProcessor``
(``init_video_session`` / ``add_inputs_to_inference_session`` /
``post_process_masks``), so the whole streaming contract is inherited
from ``HFStreamingVideoBase`` unchanged.  Text prompts are *not* part
of this family's vocabulary (no ``add_text_prompt``) and are rejected
at the base-class level; for text-prompted tracking use ``sam3video``.
"""

from typing import Any, Tuple

from inference_models.errors import MissingDependencyError
from inference_models.models.common.hf_streaming_video import HFStreamingVideoBase


class SAM3TrackerVideo(HFStreamingVideoBase):
    """SAM3 visually prompted streaming tracker via the transformers port.

    Accepts box prompts only (the SAM2-shaped contract); text prompting
    belongs to the ``sam3video`` concept tracker.
    """

    _supports_text_prompts = False

    @classmethod
    def _resolve_transformers_classes(cls) -> Tuple[Any, Any]:
        try:
            from transformers import (
                Sam3TrackerVideoModel,
                Sam3TrackerVideoProcessor,
            )
        except ImportError as import_error:
            raise MissingDependencyError(
                message=(
                    "Could not import Sam3TrackerVideoModel / "
                    "Sam3TrackerVideoProcessor from transformers.  Ensure a "
                    "transformers version that ships SAM3 video support is "
                    "installed."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/"
                    "runtime-environment/#missingdependencyerror"
                ),
            ) from import_error
        return Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
