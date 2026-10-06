"""Streaming SAM3 concept tracker backed by ``transformers.Sam3VideoModel``.

SAM3's video stack in transformers comes in two flavours:

- ``Sam3VideoModel`` / ``Sam3VideoProcessor`` — open-vocabulary
  *concept* tracking.  Text prompts are registered on the session once;
  every subsequent frame is a fused detect-and-track step, so objects
  matching a concept are picked up automatically mid-stream and keep
  stable ids.  This is what this class wraps.
- ``Sam3TrackerVideoModel`` / ``Sam3TrackerVideoProcessor`` — the
  SAM2-style visually prompted (box / point) tracker.  Not wrapped
  here; use ``sam2video`` for detector-driven tracking.

The native ``sam3`` package's video predictor is session based and
requires a pre-existing MP4 / JPEG directory — it cannot consume frames
one at a time from memory.  The transformers port exposes the streaming
interface (``init_video_session`` + per-frame
``model(inference_session=..., frame=...)``) that
``InferencePipeline``-style processing needs.

API shape
---------
Unlike the SAM2-shaped ``HFStreamingVideoBase`` contract (where prompts
seed specific frames and ``track`` only ever propagates), SAM3 concept
prompts live on the session and *every* frame may both detect new
objects and propagate existing ones.  Per-frame results therefore carry
detection scores and the prompt→object-ids mapping so callers can label
and threshold each mask:

.. code-block:: python

    result = model.prompt(image=frame0, text=["person", "dog"])
    result = model.track(image=frame1, state_dict=result.state_dict)
    result.masks                 # (N, H, W) bool
    result.object_ids            # (N,) int64 — stable across frames
    result.scores                # (N,) float32 detection scores
    result.prompt_to_object_ids  # {"person": [0, 2], "dog": [1]}

The ``state_dict`` holds a live HF inference session with GPU tensor
references — it is **not serialisable across processes**.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from inference_models.errors import MissingDependencyError, ModelRuntimeError
from inference_models.models.common.hf_streaming_video import (
    SESSION_KEY,
    HFVideoModelBase,
    _ensure_numpy_image,
)

_HELP_URL = (
    "https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror"
)


@dataclass(frozen=True)
class SAM3VideoFrameResult:
    """Outputs of one streaming step of the SAM3 concept tracker."""

    #: ``(N, H, W)`` binary masks at the input frame's resolution.
    masks: np.ndarray
    #: ``(N,)`` int64 object ids, stable across the session's frames.
    object_ids: np.ndarray
    #: ``(N,)`` float32 per-object detection scores.
    scores: np.ndarray
    #: ``(N, 4)`` float32 boxes in xyxy, derived from the masks.
    boxes: np.ndarray
    #: Maps each prompt text to the object ids it currently claims.
    prompt_to_object_ids: Dict[str, List[int]]
    #: Opaque session handle — thread back into the next call.
    state_dict: dict


class SAM3Video(HFVideoModelBase):
    """SAM3 streaming concept tracker via the transformers port."""

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

    @torch.inference_mode()
    def prompt(
        self,
        image: Union[np.ndarray, torch.Tensor],
        text: Union[str, Sequence[str]],
        state_dict: Optional[dict] = None,
        clear_old_prompts: bool = True,
    ) -> SAM3VideoFrameResult:
        """Register concept prompts and run one streaming step.

        Each entry in ``text`` is an independent concept; matching
        objects are reported under that entry in
        ``prompt_to_object_ids``.  ``clear_old_prompts=True`` starts a
        fresh session; pass ``False`` along with ``state_dict`` to add
        concepts to an ongoing session.
        """
        texts = _normalise_text_prompts(text)
        if not texts:
            raise ModelRuntimeError(
                message=(
                    "SAM3Video.prompt requires at least one non-empty text "
                    "prompt. SAM3's concept tracker is text-prompted; for "
                    "box-prompted video tracking use the `sam2video` model."
                ),
                help_url=_HELP_URL,
            )
        with self._lock:
            image_np = _ensure_numpy_image(image)
            session = self._resolve_session(
                state_dict=state_dict, reset=clear_old_prompts
            )
            session = self._processor.add_text_prompt(
                inference_session=session,
                text=texts,
            )
            return self._process_frame(session=session, image_np=image_np)

    @torch.inference_mode()
    def track(
        self,
        image: Union[np.ndarray, torch.Tensor],
        state_dict: Optional[dict] = None,
    ) -> SAM3VideoFrameResult:
        """Run one streaming step on an already prompted session.

        Detection runs continuously: objects entering the scene that
        match a registered concept appear with fresh ids — there is no
        need to re-prompt.
        """
        with self._lock:
            session = state_dict.get(SESSION_KEY) if state_dict is not None else None
            if session is None:
                raise ModelRuntimeError(
                    message=(
                        "Attempt to track with no prior call to prompt; "
                        "prompt must be called first (pass the state_dict "
                        "returned from prompt back into track)"
                    ),
                    help_url=_HELP_URL,
                )
            image_np = _ensure_numpy_image(image)
            return self._process_frame(session=session, image_np=image_np)

    def _process_frame(
        self, session: Any, image_np: np.ndarray
    ) -> SAM3VideoFrameResult:
        inputs = self._processor(
            images=image_np, device=self._device, return_tensors="pt"
        )
        model_outputs = self._model(
            inference_session=session,
            frame=inputs.pixel_values[0],
        )
        processed = self._processor.postprocess_outputs(
            session,
            model_outputs,
            original_sizes=inputs.original_sizes,
        )
        return SAM3VideoFrameResult(
            masks=_to_numpy(processed["masks"]).astype(bool),
            object_ids=_to_numpy(processed["object_ids"]).astype(np.int64),
            scores=_to_numpy(processed["scores"]).astype(np.float32),
            boxes=_to_numpy(processed["boxes"]).astype(np.float32),
            prompt_to_object_ids={
                str(prompt): [int(obj_id) for obj_id in obj_ids]
                for prompt, obj_ids in processed["prompt_to_obj_ids"].items()
            },
            state_dict={SESSION_KEY: session},
        )


def _normalise_text_prompts(text: Union[str, Sequence[str], None]) -> List[str]:
    if text is None:
        return []
    if isinstance(text, str):
        candidates = [text]
    else:
        candidates = list(text)
    return [str(t).strip() for t in candidates if t is not None and str(t).strip()]


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
