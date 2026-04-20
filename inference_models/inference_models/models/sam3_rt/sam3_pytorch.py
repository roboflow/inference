"""Streaming SAM3 tracker backed by the HuggingFace transformers port.

``SAM3ForStream`` mirrors ``SAM2ForStream`` (see
``inference_models.models.sam2_rt.sam2_pytorch``): it exposes ``prompt``
and ``track`` methods that return ``(masks, object_ids, state_dict)`` so
the caller can ferry state across frames.

Why transformers?  The native ``sam3`` package's video predictor is
session based and requires a pre-existing MP4 / JPEG directory supplied
via ``start_session(resource_path=...)``; it isn't designed for
frame-by-frame streaming from memory.  HuggingFace's ``Sam3VideoModel``
exposes the underlying model's streaming interface through
``init_video_session`` + per-frame ``model(inference_session=..., frame=...)``,
which is what we need.

``state_dict`` contract
------------------------
Unlike SAM2's PyTorch state dict (a serializable ``{str: Tensor}`` map),
SAM3's streaming state is an opaque HuggingFace ``Sam3VideoInferenceSession``
object that holds GPU tensors.  We wrap it in a ``dict`` so callers still
get a dict back — but the dict is **not serializable across processes**;
it's a live handle that must be kept in memory by the caller.
"""

from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import MissingDependencyError, ModelRuntimeError

try:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
except ImportError as import_error:
    raise MissingDependencyError(
        message=(
            "Could not import Sam3VideoModel / Sam3VideoProcessor from "
            "transformers.  Ensure a transformers version that ships SAM3 "
            "video support is installed."
        ),
        help_url=(
            "https://inference-models.roboflow.com/errors/runtime-environment/"
            "#missingdependencyerror"
        ),
    ) from import_error


#: Key under which the wrapped HF inference session lives inside the
#: opaque ``state_dict`` this class returns.
_SESSION_KEY = "_sam3_video_inference_session"


class SAM3ForStream:
    """Frame-by-frame SAM3 video tracker.

    One instance holds the model weights on GPU.  Per-video state lives
    inside the ``state_dict`` returned from ``prompt`` / ``track`` —
    callers are expected to keep it in memory and pass it back on the
    next call to continue tracking the same video.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "SAM3ForStream":
        """Load SAM3 video model + processor from a package directory.

        ``model_name_or_path`` must be a directory laid out like a
        standard HuggingFace transformers export (``config.json``,
        ``preprocessor_config.json``, ``model*.safetensors`` and any
        tokenizer / video-processor files the exported checkpoint
        relies on).  ``Sam3VideoProcessor.from_pretrained`` with
        ``local_files_only=True`` will locate everything by itself.
        """
        model = (
            Sam3VideoModel.from_pretrained(
                model_name_or_path,
                local_files_only=True,
            )
            .eval()
            .to(device, dtype=dtype)
        )
        processor = Sam3VideoProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=True,
        )
        return cls(model=model, processor=processor, device=device, dtype=dtype)

    def __init__(
        self,
        model: Sam3VideoModel,
        processor: Sam3VideoProcessor,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def prompt(
        self,
        image: Union[np.ndarray, torch.Tensor],
        bboxes: Optional[
            Union[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]]
        ] = None,
        text: Optional[str] = None,
        state_dict: Optional[dict] = None,
        clear_old_prompts: bool = True,
        frame_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Seed a SAM3 session and run one streaming step.

        Supply ``bboxes`` (visual prompts) or ``text`` (open-vocabulary)
        or both.  ``clear_old_prompts=True`` drops any existing session
        and starts fresh; pass ``False`` along with ``state_dict`` to
        append prompts to an ongoing session.
        """
        with self._lock:
            image_np = _ensure_numpy_image(image)
            session = self._resolve_session(
                state_dict=state_dict, reset=clear_old_prompts
            )

            inputs = self._processor(
                images=image_np, device=self._device, return_tensors="pt"
            )
            original_sizes = inputs.original_sizes

            if text is not None:
                session = self._processor.add_text_prompt(
                    inference_session=session,
                    text=text,
                )

            box_list = _normalise_bboxes(bboxes)
            if box_list:
                self._processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=frame_idx,
                    obj_ids=list(range(len(box_list))),
                    input_boxes=[[[[float(v) for v in xyxy] for xyxy in box_list]]],
                    original_size=original_sizes[0],
                )

            model_outputs = self._model(
                inference_session=session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )
            masks, object_ids = self._extract_masks_and_ids(
                session=session,
                model_outputs=model_outputs,
                original_sizes=original_sizes,
            )
            return masks, object_ids, {_SESSION_KEY: session}

    @torch.inference_mode()
    def track(
        self,
        image: Union[np.ndarray, torch.Tensor],
        state_dict: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Propagate existing tracks onto ``image``.

        Must be called after ``prompt`` (either in the same invocation
        chain or by threading the ``state_dict`` returned from a prior
        call).
        """
        with self._lock:
            image_np = _ensure_numpy_image(image)
            session = (
                state_dict.get(_SESSION_KEY) if state_dict is not None else None
            )
            if session is None:
                raise ModelRuntimeError(
                    message=(
                        "Attempt to track with no prior call to prompt; "
                        "prompt must be called first (pass the state_dict "
                        "returned from prompt back into track)"
                    ),
                    help_url=(
                        "https://inference-models.roboflow.com/errors/"
                        "models-runtime/#modelruntimeerror"
                    ),
                )

            inputs = self._processor(
                images=image_np, device=self._device, return_tensors="pt"
            )
            model_outputs = self._model(
                inference_session=session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )
            masks, object_ids = self._extract_masks_and_ids(
                session=session,
                model_outputs=model_outputs,
                original_sizes=inputs.original_sizes,
            )
            return masks, object_ids, {_SESSION_KEY: session}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_session(
        self, state_dict: Optional[dict], reset: bool
    ) -> Any:
        """Return a HF inference session for ``prompt``.

        ``reset=True`` (the default for a fresh prompt) always returns a
        new session.  ``reset=False`` reuses the session embedded in
        ``state_dict`` if present, otherwise starts a new one.
        """
        if reset:
            return self._new_session()
        if state_dict is not None:
            session = state_dict.get(_SESSION_KEY)
            if session is not None:
                return session
        return self._new_session()

    def _new_session(self) -> Any:
        return self._processor.init_video_session(
            inference_device=self._device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self._dtype,
        )

    def _extract_masks_and_ids(
        self,
        session: Any,
        model_outputs: Any,
        original_sizes: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        postprocess = getattr(self._processor, "postprocess_outputs", None)
        if callable(postprocess):
            processed = postprocess(
                session,
                model_outputs,
                original_sizes=original_sizes,
            )
            masks, object_ids = _unpack_processed_outputs(processed)
            if masks is not None:
                return masks, object_ids

        # Fallback to the SAM2-style mask post-processing API.
        pred_masks = getattr(model_outputs, "pred_masks", None)
        if pred_masks is None:
            return np.zeros((0, 0, 0), dtype=bool), np.zeros((0,), dtype=np.int64)
        masks = self._processor.post_process_masks(
            [pred_masks],
            original_sizes=original_sizes,
            binarize=True,
        )[0]
        masks_np = _to_numpy_binary_masks(masks)
        object_ids = _extract_object_ids(model_outputs, n=masks_np.shape[0])
        return masks_np, object_ids


# ---------------------------------------------------------------------------
# Module-level helpers (tested in isolation)
# ---------------------------------------------------------------------------


def _ensure_numpy_image(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        return image.detach().cpu().numpy()
    return image


def _normalise_bboxes(
    bboxes: Optional[
        Union[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]]
    ],
) -> List[Tuple[float, float, float, float]]:
    """Accept a single xyxy tuple or a list of them and return a flat list.

    Empty / missing input returns ``[]``; malformed entries (< 4 coords)
    are dropped.
    """
    if bboxes is None:
        return []
    if not isinstance(bboxes, list):
        bboxes = [bboxes]
    out: List[Tuple[float, float, float, float]] = []
    for box in bboxes:
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        x_lt = float(min(x1, x2))
        y_lt = float(min(y1, y2))
        x_rb = float(max(x1, x2))
        y_rb = float(max(y1, y2))
        out.append((x_lt, y_lt, x_rb, y_rb))
    return out


def _to_numpy_binary_masks(masks: Any) -> np.ndarray:
    if masks is None:
        return np.zeros((0, 0, 0), dtype=bool)
    if hasattr(masks, "detach"):
        arr = masks.detach().cpu().numpy()
    else:
        arr = np.asarray(masks)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr.astype(bool)


def _extract_object_ids(model_outputs: Any, n: int) -> np.ndarray:
    for attr in ("obj_ids", "object_ids"):
        ids = getattr(model_outputs, attr, None)
        if ids is not None:
            if hasattr(ids, "detach"):
                ids = ids.detach().cpu().numpy()
            return np.asarray(list(ids), dtype=np.int64)
    return np.arange(n, dtype=np.int64)


def _first_present(container: Any, keys: Tuple[str, ...]) -> Any:
    """Return the first value in ``container`` whose key is in ``keys``.

    Needed because ``dict.get(a) or dict.get(b)`` raises on numpy /
    tensor values (ambiguous truthiness).
    """
    for key in keys:
        if key in container and container[key] is not None:
            return container[key]
    return None


def _unpack_processed_outputs(
    processed: Any,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Best-effort extraction of ``(masks, obj_ids)`` from the processor's
    ``postprocess_outputs`` return value, which varies between
    transformers versions.
    """
    if processed is None:
        return None, np.zeros((0,), dtype=np.int64)

    if isinstance(processed, dict):
        masks = _first_present(processed, ("masks", "pred_masks"))
        ids = _first_present(processed, ("obj_ids", "object_ids", "track_ids"))
        if masks is None:
            return None, np.zeros((0,), dtype=np.int64)
        masks_np = _to_numpy_binary_masks(masks)
        if ids is None:
            ids = np.arange(masks_np.shape[0], dtype=np.int64)
        else:
            if hasattr(ids, "detach"):
                ids = ids.detach().cpu().numpy()
            ids = np.asarray(list(ids), dtype=np.int64)
        return masks_np, ids

    if isinstance(processed, (list, tuple)) and processed:
        mask_list: List[np.ndarray] = []
        id_list: List[int] = []
        for idx, item in enumerate(processed):
            if not isinstance(item, dict):
                continue
            m = _first_present(item, ("mask", "masks", "segmentation"))
            if m is None:
                continue
            oid = _first_present(item, ("obj_id", "object_id", "track_id"))
            if oid is None:
                oid = idx
            m_np = _to_numpy_binary_masks(m)
            if m_np.ndim == 3 and m_np.shape[0] == 1:
                mask_list.append(m_np[0])
            elif m_np.ndim == 2:
                mask_list.append(m_np)
            else:
                for row in m_np:
                    mask_list.append(row)
            id_list.append(int(oid))
        if mask_list:
            return (
                np.stack(mask_list, axis=0).astype(bool),
                np.asarray(id_list, dtype=np.int64),
            )

    return None, np.zeros((0,), dtype=np.int64)
