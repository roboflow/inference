"""Video-enabled SAM3 for frame-by-frame tracking.

The native ``sam3`` package's ``build_sam3_video_predictor`` is session
based and expects a ``resource_path`` pointing to an MP4 file or JPEG
folder — it is not designed for true streaming from memory.  The HF
``transformers`` port (``Sam3VideoModel`` / ``Sam3VideoProcessor``)
exposes the underlying model's streaming interface via
``init_video_session`` + per-frame ``model(inference_session=..., frame=...)``,
which is what ``InferencePipeline`` needs.  This wrapper uses that path.

Each concurrently tracked video owns an ``inference_session`` that
carries SAM3's temporal memory.  Sessions are keyed by an opaque
``video_id``; callers are expected to reset a session when the
originating stream restarts.
"""

from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from inference.core.env import DEVICE

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


DEFAULT_SAM3_HF_ID = "facebook/sam3"


class _SAM3VideoSession:
    def __init__(self, inference_session: Any):
        self.inference_session = inference_session
        self.has_prompts = False


class SegmentAnything3Video:
    """Frame-by-frame SAM3 video tracker built on HF transformers.

    The model and processor are loaded once and shared across every
    video.  Per-video state lives in an ``inference_session`` from the
    processor.  Unlike ``SegmentAnything2Video`` this is not a
    ``RoboflowCoreModel`` — weights are pulled from the HF Hub because
    the SAM3 streaming path is only exposed through ``transformers``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_SAM3_HF_ID,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self.model_id = model_id
        self._device = torch.device(DEVICE)
        self._dtype = dtype
        self._model = Sam3VideoModel.from_pretrained(model_id).to(
            self._device, dtype=dtype
        )
        self._processor = Sam3VideoProcessor.from_pretrained(model_id)
        self._sessions: Dict[str, _SAM3VideoSession] = {}
        self._sessions_lock = RLock()
        self.task_type = "unsupervised-segmentation"

    def _init_session(self) -> _SAM3VideoSession:
        inference_session = self._processor.init_video_session(
            inference_device=self._device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self._dtype,
        )
        return _SAM3VideoSession(inference_session=inference_session)

    def reset_session(self, video_id: str) -> None:
        with self._sessions_lock:
            self._sessions.pop(video_id, None)

    def has_session(self, video_id: str) -> bool:
        with self._sessions_lock:
            session = self._sessions.get(video_id)
            return session is not None and session.has_prompts

    def _get_or_create_session(
        self, video_id: str, reset: bool
    ) -> _SAM3VideoSession:
        with self._sessions_lock:
            if reset:
                self._sessions.pop(video_id, None)
            session = self._sessions.get(video_id)
            if session is None:
                session = self._init_session()
                self._sessions[video_id] = session
            return session

    @torch.inference_mode()
    def prompt_and_track(
        self,
        video_id: str,
        frame: np.ndarray,
        frame_index: int,
        text: Optional[str] = None,
        boxes_xyxy: Optional[List[Tuple[float, float, float, float]]] = None,
        clear_old_prompts: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add prompts to the session and run one streaming step.

        Supply either ``text`` (open-vocabulary) or ``boxes_xyxy`` (visual
        prompts).  ``clear_old_prompts=True`` tears down the session so
        prompts start fresh on this frame — pass False to append new
        prompts to an ongoing session.

        Returns ``(masks, object_ids)`` with masks shape ``(N, H, W)``.
        """
        session = self._get_or_create_session(
            video_id=video_id, reset=clear_old_prompts
        )

        inputs = self._processor(
            images=frame, device=self._device, return_tensors="pt"
        )
        original_sizes = inputs.original_sizes

        if text is not None:
            session.inference_session = self._processor.add_text_prompt(
                inference_session=session.inference_session,
                text=text,
            )

        if boxes_xyxy:
            # Processors commonly accept nested list shape [[[[x1, y1, x2, y2], ...]]]
            formatted_boxes = [[[list(map(float, xyxy)) for xyxy in boxes_xyxy]]]
            self._processor.add_inputs_to_inference_session(
                inference_session=session.inference_session,
                frame_idx=frame_index,
                obj_ids=list(range(len(boxes_xyxy))),
                input_boxes=formatted_boxes,
                original_size=original_sizes[0],
            )

        session.has_prompts = True

        model_outputs = self._model(
            inference_session=session.inference_session,
            frame=inputs.pixel_values[0],
            reverse=False,
        )
        return self._extract_masks_and_ids(
            session=session,
            model_outputs=model_outputs,
            original_sizes=original_sizes,
        )

    @torch.inference_mode()
    def track(
        self,
        video_id: str,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run SAM3 on ``frame`` using the existing session's prompts."""
        with self._sessions_lock:
            session = self._sessions.get(video_id)
        if session is None or not session.has_prompts:
            raise RuntimeError(
                f"No SAM3 video session exists for video_id={video_id!r}; "
                "call prompt_and_track before tracking"
            )

        inputs = self._processor(
            images=frame, device=self._device, return_tensors="pt"
        )
        model_outputs = self._model(
            inference_session=session.inference_session,
            frame=inputs.pixel_values[0],
            reverse=False,
        )
        return self._extract_masks_and_ids(
            session=session,
            model_outputs=model_outputs,
            original_sizes=inputs.original_sizes,
        )

    def _extract_masks_and_ids(
        self,
        session: _SAM3VideoSession,
        model_outputs: Any,
        original_sizes: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Try the SAM3-specific post-processor first — it knows about
        # SAM3's tracking-head output layout and object id reordering.
        postprocess = getattr(self._processor, "postprocess_outputs", None)
        if callable(postprocess):
            processed = postprocess(
                session.inference_session,
                model_outputs,
                original_sizes=original_sizes,
            )
            masks, object_ids = _unpack_processed_outputs(processed)
            if masks is not None:
                return masks, object_ids

        # Fall back to the SAM2-style mask post-processing API.
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


def _unpack_processed_outputs(
    processed: Any,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Best-effort extraction of (masks, obj_ids) from a processor output.

    The SAM3 processor's ``postprocess_outputs`` return type varies
    between transformers versions: sometimes a dict, sometimes a list of
    per-object dicts.  We accept either shape and produce dense arrays.
    """
    if processed is None:
        return None, np.zeros((0,), dtype=np.int64)

    if isinstance(processed, dict):
        masks = processed.get("masks") or processed.get("pred_masks")
        ids = (
            processed.get("obj_ids")
            or processed.get("object_ids")
            or processed.get("track_ids")
        )
        masks_np = _to_numpy_binary_masks(masks) if masks is not None else None
        if masks_np is None:
            return None, np.zeros((0,), dtype=np.int64)
        if ids is None:
            ids = np.arange(masks_np.shape[0], dtype=np.int64)
        else:
            if hasattr(ids, "detach"):
                ids = ids.detach().cpu().numpy()
            ids = np.asarray(list(ids), dtype=np.int64)
        return masks_np, ids

    if isinstance(processed, (list, tuple)) and processed:
        mask_list = []
        id_list = []
        for idx, item in enumerate(processed):
            if isinstance(item, dict):
                m = item.get("mask") or item.get("masks") or item.get("segmentation")
                oid = (
                    item.get("obj_id")
                    or item.get("object_id")
                    or item.get("track_id")
                    or idx
                )
                if m is None:
                    continue
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
