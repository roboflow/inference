"""Video-enabled SAM3 for frame-by-frame tracking.

The native ``sam3`` package's ``build_sam3_video_predictor`` is session
based and expects a ``resource_path`` pointing to an MP4 file or JPEG
folder — it is not designed for true streaming from memory.  The HF
``transformers`` port (``Sam3VideoModel`` / ``Sam3VideoProcessor``)
exposes the underlying model's streaming interface via
``init_video_session`` + per-frame ``model(inference_session=..., frame=...)``,
which is what ``InferencePipeline`` needs.  This wrapper uses that path,
but pulls weights from the Roboflow model cache (not directly from HF
Hub) so deployments work without outbound access to huggingface.co.

Each concurrently tracked video owns an ``inference_session`` that
carries SAM3's temporal memory.  Sessions are keyed by an opaque
``video_id``; callers are expected to reset a session when the
originating stream restarts.
"""

import re
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from inference.core.env import DEVICE
from inference.models.transformers import TransformerModel

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


#: Roboflow registry id for the default SAM3 video checkpoint.  Mirrors
#: the ``sam3/sam3_final`` naming used by the image-path SAM3 model.
DEFAULT_SAM3_VIDEO_MODEL_ID = "sam3/sam3_video"


class _SAM3VideoSession:
    def __init__(self, inference_session: Any):
        self.inference_session = inference_session
        self.has_prompts = False


class SegmentAnything3Video(TransformerModel):
    """Frame-by-frame SAM3 video tracker built on HF transformers.

    Inherits from ``TransformerModel`` so weight management follows the
    standard Roboflow transformers pattern: weights are pulled into the
    local model cache directory and then passed to
    ``Sam3VideoModel.from_pretrained(self.cache_dir)``.

    The model and processor are loaded once and shared across every
    concurrently tracked video.  Per-video state lives in an
    ``inference_session`` returned by
    ``Sam3VideoProcessor.init_video_session``.
    """

    task_type = "unsupervised-segmentation"
    load_weights_as_transformers = True
    load_base_from_roboflow = True

    def __init__(
        self,
        *args,
        model_id: str = DEFAULT_SAM3_VIDEO_MODEL_ID,
        **kwargs,
    ):
        # Populate the lazy class-level attributes TransformerModel reads
        # from during ``initialize_model``.  Doing this here (rather than
        # as class statements) keeps the transformers import inside the
        # constructor so the module remains importable in environments
        # that only need the workflow schema.
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        if SegmentAnything3Video.transformers_class is None:
            SegmentAnything3Video.transformers_class = Sam3VideoModel
        if SegmentAnything3Video.processor_class is None:
            SegmentAnything3Video.processor_class = Sam3VideoProcessor
        if SegmentAnything3Video.default_dtype is None:
            SegmentAnything3Video.default_dtype = torch.bfloat16

        super().__init__(model_id, *args, **kwargs)

        self._sessions: Dict[str, _SAM3VideoSession] = {}
        self._sessions_lock = RLock()

    def initialize_model(self, **kwargs):
        """Load SAM3 video model and processor from the Roboflow cache.

        We override ``TransformerModel.initialize_model`` because SAM3
        video needs bfloat16 on GPU and does not accept the generic
        ``attn_implementation``/``device_map`` kwargs used by LMMs.
        """
        self.model = (
            self.transformers_class.from_pretrained(self.cache_dir)
            .eval()
            .to(DEVICE, dtype=self.dtype)
        )
        self.processor = self.processor_class.from_pretrained(self.cache_dir)

    def get_infer_bucket_file_list(self) -> list:
        """Files required from the Roboflow weights bucket.

        The exact file set for a transformers-format SAM3 video export is
        the usual HF model directory.  ``model*.safetensors`` is a regex
        so we accept either a single weights file or a sharded set.
        """
        return [
            "config.json",
            "preprocessor_config.json",
            re.compile(r"model.*\.safetensors"),
        ]

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _init_session(self) -> _SAM3VideoSession:
        inference_session = self.processor.init_video_session(
            inference_device=torch.device(DEVICE),
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
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

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

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
        """Add prompts to the session and run one streaming step."""
        session = self._get_or_create_session(
            video_id=video_id, reset=clear_old_prompts
        )

        inputs = self.processor(
            images=frame, device=DEVICE, return_tensors="pt"
        )
        original_sizes = inputs.original_sizes

        if text is not None:
            session.inference_session = self.processor.add_text_prompt(
                inference_session=session.inference_session,
                text=text,
            )

        if boxes_xyxy:
            formatted_boxes = [[[list(map(float, xyxy)) for xyxy in boxes_xyxy]]]
            self.processor.add_inputs_to_inference_session(
                inference_session=session.inference_session,
                frame_idx=frame_index,
                obj_ids=list(range(len(boxes_xyxy))),
                input_boxes=formatted_boxes,
                original_size=original_sizes[0],
            )

        session.has_prompts = True

        model_outputs = self.model(
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

        inputs = self.processor(
            images=frame, device=DEVICE, return_tensors="pt"
        )
        model_outputs = self.model(
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
        postprocess = getattr(self.processor, "postprocess_outputs", None)
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
        masks = self.processor.post_process_masks(
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

    ``Sam3VideoProcessor.postprocess_outputs`` return type varies across
    transformers versions: sometimes a dict, sometimes a list of
    per-object dicts.  Accept either shape and produce dense arrays.
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
        mask_list = []
        id_list = []
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


def _first_present(container: Any, keys: Tuple[str, ...]) -> Any:
    """Return the first value in ``container`` whose key is in ``keys``.

    ``dict.get(...) or dict.get(...)`` is unsafe here because the values
    may be numpy arrays or tensors whose truthiness raises.
    """
    for key in keys:
        if key in container and container[key] is not None:
            return container[key]
    return None
