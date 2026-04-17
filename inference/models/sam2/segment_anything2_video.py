"""Video-enabled SAM2 for frame-by-frame tracking.

Wraps ``SAM2CameraPredictor`` from the sam2 library to expose a per-video
session interface suitable for streaming use (one frame at a time, as
produced by ``InferencePipeline``).  A SAM2 camera predictor is inherently
stateful — it ingests a first frame plus prompts and then propagates masks
forward as new frames arrive.  Each concurrently tracked video needs its
own predictor instance, so this class manages a dict of sessions keyed by
an opaque ``video_id``.
"""

from threading import RLock
from typing import Dict, List, Optional, Tuple

import numpy as np
import sam2.utils.misc
import torch
from torch.nn.attention import SDPBackend

sam2.utils.misc.get_sdp_backends = lambda z: [
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]
from sam2.build_sam import build_sam2_camera_predictor
from sam2.sam2_camera_predictor import SAM2CameraPredictor

from inference.core.env import DEVICE, SAM2_VERSION_ID
from inference.core.models.roboflow import RoboflowCoreModel

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


_MODEL_CFG_BY_VERSION: Dict[str, str] = {
    "hiera_large": "sam2_hiera_l.yaml",
    "hiera_small": "sam2_hiera_s.yaml",
    "hiera_tiny": "sam2_hiera_t.yaml",
    "hiera_b_plus": "sam2_hiera_b+.yaml",
}


class _SAM2VideoSession:
    """Holds one SAM2 camera predictor and associated tracking state.

    ``next_obj_id`` is monotonically increasing across the session so that
    re-prompting mid-stream does not collide with ids of earlier prompts.
    """

    def __init__(self, predictor: SAM2CameraPredictor):
        self.predictor = predictor
        self.next_obj_id = 0
        self.first_frame_loaded = False


class SegmentAnything2Video(RoboflowCoreModel):
    """Frame-by-frame SAM2 video tracker.

    One instance of this class is loaded once (weights in GPU memory) and
    multiplexed across many concurrent video streams via ``video_id``.
    Each ``video_id`` owns its own ``SAM2CameraPredictor`` — SAM2's camera
    predictor keeps its rolling temporal memory inside the predictor
    object itself, so separate videos cannot share one predictor.
    """

    def __init__(
        self,
        *args,
        model_id: str = f"sam2/{SAM2_VERSION_ID}",
        **kwargs,
    ):
        super().__init__(*args, model_id=model_id, **kwargs)
        self._checkpoint_path = self.cache_file("weights.pt")
        if self.version_id not in _MODEL_CFG_BY_VERSION:
            raise ValueError(
                f"Unknown SAM2 version_id '{self.version_id}'. "
                f"Expected one of {list(_MODEL_CFG_BY_VERSION)}."
            )
        self._model_cfg = _MODEL_CFG_BY_VERSION[self.version_id]
        self._sessions: Dict[str, _SAM2VideoSession] = {}
        self._sessions_lock = RLock()
        self.task_type = "unsupervised-segmentation"

    def get_infer_bucket_file_list(self) -> List[str]:
        return ["weights.pt"]

    def _build_predictor(self) -> SAM2CameraPredictor:
        return build_sam2_camera_predictor(
            config_file=self._model_cfg,
            ckpt_path=self._checkpoint_path,
            device=DEVICE,
        )

    def reset_session(self, video_id: str) -> None:
        """Drop all state for ``video_id``.

        Call on stream restart (frame_number returns to 0) or when a video
        finishes.  Frees the associated predictor so GPU memory is
        reclaimed for new streams.
        """
        with self._sessions_lock:
            self._sessions.pop(video_id, None)

    def _get_or_create_session(self, video_id: str) -> _SAM2VideoSession:
        with self._sessions_lock:
            session = self._sessions.get(video_id)
            if session is None:
                session = _SAM2VideoSession(predictor=self._build_predictor())
                self._sessions[video_id] = session
            return session

    @torch.inference_mode()
    def prompt_and_track(
        self,
        video_id: str,
        frame: np.ndarray,
        boxes_xyxy: Optional[List[Tuple[float, float, float, float]]] = None,
        clear_old_prompts: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Seed the tracker for ``video_id`` and segment the current frame.

        Treats ``frame`` as the reference frame for the supplied prompts.
        Returns ``(masks, object_ids)`` where masks has shape ``(N, H, W)``
        of boolean values and ``object_ids`` is a 1D int array of the same
        length.  If ``clear_old_prompts`` is True any existing tracks on
        this session are replaced; otherwise new object ids are appended.
        """
        if boxes_xyxy is None:
            boxes_xyxy = []

        session = self._get_or_create_session(video_id)
        predictor = session.predictor

        if clear_old_prompts or not session.first_frame_loaded:
            predictor.load_first_frame(frame)
            session.first_frame_loaded = True
            session.next_obj_id = 0

        masks_out: Optional[np.ndarray] = None
        object_ids_out: List[int] = []
        for xyxy in boxes_xyxy:
            x1, y1, x2, y2 = xyxy
            x_lt = int(round(min(x1, x2)))
            y_lt = int(round(min(y1, y2)))
            x_rb = int(round(max(x1, x2)))
            y_rb = int(round(max(y1, y2)))
            bbox = np.array([[x_lt, y_lt, x_rb, y_rb]])

            obj_id = session.next_obj_id
            session.next_obj_id += 1
            _, object_ids, mask_logits = predictor.add_new_prompt(
                frame_idx=0,
                obj_id=obj_id,
                bbox=bbox,
                clear_old_points=clear_old_prompts,
                normalize_coords=True,
            )
            masks_out = mask_logits
            object_ids_out = list(object_ids)

        if masks_out is None:
            # No boxes prompted — still return an empty result of the right
            # shape so callers can consume uniformly.
            h, w = frame.shape[:2]
            return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.int64)

        masks_np = (masks_out > 0.0).detach().cpu().numpy()
        masks_np = np.squeeze(masks_np, axis=1) if masks_np.ndim == 4 else masks_np
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        return masks_np.astype(bool), np.asarray(object_ids_out, dtype=np.int64)

    @torch.inference_mode()
    def track(
        self,
        video_id: str,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate existing tracks onto ``frame`` — no new prompts added.

        Raises ``RuntimeError`` if no session exists for ``video_id`` yet
        (caller must prompt first).
        """
        with self._sessions_lock:
            session = self._sessions.get(video_id)
        if session is None or not session.first_frame_loaded:
            raise RuntimeError(
                f"No SAM2 video session exists for video_id={video_id!r}; "
                "call prompt_and_track before tracking"
            )

        object_ids, mask_logits = session.predictor.track(frame)
        masks_np = (mask_logits > 0.0).detach().cpu().numpy()
        masks_np = np.squeeze(masks_np, axis=1) if masks_np.ndim == 4 else masks_np
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        return masks_np.astype(bool), np.asarray(list(object_ids), dtype=np.int64)

    def has_session(self, video_id: str) -> bool:
        with self._sessions_lock:
            session = self._sessions.get(video_id)
            return session is not None and session.first_frame_loaded
