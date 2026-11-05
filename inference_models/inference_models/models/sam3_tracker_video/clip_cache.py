"""Per-clip state shared by every offline tracking session on a clip.

An offline (whole-clip) session runs on a clip's preprocessed frames and
reuses the image-encoder output of every frame it has already seen. Both
depend only on the frames, not on the prompts, so they are kept per clip and
shared by all sessions on it instead of being rebuilt per request.
"""

import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch

from inference_models.errors import EnvironmentConfigurationError, ModelInputError


class VideoClip:
    """A clip's preprocessed frames and the encoder output of the frames seen so far.

    Instances stand in for the HF session's ``Sam3TrackerVideoInferenceCache``
    (which holds one frame and lives only as long as its session): sessions
    get it as their ``cache`` and the model reads and writes vision features
    through the same three methods. Features are kept on the CPU and moved to
    the inference device per use; the position embeddings are the same for
    every frame, so one copy is kept.

    Args:
        pixel_values: Preprocessed frames, ``(n_frames, 3, H, W)``.
        video_size: ``(height, width)`` of the original frames.
        inference_device: Device the model runs on.
    """

    def __init__(
        self,
        pixel_values: torch.Tensor,
        *,
        video_size: Tuple[int, int],
        inference_device: torch.device,
    ):
        self.pixel_values = pixel_values
        self.video_size = video_size
        self.inference_device = inference_device
        self._feats: Dict[int, List[torch.Tensor]] = {}
        self._pos_embeds: Optional[List[torch.Tensor]] = None

    @property
    def n_frames(self) -> int:
        """Number of frames in the clip."""
        return len(self.pixel_values)

    def has_features(self, frame_idx: int) -> bool:
        """Whether the encoder output of ``frame_idx`` is cached.

        Args:
            frame_idx: Frame index in the clip.

        Returns:
            True once a session has encoded the frame.
        """
        return frame_idx in self._feats

    def cache_vision_features(self, frame_idx: int, features: dict) -> None:
        """Keep the encoder output of one frame (called by the HF model).

        Args:
            frame_idx: Frame index in the clip.
            features: ``vision_feats`` and ``vision_pos_embeds`` of the frame.

        Raises:
            ModelInputError: If the frame's position embeddings differ from the
                clip's, which would make sharing them wrong.
        """
        pos = features["vision_pos_embeds"]
        if self._pos_embeds is None:
            self._pos_embeds = [p.to("cpu") for p in pos]
        elif not all(
            torch.equal(a, b.to("cpu")) for a, b in zip(self._pos_embeds, pos)
        ):
            raise ModelInputError(
                message=f"Position embeddings differ on frame {frame_idx}; "
                "frames of one clip must share them."
            )
        self._feats[frame_idx] = [f.to("cpu") for f in features["vision_feats"]]

    def get_vision_features(self, frame_idx: int) -> Optional[dict]:
        """Return the cached encoder output of one frame (called by the HF model).

        Args:
            frame_idx: Frame index in the clip.

        Returns:
            The frame's features on the inference device, or None when the
            frame has not been encoded yet.
        """
        if frame_idx not in self._feats:
            return None

        features = {
            "vision_feats": [
                f.to(self.inference_device, non_blocking=True)
                for f in self._feats[frame_idx]
            ],
            "vision_pos_embeds": [
                p.to(self.inference_device, non_blocking=True) for p in self._pos_embeds
            ],
        }

        return features

    def clear_all(self) -> None:
        """Drop the cached encoder output (called by the HF model on reset)."""
        self._feats.clear()
        self._pos_embeds = None


class VideoClipCache:
    """Clips by id, oldest evicted first once ``size_limit`` is reached.

    Args:
        size_limit: Most clips kept at once; None for no limit.
    """

    def __init__(self, size_limit: Optional[int]):
        if size_limit is not None and size_limit < 1:
            raise EnvironmentConfigurationError(
                message=f"SAM3 video clip cache size must be at least 1, got {size_limit}."
            )

        self._size_limit = size_limit
        self._clips: "OrderedDict[str, VideoClip]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, video_id: str, clip: VideoClip) -> None:
        """Store ``clip`` under ``video_id``, replacing any clip already there.

        Args:
            video_id: Caller's id for the clip.
            clip: The clip.
        """
        with self._lock:
            self._clips.pop(video_id, None)
            while self._size_limit is not None and len(self._clips) >= self._size_limit:
                self._clips.popitem(last=False)
            self._clips[video_id] = clip

    def holds(self, video_id: str, clip: VideoClip) -> bool:
        """Whether ``clip`` is still the clip stored under ``video_id``.

        Args:
            video_id: Caller's id for the clip.
            clip: The clip.

        Returns:
            False once the clip was replaced or evicted.
        """
        with self._lock:
            held = self._clips.get(video_id) is clip

        return held

    def get(self, video_id: str) -> VideoClip:
        """Return the clip stored under ``video_id``.

        Args:
            video_id: Caller's id for the clip.

        Returns:
            The clip.

        Raises:
            ModelInputError: If no clip is stored under ``video_id`` (never
                embedded, or evicted since).
        """
        with self._lock:
            clip = self._clips.get(video_id)
            if clip is None:
                raise ModelInputError(
                    message=f"No video embedded under id {video_id!r}; embed it "
                    "first (it may have been evicted)."
                )
            self._clips.move_to_end(video_id)

        return clip
