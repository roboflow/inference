"""Per-clip state shared by every offline tracking session on a clip.

An offline (whole-clip) session runs on a clip's preprocessed frames and
reuses the image-encoder output of every frame it has already seen. Both
depend only on the frames, not on the prompts, so they are kept per clip and
shared by all sessions on it instead of being rebuilt per request.

A frame is held either as its preprocessed pixels or, once encoded, as its
encoder output: the pixels are released the moment the output is cached, and
the output is then kept for as long as the clip is. The cache is sized in
frames, from a limit in GB of host RAM and the bytes one frame can hold.
"""

import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

from inference_models.errors import (
    EnvironmentConfigurationError,
    ModelInputError,
    ModelRuntimeError,
)

CACHE_LIMIT_SETTING = "SAM3_VIDEO_CACHE_MAX_GB"


def cached_frame_bytes(model_config: Any, dtype: torch.dtype) -> int:
    """Most bytes one cached frame holds, from the model's shapes and dtype.

    A frame holds its preprocessed pixels until it is encoded and its encoder
    output afterwards, so the larger of the two. The encoder output is what
    ``Sam3TrackerVideoModel.get_image_features`` returns as
    ``fpn_hidden_states``: one map per backbone feature size, the first two
    projected by the mask decoder to an eighth and a quarter of its width.

    Args:
        model_config: The ``Sam3TrackerVideoConfig`` of the loaded model.
        dtype: Dtype the model runs in; pixels and encoder output are kept in it.

    Returns:
        Bytes of the larger of one frame's pixels and its encoder output.
    """
    item_bytes = torch.empty((), dtype=dtype).element_size()
    pixel_values = 3 * model_config.image_size**2
    feature_sizes = model_config.vision_config.backbone_feature_sizes
    decoder_width = model_config.mask_decoder_config.hidden_size
    widths = [decoder_width // 8, decoder_width // 4] + [
        model_config.vision_config.fpn_hidden_size
    ] * (len(feature_sizes) - 2)
    feature_values = sum(
        height * width * channels
        for (height, width), channels in zip(feature_sizes, widths)
    )
    frame_bytes = max(pixel_values, feature_values) * item_bytes

    return frame_bytes


def cache_frame_limit(max_gb: float, *, frame_bytes: int) -> int:
    """Frames a cache of ``max_gb`` GB holds.

    Args:
        max_gb: Limit on the cached frames, in GB (1e9 bytes) of host RAM.
        frame_bytes: Most bytes one cached frame holds.

    Returns:
        ``floor(max_gb * 1e9 / frame_bytes)``.

    Raises:
        EnvironmentConfigurationError: If ``max_gb`` holds less than one frame.
    """
    max_frames = int(max_gb * 1e9 // frame_bytes)
    if max_frames < 1:
        raise EnvironmentConfigurationError(
            message=f"{CACHE_LIMIT_SETTING}={max_gb} holds no frame: one cached "
            f"frame takes {frame_bytes} bytes, so it must be at least "
            f"{frame_bytes / 1e9:.4f}."
        )

    return max_frames


class PositionEmbeddings:
    """The encoder's position embeddings, one copy for every clip of a model.

    They depend only on the model's feature map sizes, not on the frame, so
    every frame of every clip must produce the same ones; a frame that does
    not is refused.
    """

    def __init__(self):
        self._embeds: Optional[List[torch.Tensor]] = None

    def keep(self, frame_idx: int, embeds: List[torch.Tensor]) -> None:
        """Keep the position embeddings of a frame, or check them against the kept ones.

        Args:
            frame_idx: Frame they came from, for the message.
            embeds: The frame's position embeddings.

        Raises:
            ModelInputError: If they differ from the ones kept, which would
                make sharing them wrong.
        """
        if self._embeds is None:
            self._embeds = [p.to("cpu") for p in embeds]
        elif not all(torch.equal(a, b.to("cpu")) for a, b in zip(self._embeds, embeds)):
            raise ModelInputError(
                message=f"Position embeddings differ on frame {frame_idx}; "
                "every frame must share them."
            )

    def on(self, device: torch.device) -> List[torch.Tensor]:
        """The kept position embeddings on ``device``.

        Args:
            device: Device to move them to.

        Returns:
            The position embeddings.
        """
        embeds = [p.to(device, non_blocking=True) for p in self._embeds]

        return embeds


class ClipFrames:
    """A clip's preprocessed frames, each its own tensor so each can be released.

    Stands in for the HF session's ``processed_frames`` (a dict of frames by
    index, of which the model uses ``len()`` and ``[frame_idx]``).

    Args:
        frames: Preprocessed frames ``(3, H, W)``, one tensor (and one
            allocation) per frame.
    """

    def __init__(self, frames: List[torch.Tensor]):
        self._frames: List[Optional[torch.Tensor]] = list(frames)

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, frame_idx: int) -> torch.Tensor:
        frame = self._frames[frame_idx]
        if frame is None:
            raise ModelRuntimeError(
                message=f"The pixels of frame {frame_idx} were released when its "
                "encoder output was cached, so the frame cannot be encoded again."
            )

        return frame

    def holds(self, frame_idx: int) -> bool:
        """Whether the pixels of ``frame_idx`` are still held.

        Args:
            frame_idx: Frame index in the clip.

        Returns:
            False once the frame was released.
        """
        return self._frames[frame_idx] is not None

    def release(self, frame_idx: int) -> None:
        """Drop the pixels of ``frame_idx``.

        Args:
            frame_idx: Frame index in the clip.
        """
        self._frames[frame_idx] = None


class VideoClip:
    """A clip's preprocessed frames and the encoder output of the frames seen so far.

    Instances stand in for the HF session's ``Sam3TrackerVideoInferenceCache``
    (which holds one frame and lives only as long as its session): sessions
    get it as their ``cache`` and the model reads and writes vision features
    through the same three methods. Features are kept on the CPU and moved to
    the inference device per use.

    A frame's pixels are needed only to encode it, so they are released when
    its encoder output is cached. The output is then never dropped while the
    clip lives (``clear_all`` refuses), so a released frame never has to be
    encoded again.

    Args:
        frames: Preprocessed frames ``(3, H, W)``, one tensor per frame.
        video_size: ``(height, width)`` of the original frames.
        inference_device: Device the model runs on.
        position_embeddings: The model's shared position embeddings.
    """

    def __init__(
        self,
        frames: List[torch.Tensor],
        *,
        video_size: Tuple[int, int],
        inference_device: torch.device,
        position_embeddings: PositionEmbeddings,
    ):
        self.frames = ClipFrames(frames)
        self.video_size = video_size
        self.inference_device = inference_device
        self._position_embeddings = position_embeddings
        self._feats: Dict[int, List[torch.Tensor]] = {}

    @property
    def n_frames(self) -> int:
        """Number of frames in the clip."""
        return len(self.frames)

    def has_features(self, frame_idx: int) -> bool:
        """Whether the encoder output of ``frame_idx`` is cached.

        Args:
            frame_idx: Frame index in the clip.

        Returns:
            True once a session has encoded the frame.
        """
        return frame_idx in self._feats

    def cache_vision_features(self, frame_idx: int, features: dict) -> None:
        """Keep the encoder output of one frame and release its pixels
        (called by the HF model).

        Args:
            frame_idx: Frame index in the clip.
            features: ``vision_feats`` and ``vision_pos_embeds`` of the frame.

        Raises:
            ModelInputError: If the frame's position embeddings differ from the
                model's, which would make sharing them wrong.
        """
        self._position_embeddings.keep(frame_idx, features["vision_pos_embeds"])
        self._feats[frame_idx] = [f.to("cpu") for f in features["vision_feats"]]
        self.frames.release(frame_idx)

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
            "vision_pos_embeds": self._position_embeddings.on(self.inference_device),
        }

        return features

    def clear_all(self) -> None:
        """Refuse to drop the cached encoder output (the HF session calls this
        from ``reset_inference_session``).

        The output belongs to the clip, not to one session, and the pixels it
        was computed from are gone.

        Raises:
            ModelRuntimeError: If any frame's encoder output is cached.
        """
        if self._feats:
            raise ModelRuntimeError(
                message="A clip's cached encoder output cannot be cleared: it is "
                "shared by every session on the clip and the pixels of encoded "
                "frames were released. Embed the clip again instead."
            )


class VideoClipCache:
    """Clips by id, least recently used evicted first to stay within ``max_frames``.

    Evicting a clip only forgets it here: a call that already got the clip
    keeps it and finishes on it.

    Args:
        max_frames: Most frames held at once, over all clips.
    """

    def __init__(self, max_frames: int):
        if max_frames < 1:
            raise EnvironmentConfigurationError(
                message="SAM3 video clip cache must hold at least 1 frame, got "
                f"{max_frames}."
            )

        self._max_frames = max_frames
        self._clips: "OrderedDict[str, VideoClip]" = OrderedDict()
        self._lock = threading.Lock()

    @property
    def max_frames(self) -> int:
        """Most frames held at once; also the most frames one clip may have."""
        return self._max_frames

    def make_room(self, video_id: str, *, n_frames: int) -> None:
        """Forget the clip under ``video_id`` and the least recently used clips
        until a clip of ``n_frames`` fits.

        Called before a clip is built, so the memory of the clips it replaces
        is free while it is.

        Args:
            video_id: Caller's id for the clip about to be stored.
            n_frames: Its number of frames.

        Raises:
            ModelInputError: If the clip has more frames than the cache holds.
        """
        if n_frames > self._max_frames:
            raise ModelInputError(
                message=f"The video has {n_frames} frames; the most a clip may "
                f"have is {self._max_frames}, which is what {CACHE_LIMIT_SETTING} "
                "holds."
            )

        with self._lock:
            self._make_room(video_id, n_frames=n_frames)

    def put(self, video_id: str, clip: VideoClip) -> None:
        """Store ``clip`` under ``video_id``, replacing any clip already there
        and evicting the least recently used clips until it fits.

        Args:
            video_id: Caller's id for the clip.
            clip: The clip.

        Raises:
            ModelInputError: If the clip has more frames than the cache holds.
        """
        if clip.n_frames > self._max_frames:
            raise ModelInputError(
                message=f"The video has {clip.n_frames} frames; the most a clip "
                f"may have is {self._max_frames}, which is what "
                f"{CACHE_LIMIT_SETTING} holds."
            )

        with self._lock:
            self._make_room(video_id, n_frames=clip.n_frames)
            self._clips[video_id] = clip

    def _make_room(self, video_id: str, *, n_frames: int) -> None:
        self._clips.pop(video_id, None)
        held = sum(clip.n_frames for clip in self._clips.values())
        while held + n_frames > self._max_frames:
            held -= self._clips.popitem(last=False)[1].n_frames

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
