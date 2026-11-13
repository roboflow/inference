"""Per-clip state shared by every offline tracking session on a clip.

An offline (whole-clip) session runs on a clip's preprocessed frames and
reuses the image-encoder output of every frame it has already seen. Both
depend only on the frames, not on the prompts, so they are kept per clip and
shared by all sessions on it instead of being rebuilt per request.

A frame is held either as its preprocessed pixels or, once encoded, as its
encoder output: the pixels are released the moment the output is cached, and
the output is then kept for as long as the clip is. The cache is sized in
frames, from a limit in GB of host RAM and the bytes one frame can hold.

Both are kept in memory mapped per frame (``host_tensor``) rather than taken
from the process allocator, so that what a frame holds goes back to the
operating system the moment the frame is released or its clip evicted.
"""

import mmap
import sys
import threading
from collections import OrderedDict
from contextlib import contextmanager
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


def host_tensor(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A zeroed CPU tensor on an anonymous memory mapping of its own.

    The mapping is returned to the operating system when the last tensor on
    it is freed. Memory freed to the process allocator instead tends to stay
    resident: frames are allocated and freed one by one from several threads,
    and the allocator keeps what each thread freed for that thread. Measured
    with allocator-backed frames, two cached 480-frame clips grew from 17.7 GB
    to 24.2 GB of resident memory once a third clip had replaced the first.

    Args:
        shape: Shape of the tensor; no dimension may be 0.
        dtype: Dtype of the tensor.

    Returns:
        The tensor.
    """
    n_bytes = torch.empty((), dtype=dtype).element_size()
    for dimension in shape:
        n_bytes *= dimension
    if sys.platform == "win32":
        mapping = mmap.mmap(-1, n_bytes)
    else:
        # Private, so that it is ordinary anonymous memory of the process
        # rather than shared memory.
        mapping = mmap.mmap(-1, n_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    tensor = torch.frombuffer(mapping, dtype=dtype).view(shape)

    return tensor


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

    The copy stays on the device the encoder made it on, and every frame is
    checked against it there. At the model's input size it is about 56 MB in
    bfloat16: moving it to the CPU to check each frame, and back to the GPU on
    every lookup of a frame's features, cost as much as encoding the frame
    (46 ms of a frame's 130 ms in a first track on an L40S, and 10 of 22 ms
    per frame in a track of frames already encoded).
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
            self._embeds = [p.clone() for p in embeds]
            return

        if not all(torch.equal(a, b) for a, b in zip(self._embeds, embeds)):
            raise ModelInputError(
                message=f"Position embeddings differ on frame {frame_idx}; "
                "every frame must share them."
            )

    def on(self, device: torch.device) -> List[torch.Tensor]:
        """The kept position embeddings, on the device they were kept on.

        Args:
            device: Device the caller needs them on; it must be the one the
                encoder made them on.

        Returns:
            The position embeddings.

        Raises:
            ModelRuntimeError: If ``device`` is another device.
        """
        kept_on = self._embeds[0].device
        if _device_of(kept_on) != _device_of(device):
            raise ModelRuntimeError(
                message=f"Position embeddings are kept on {kept_on}, but were "
                f"asked for on {device}."
            )

        return self._embeds


def _device_of(device: torch.device) -> torch.device:
    """``device`` with its index, so that ``cuda`` and ``cuda:0`` compare equal."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    return device


class ClipFrames:
    """A clip's preprocessed frames, each its own tensor so each can be released.

    Stands in for the HF session's ``processed_frames`` (a dict of frames by
    index, of which the model uses ``len()`` and ``[frame_idx]``).

    Args:
        frames: Preprocessed frames ``(3, H, W)``, one tensor per frame, each
            on memory of its own (``host_tensor``).
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
        self._feats[frame_idx] = _on_one_mapping(features["vision_feats"])
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


def _on_one_mapping(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Copies of ``tensors`` on the CPU, side by side on one memory mapping."""
    dtypes = {tensor.dtype for tensor in tensors}
    if len(dtypes) != 1:
        raise ModelRuntimeError(
            message=f"Encoder output maps of one frame differ in dtype: {dtypes}."
        )

    block = host_tensor((sum(t.numel() for t in tensors),), dtypes.pop())
    copies, start = [], 0
    for tensor in tensors:
        copy = block[start : start + tensor.numel()].view(tensor.shape)
        copy.copy_(tensor)
        copies.append(copy)
        start += tensor.numel()

    return copies


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
        self._reserved_frames = 0

    @contextmanager
    def build(self, video_id: str):
        """Reserve storage for a clip before allocating its processed frames.

        Args:
            video_id: Id published only after a successful build.

        Yields:
            ClipReservation: Incremental capacity reservation, released on failure.
        """
        reservation = ClipReservation(self, video_id)
        try:
            yield reservation
        finally:
            reservation.close()

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
        if self._reserved_frames + n_frames > self._max_frames:
            raise ModelInputError(
                message="SAM3 video cache capacity is reserved by an embedding."
            )
        self._clips.pop(video_id, None)
        held = self._reserved_frames + sum(
            clip.n_frames for clip in self._clips.values()
        )
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

    def peek(self, video_id: str) -> Optional[VideoClip]:
        """Return the clip stored under ``video_id``, if any, as recently used.

        Args:
            video_id: Caller's id for the clip.

        Returns:
            The clip, or None when none is stored under ``video_id``.
        """
        with self._lock:
            clip = self._clips.get(video_id)
            if clip is not None:
                self._clips.move_to_end(video_id)

        return clip

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


class ClipReservation:
    """Account for one unpublished clip while its frames are allocated.

    Args:
        cache: Cache that owns the storage budget.
        video_id: Id to publish after successful preprocessing.
    """

    def __init__(self, cache: VideoClipCache, video_id: str):
        self._cache = cache
        self._video_id = video_id
        self._frames = 0
        self._closed = False

    def add(self, n_frames: int) -> None:
        """Reserve additional frames before allocating their processed tensors.

        When space is needed, reclaim the previous clip under this id before
        evicting unrelated clips in LRU order. Without that pressure, keep the
        previous clip until commit so a failed build leaves it available.
        Once reclaimed, the previous clip cannot be restored on failure.

        Args:
            n_frames: Additional frames in the next preprocessing batch.

        Raises:
            ModelInputError: The clip or combined reservations exceed capacity.
        """
        cache = self._cache
        with cache._lock:
            if self._closed or n_frames < 0:
                raise ModelInputError(message="Invalid video cache reservation.")
            if self._frames + n_frames > cache.max_frames:
                raise ModelInputError(
                    message=(
                        f"The video has more than {cache.max_frames} frames; "
                        f"the clip exceeds {CACHE_LIMIT_SETTING}."
                    )
                )
            if cache._reserved_frames + n_frames > cache.max_frames:
                raise ModelInputError(
                    message="SAM3 video cache capacity is reserved by an embedding."
                )
            held = sum(clip.n_frames for clip in cache._clips.values())
            if held + cache._reserved_frames + n_frames > cache.max_frames:
                replaced = cache._clips.pop(self._video_id, None)
                if replaced is not None:
                    held -= replaced.n_frames
                del replaced

            while held + cache._reserved_frames + n_frames > cache.max_frames:
                held -= cache._clips.popitem(last=False)[1].n_frames
            cache._reserved_frames += n_frames
            self._frames += n_frames

    def commit(self, clip: VideoClip) -> None:
        """Publish a completed clip using its already reserved capacity.

        Args:
            clip: Clip with exactly the reserved number of frames.
        """
        cache = self._cache
        with cache._lock:
            if self._closed or clip.n_frames != self._frames:
                raise ModelInputError(
                    message="Video clip does not match its reservation."
                )
            cache._reserved_frames -= self._frames
            self._frames = 0
            self._closed = True
            cache._clips.pop(self._video_id, None)
            cache._clips[self._video_id] = clip

    def close(self) -> None:
        """Release unpublished storage reservations on every exit path."""
        with self._cache._lock:
            self._cache._reserved_frames -= self._frames
            self._frames = 0
            self._closed = True
