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

Besides streaming, ``SAM3TrackerVideo`` tracks whole clips offline
(``embed_video`` / ``segment_video_frame`` / ``propagate_video``): a clip is
embedded once under a caller-chosen id, point prompts may sit on any frames,
and propagation runs forward and backward over the clip.

Offline tracking gives each object its own session. In a shared session a
model call on a frame computes every object there, so one object's clicks
would be applied in the middle of another object's tracking.

An object's anchor is its first frame with an include point. The anchor is
segmented from its clicks alone and the object is tracked from it, forward to
the end of the clip and then backward to the start. Clicks on any other frame
are a correction: they are applied when tracking reaches that frame, on top
of what the tracker remembers of the object there, and the corrected frame
then conditions the frames tracked after it. Exclude points alone are valid
on a frame other than the anchor: what they are on is taken off the mask
the tracker has there, which is empty when that was all of it. See
``_track_object``.

Embedded clips are kept in host RAM, within ``SAM3_VIDEO_CACHE_MAX_GB``: a
frame is held as its preprocessed pixels until it is encoded and as its
encoder output afterwards. The limit is turned into a number of frames from
the bytes one frame can hold; the least recently used clips are evicted to
stay within it, and it is also the longest clip that can be embedded.
"""

import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from inference_models.configuration import SAM3_VIDEO_CACHE_MAX_GB
from inference_models.errors import MissingDependencyError, ModelInputError
from inference_models.logger import LOGGER
from inference_models.models.common.hf_streaming_video import HFStreamingVideoBase
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle
from inference_models.models.sam3_tracker_video.clip_cache import (
    CACHE_LIMIT_SETTING,
    PositionEmbeddings,
    VideoClip,
    VideoClipCache,
    cache_frame_limit,
    cached_frame_bytes,
    host_tensor,
)

#: One object's point prompts: frame index -> ((x, y, label), ...) in click order.
ObjectPrompts = Dict[int, Tuple[Tuple[float, float, int], ...]]

#: Frames preprocessed at a time when a clip is embedded. The video processor
#: holds about 40 MB per frame while it runs (measured: resized, rescaled and
#: normalised full-precision copies), so 16 frames keep the extra memory of an
#: embed near 0.65 GB whatever the clip's length. Preprocessing time per frame
#: is the same from 8 to 32 frames at a time.
EMBED_CHUNK_FRAMES = 16


class SAM3TrackerVideo(HFStreamingVideoBase):
    """SAM3 visually prompted streaming tracker via the transformers port.

    Accepts point and box prompts. Text prompting belongs to the
    ``sam3video`` concept tracker.
    """

    _supports_text_prompts = False

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._cached_frame_bytes = cached_frame_bytes(self._model.config, self._dtype)
        self._video_clip_cache = VideoClipCache(
            max_frames=cache_frame_limit(
                SAM3_VIDEO_CACHE_MAX_GB, frame_bytes=self._cached_frame_bytes
            )
        )
        self._position_embeddings = PositionEmbeddings()
        self._open_requests = 0
        self._open_requests_lock = threading.Lock()
        self._no_open_requests = threading.Event()
        self._no_open_requests.set()
        self._encoding_clips: set = set()
        LOGGER.info(
            "SAM3 video clip cache: %s=%s GB, %d bytes per cached frame, "
            "limit of %d frames.",
            CACHE_LIMIT_SETTING,
            SAM3_VIDEO_CACHE_MAX_GB,
            self._cached_frame_bytes,
            self._video_clip_cache.max_frames,
        )

    @classmethod
    def _resolve_transformers_classes(cls) -> Tuple[Any, Any]:
        try:
            from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
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

    # ------------------------------------------------------------------
    # Offline (whole-clip) tracking
    # ------------------------------------------------------------------

    def video_cache_limits(self) -> dict:
        """What the clip cache holds, for a caller that refuses a clip before
        decoding it.

        Returns:
            ``max_gb`` (the ``SAM3_VIDEO_CACHE_MAX_GB`` setting),
            ``bytes_per_frame`` (most bytes one cached frame holds) and
            ``max_frames`` (frames the cache holds, which is also the most
            frames one clip may have).
        """
        limits = {
            "max_gb": SAM3_VIDEO_CACHE_MAX_GB,
            "bytes_per_frame": self._cached_frame_bytes,
            "max_frames": self._video_clip_cache.max_frames,
        }

        return limits

    @torch.inference_mode()
    def embed_video(
        self,
        frames: List[np.ndarray],
        *,
        video_id: str,
        encode_frames: bool = False,
        keep_existing: bool = False,
    ) -> dict:
        """Preprocess a clip and keep it under ``video_id`` for offline tracking.

        Embedding a clip again under the same id replaces it, unless
        ``keep_existing`` is set and the clip already under that id has the
        same number of frames and frame size: that clip is then kept as it is,
        with every frame it has encoded, and nothing is preprocessed. Use it
        when an id always names the same clip, so that asking again for a clip
        the worker still has costs no encoding. The clip under
        that id and the least recently used clips that have to go for this
        one to fit are forgotten before it is preprocessed, so their memory
        is free while it is built; a call already running on one of them
        keeps it and finishes. Frames are preprocessed ``EMBED_CHUNK_FRAMES``
        at a time and each is kept as its own tensor in the model's dtype, so
        the extra memory of an embed is one chunk in full precision.

        The image encoder runs on a frame the first time any session needs
        it, after which the frame's pixels are released; with
        ``encode_frames`` a background thread also encodes every frame, one
        frame per hold of the model lock so other calls interleave, and stops
        once the clip is replaced or evicted.

        Args:
            frames: The clip's frames in order, RGB ``(H, W, 3)`` uint8, all
                the same size.
            video_id: Caller-chosen id the clip is kept under.
            encode_frames: Start encoding every frame in the background.
            keep_existing: Keep the clip already under ``video_id`` when it
                has this many frames of this size, instead of replacing it.

        Returns:
            ``video_id``, ``n_frames``, ``height`` and ``width`` of the clip.

        Raises:
            ModelInputError: If there are no frames, they differ in size, or
                there are more than the cache holds.
        """
        if not frames:
            raise ModelInputError(message="A video needs at least one frame.")

        sizes = {frame.shape[:2] for frame in frames}
        if len(sizes) != 1:
            raise ModelInputError(
                message=f"All frames of a video must have one size, got {sorted(sizes)}."
            )

        video_size = tuple(int(v) for v in sizes.pop())
        clip = self._video_clip_cache.peek(video_id) if keep_existing else None
        if clip is None or (clip.n_frames, clip.video_size) != (
            len(frames),
            video_size,
        ):
            self._video_clip_cache.make_room(video_id, n_frames=len(frames))
            clip = VideoClip(
                self._preprocess_frames(frames),
                video_size=video_size,
                inference_device=self._device,
                position_embeddings=self._position_embeddings,
            )
            self._video_clip_cache.put(video_id, clip)

        if encode_frames:
            self._start_encoding(video_id, clip)

        description = {
            "video_id": video_id,
            "n_frames": clip.n_frames,
            "height": clip.video_size[0],
            "width": clip.video_size[1],
        }

        return description

    def _preprocess_frames(self, frames: List[np.ndarray]) -> List[torch.Tensor]:
        """The model input of each frame, as one tensor per frame in the model's dtype.

        The video processor resizes, rescales and normalises each frame on its
        own, so preprocessing a chunk at a time gives the same values as
        preprocessing the whole clip.
        """
        preprocessed: List[torch.Tensor] = []
        for start in range(0, len(frames), EMBED_CHUNK_FRAMES):
            chunk = self._processor.video_processor(
                videos=[
                    np.asarray(frame)
                    for frame in frames[start : start + EMBED_CHUNK_FRAMES]
                ],
                device="cpu",
                return_tensors="pt",
            ).pixel_values_videos[0]
            # A copy per frame on memory of its own, so that a frame can be
            # released alone and none keeps the full-precision chunk alive.
            for frame in chunk:
                preprocessed.append(host_tensor(tuple(frame.shape), self._dtype))
                preprocessed[-1].copy_(frame)
            del chunk, frame

        return preprocessed

    @contextmanager
    def _request_lock(self) -> Iterator[None]:
        """The model lock for a caller's request, taken ahead of background encoding.

        The encoding thread takes the lock once per frame and takes it again
        the moment it lets go, so a request waiting on the plain lock was not
        served until the whole clip was encoded. While any request is waiting
        or running, the encoding thread starts no further frame.
        """
        with self._open_requests_lock:
            self._open_requests += 1
            self._no_open_requests.clear()

        try:
            with self._lock:
                yield
        finally:
            with self._open_requests_lock:
                self._open_requests -= 1
                if self._open_requests == 0:
                    self._no_open_requests.set()

    def _start_encoding(self, video_id: str, clip: VideoClip) -> None:
        """Encode the clip's frames in the background, one thread per clip.

        A clip kept by ``keep_existing`` may still be encoding from its first
        embed, or have finished; either way no second thread is started for it.
        """
        with self._open_requests_lock:
            if id(clip) in self._encoding_clips:
                return

            self._encoding_clips.add(id(clip))

        def encode() -> None:
            try:
                self._encode_clip_frames(video_id, clip)
            finally:
                with self._open_requests_lock:
                    self._encoding_clips.discard(id(clip))

        threading.Thread(target=encode, daemon=True).start()

    def _encode_clip_frames(self, video_id: str, clip: VideoClip) -> None:
        session = self._clip_session(clip)
        for frame_idx in range(clip.n_frames):
            self._no_open_requests.wait()

            with self._lock, torch.inference_mode():
                if not self._video_clip_cache.holds(video_id, clip):
                    return

                if not clip.has_features(frame_idx):
                    self._model._prepare_vision_features(
                        session, frame_idx, batch_size=1
                    )

    @torch.inference_mode()
    def segment_video_frame(
        self,
        video_id: str,
        *,
        prompts: List[dict],
        frame_idx: int,
    ) -> dict:
        """Masks on one frame for the objects with prompts on it.

        The preview of a click: each such object gets the mask
        ``propagate_video`` would give it on this frame from the same
        prompts. On the object's anchor (its first frame with an include
        point) that is one model call. On any other frame the object is
        tracked from its anchor to this frame, with the clicks on the frames
        in between applied on the way, so the time grows with the distance
        from the anchor (see ``_track_object``).

        Args:
            video_id: Id the clip was embedded under.
            prompts: Point prompts ``{"frame_idx", "obj_id", "x", "y",
                "label"}`` in pixels of the original frames, label 1 to
                include and 0 to exclude, in click order.
            frame_idx: Frame to return masks for.

        Returns:
            ``frame_idx`` and ``masks``: COCO RLE (``size``, string
            ``counts``) by object id, as a string.

        Raises:
            ModelInputError: If the clip is unknown, a prompt is out of range,
                or an object has no include point on any frame.
        """
        clip = self._video_clip_cache.get(video_id)
        per_object = _prompts_by_object(prompts, n_frames=clip.n_frames)
        if not 0 <= frame_idx < clip.n_frames:
            raise ModelInputError(
                message=f"frame_idx {frame_idx} is outside the clip's "
                f"{clip.n_frames} frames."
            )

        masks = {}
        with self._request_lock():
            for obj_id, object_prompts in per_object.items():
                if frame_idx not in object_prompts:
                    continue

                for frame, pred_masks in self._track_object(
                    clip, obj_id, object_prompts, only_frame=frame_idx
                ):
                    if frame == frame_idx:
                        masks[str(obj_id)] = self._clip_masks(clip, pred_masks)[0]

        result = {"frame_idx": frame_idx, "masks": masks}

        return result

    @torch.inference_mode()
    def propagate_video(
        self,
        video_id: str,
        *,
        prompts: List[dict],
        frame_indices: Optional[List[int]] = None,
    ) -> List[dict]:
        """Track every prompted object over the whole clip.

        Each object gets its own session and is tracked from its anchor, its
        first frame with an include point: forward to the end of the clip,
        then backward to the start. Its clicks on other frames are applied
        when tracking reaches them (see ``_track_object``), so every frame is
        computed once per object, and a frame with such clicks twice.

        Args:
            video_id: Id the clip was embedded under.
            prompts: Point prompts as for ``segment_video_frame``; every object
                needs an include point on at least one frame.
            frame_indices: Frames to return masks for; all frames when None.
                The others are still tracked through.

        Returns:
            One entry per returned frame in frame order: ``frame_idx`` and
            ``masks``, COCO RLE by object id as a string.

        Raises:
            ModelInputError: If the clip is unknown, there are no prompts, a
                prompt or frame index is out of range, or an object has no
                include point on any frame.
        """
        clip = self._video_clip_cache.get(video_id)
        per_object = _prompts_by_object(prompts, n_frames=clip.n_frames)
        if not per_object:
            raise ModelInputError(message="Add at least one include point to track.")

        wanted = set(range(clip.n_frames) if frame_indices is None else frame_indices)
        out_of_range = sorted(i for i in wanted if not 0 <= i < clip.n_frames)
        if out_of_range:
            raise ModelInputError(
                message=f"frame_indices {out_of_range} are outside the clip's "
                f"{clip.n_frames} frames."
            )

        masks: Dict[int, Dict[str, dict]] = {i: {} for i in sorted(wanted)}
        with self._request_lock():
            for obj_id, object_prompts in per_object.items():
                for frame, pred_masks in self._track_object(
                    clip, obj_id, object_prompts
                ):
                    if frame in wanted:
                        masks[frame][str(obj_id)] = self._clip_masks(clip, pred_masks)[
                            0
                        ]

        results = [{"frame_idx": i, "masks": masks[i]} for i in sorted(masks)]

        return results

    def _track_object(
        self,
        clip: VideoClip,
        obj_id: int,
        object_prompts: ObjectPrompts,
        *,
        only_frame: Optional[int] = None,
    ) -> Iterator[Tuple[int, torch.Tensor]]:
        """Track one object in a session of its own, from its anchor outward.

        The anchor, the object's first frame with an include point, is an
        initial conditioning frame: the model segments it from its clicks
        alone. The object is then tracked forward from the anchor and, after
        that, backward from it.

        Clicks on another frame are applied when tracking reaches the frame.
        The model treats inputs on a frame it has tracked as a correction,
        where inputs on a frame it has not tracked would make another initial
        conditioning frame, segmented without memory:

        - Clicks with an include point: the model runs the frame again with
          the object's memory and the clicks together.
        - Exclude points alone: what each point is on is segmented on its
          own (the frame from that point as an include point, no memory) and
          taken off the mask tracking gave the frame, and the rest is given
          to the model as the frame's result. A mask on the object and on
          something else loses the other thing; a mask that moved to
          something else after the object left the picture comes out empty.
          Measured on MeViS clips with the masks of two objects tracked as
          one and an exclude click on the second, the frames after the click
          had IoU 0.90 with the first object alone and 6% of the second left.
          Applied as clicks, with or without the tracked mask as the
          decoder's mask prompt, exclude points alone left most of the second
          object (73% with the mask prompt), and an empty mask for the frame
          lost the first object on half of the frames after it.

        Two things are then done with the corrected frame, both measured on
        the reported failure (the object leaves the picture and the mask
        moves to another one):

        - Its output is kept as a conditioning frame, which every frame
          tracked after it in the same direction attends to. Left where the
          model stores it, it would be one of the six most recent frames and
          stop mattering seven frames later; a mask taken off the wrong
          object was back on it on the next frame.
        - The frames tracked just before it are set aside, as many as a
          tracked frame can read back (the memory and object pointer
          window), so the frames after it start from the conditioning frames
          alone. With them, a mask taken off in the middle of a wrong stretch
          was back on the wrong object four frames later.

        Both are undone when the direction ends, so the backward pass is not
        conditioned on forward corrections. A correction therefore changes
        its own frame and the frames tracked after it, and leaves the frames
        between the anchor and itself as they are without it. The frames on
        the other side of the anchor change only when the correction is
        within the window of the anchor, because the backward pass reads the
        forward frames next to the anchor.

        With ``only_frame``, tracking stops at that frame. A frame before the
        anchor is reached as in the whole clip: forward from the anchor as
        far as a tracked frame reads back, then backward to it.

        Args:
            clip: The embedded clip.
            obj_id: The object's id.
            object_prompts: The object's points by frame.
            only_frame: Frame to stop at; the whole clip when None.

        Yields:
            ``(frame_idx, pred_masks)`` for each frame, in tracking order.
        """
        anchor = _anchor_frame(object_prompts)
        session = self._clip_session(clip)
        yield anchor, self._apply_points(
            session, obj_id, anchor, object_prompts[anchor], reverse=False
        )

        config = self._model.config
        # Frames a tracked frame reads back: memory frames and object pointers.
        reach = max(config.num_maskmem, config.max_object_pointers_in_encoder) - 1
        if only_frame is None:
            passes = [(False, None), (True, None)]
        elif only_frame > anchor:
            passes = [(False, only_frame - anchor)]
        elif only_frame < anchor:
            # The backward pass reads the forward frames next to the anchor.
            passes = [(False, reach), (True, anchor - only_frame)]
        else:
            passes = []

        outputs = session.output_dict_per_obj[session.obj_id_to_idx(obj_id)]
        conditioning = outputs["cond_frame_outputs"]
        tracked = outputs["non_cond_frame_outputs"]
        for reverse, n_frames in passes:
            corrected, set_aside = [], {}
            for output in self._model.propagate_in_video_iterator(
                session,
                start_frame_idx=anchor,
                max_frame_num_to_track=n_frames,
                reverse=reverse,
            ):
                frame_idx = int(output.frame_idx)
                if frame_idx == anchor:
                    continue

                pred_masks = output.pred_masks
                if frame_idx in object_prompts:
                    points = object_prompts[frame_idx]
                    if _has_include_point(points):
                        pred_masks = self._apply_points(
                            session, obj_id, frame_idx, points, reverse=reverse
                        )
                    else:
                        pred_masks = self._apply_excludes(
                            clip, session, obj_id, frame_idx, points, reverse=reverse
                        )
                    conditioning[frame_idx] = tracked.pop(frame_idx)
                    corrected.append(frame_idx)
                    step = 1 if reverse else -1
                    for behind in range(1, reach + 1):
                        earlier = frame_idx + step * behind
                        if earlier in tracked:
                            set_aside[earlier] = tracked.pop(earlier)
                yield frame_idx, pred_masks

            for frame_idx in corrected:
                tracked[frame_idx] = conditioning.pop(frame_idx)
            tracked.update(set_aside)

    def _clip_session(self, clip: VideoClip) -> Any:
        from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
            Sam3TrackerVideoInferenceSession,
        )

        session = Sam3TrackerVideoInferenceSession(
            video_height=clip.video_size[0],
            video_width=clip.video_size[1],
            inference_device=self._device,
            video_storage_device="cpu",
            inference_state_device=self._device,
            dtype=self._dtype,
        )
        # The session reads frames from the clip (the model uses the number
        # of frames and one frame by index), so a frame released by the clip
        # is held by no session either.
        session.processed_frames = clip.frames
        # The clip stands in for the session's one-frame feature cache, so
        # every session on the clip reuses the frames already encoded.
        session.cache = clip

        return session

    def _apply_points(
        self,
        session: Any,
        obj_id: int,
        frame_idx: int,
        points: Tuple[Tuple[float, float, int], ...],
        *,
        reverse: bool,
    ) -> torch.Tensor:
        """Run a frame with an object's points on it; the predicted masks."""
        self._processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=frame_idx,
            obj_ids=obj_id,
            input_points=[[[[x, y] for x, y, _ in points]]],
            input_labels=[[[label for _, _, label in points]]],
        )
        pred_masks = self._model(
            inference_session=session, frame_idx=frame_idx, reverse=reverse
        ).pred_masks

        return pred_masks

    def _apply_excludes(
        self,
        clip: VideoClip,
        session: Any,
        obj_id: int,
        frame_idx: int,
        points: Tuple[Tuple[float, float, int], ...],
        *,
        reverse: bool,
    ) -> torch.Tensor:
        """Run a tracked frame with exclude points alone; the predicted masks.

        What each point is on is segmented on its own, as an include point on
        the frame with no memory, and taken off the mask tracking gave the
        frame. What is left is given to the model as the frame's result, the
        way an empty mask marks an object absent.
        """
        obj_idx = session.obj_id_to_idx(obj_id)
        tracked = session.output_dict_per_obj[obj_idx]["non_cond_frame_outputs"][
            frame_idx
        ]["pred_masks"]
        kept = self._full_size_mask(clip, tracked)
        for x, y, _ in points:
            kept &= ~self._clicked_thing(clip, frame_idx, x, y)
        self._processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=frame_idx,
            obj_ids=obj_id,
            input_masks=kept.cpu().numpy(),
        )
        pred_masks = self._model(
            inference_session=session, frame_idx=frame_idx, reverse=reverse
        ).pred_masks

        return pred_masks

    def _clicked_thing(
        self, clip: VideoClip, frame_idx: int, x: float, y: float
    ) -> torch.Tensor:
        """The mask of what a point is on: the frame segmented from it alone."""
        session = self._clip_session(clip)
        pred_masks = self._apply_points(
            session, 0, frame_idx, ((x, y, 1),), reverse=False
        )

        return self._full_size_mask(clip, pred_masks)

    def _full_size_mask(
        self, clip: VideoClip, pred_masks: torch.Tensor
    ) -> torch.Tensor:
        return self._processor.post_process_masks(
            [pred_masks], original_sizes=[list(clip.video_size)], binarize=True
        )[0][0, 0].bool()

    def _clip_masks(self, clip: VideoClip, pred_masks: torch.Tensor) -> List[dict]:
        masks = self._processor.post_process_masks(
            [pred_masks], original_sizes=[list(clip.video_size)], binarize=True
        )[0][:, 0]
        rles = []
        for mask in masks:
            rle = torch_mask_to_coco_rle(mask.to(torch.uint8))
            rles.append({"size": list(rle["size"]), "counts": rle["counts"].decode()})

        return rles


def _prompts_by_object(
    prompts: List[dict], *, n_frames: int
) -> Dict[int, ObjectPrompts]:
    """Group point prompts by object, then frame, keeping click order."""
    grouped: Dict[int, Dict[int, list]] = {}
    for prompt in prompts:
        frame_idx, obj_id = int(prompt["frame_idx"]), int(prompt["obj_id"])
        if not 0 <= frame_idx < n_frames:
            raise ModelInputError(
                message=f"Prompt on frame {frame_idx} is outside the clip's {n_frames} frames."
            )
        label = int(prompt["label"])
        if label not in (0, 1):
            raise ModelInputError(message=f"Prompt label must be 0 or 1, got {label}.")
        grouped.setdefault(obj_id, {}).setdefault(frame_idx, []).append(
            (float(prompt["x"]), float(prompt["y"]), label)
        )

    no_include = sorted(
        obj_id
        for obj_id, frames in grouped.items()
        if not any(_has_include_point(points) for points in frames.values())
    )
    if no_include:
        # The anchor is segmented from its own points, so an object with
        # exclude points only has nothing to start from.
        raise ModelInputError(
            message=f"Objects {no_include} have no include point; every object "
            "needs one."
        )

    per_object = {
        obj_id: {frame_idx: tuple(points) for frame_idx, points in frames.items()}
        for obj_id, frames in sorted(grouped.items())
    }

    return per_object


def _anchor_frame(object_prompts: ObjectPrompts) -> int:
    """The object's first frame with an include point."""
    anchor = min(
        frame_idx
        for frame_idx, points in object_prompts.items()
        if _has_include_point(points)
    )

    return anchor


def _has_include_point(points: Tuple[Tuple[float, float, int], ...]) -> bool:
    return any(label == 1 for _, _, label in points)
