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
model call on a frame computes every object there, and SAM3 keeps an
object's points as a conditioning frame only when they are added before that
frame was tracked for the object; points added to an already tracked frame
are recomputed away by the next propagation. So in a shared session one
object's prompts on a frame another object was already run on would be
silently dropped. Each object is propagated forward and then backward from
its own first prompted frame, so every frame is computed once per object.
"""

import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from inference_models.configuration import SAM3_VIDEO_MAX_CACHED_CLIPS
from inference_models.errors import MissingDependencyError, ModelInputError
from inference_models.models.common.hf_streaming_video import HFStreamingVideoBase
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle
from inference_models.models.sam3_tracker_video.clip_cache import (
    VideoClip,
    VideoClipCache,
)

#: One object's point prompts: frame index -> ((x, y, label), ...) in click order.
ObjectPrompts = Dict[int, Tuple[Tuple[float, float, int], ...]]


class SAM3TrackerVideo(HFStreamingVideoBase):
    """SAM3 visually prompted streaming tracker via the transformers port.

    Accepts point and box prompts. Text prompting belongs to the
    ``sam3video`` concept tracker.
    """

    _supports_text_prompts = False

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._video_clip_cache = VideoClipCache(size_limit=SAM3_VIDEO_MAX_CACHED_CLIPS)

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

    @torch.inference_mode()
    def embed_video(
        self,
        frames: List[np.ndarray],
        *,
        video_id: str,
        encode_frames: bool = False,
    ) -> dict:
        """Preprocess a clip and keep it under ``video_id`` for offline tracking.

        Embedding a clip again under the same id replaces it. The image
        encoder runs on a frame the first time any session needs it; with
        ``encode_frames`` a background thread also encodes every frame, one
        frame per hold of the model lock so other calls interleave, and stops
        once the clip is replaced or evicted.

        Args:
            frames: The clip's frames in order, RGB ``(H, W, 3)`` uint8, all
                the same size.
            video_id: Caller-chosen id the clip is kept under.
            encode_frames: Start encoding every frame in the background.

        Returns:
            ``video_id``, ``n_frames``, ``height`` and ``width`` of the clip.

        Raises:
            ModelInputError: If there are no frames or they differ in size.
        """
        if not frames:
            raise ModelInputError(message="A video needs at least one frame.")

        sizes = {frame.shape[:2] for frame in frames}
        if len(sizes) != 1:
            raise ModelInputError(
                message=f"All frames of a video must have one size, got {sorted(sizes)}."
            )

        processed = self._processor.video_processor(
            videos=[np.asarray(frame) for frame in frames],
            device="cpu",
            return_tensors="pt",
        )
        clip = VideoClip(
            processed.pixel_values_videos[0].to(dtype=self._dtype),
            video_size=tuple(int(v) for v in processed.original_sizes[0]),
            inference_device=self._device,
        )
        self._video_clip_cache.put(video_id, clip)

        if encode_frames:
            threading.Thread(
                target=self._encode_clip_frames, args=(video_id, clip), daemon=True
            ).start()

        description = {
            "video_id": video_id,
            "n_frames": clip.n_frames,
            "height": clip.video_size[0],
            "width": clip.video_size[1],
        }

        return description

    def _encode_clip_frames(self, video_id: str, clip: VideoClip) -> None:
        session = self._clip_session(clip)
        for frame_idx in range(clip.n_frames):
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

        Each such object is segmented from all its prompts on the clip,
        applied frame by frame in frame order, without propagation, as a
        preview of a click before tracking.

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
                or a frame has exclude points of an object without an include
                point.
        """
        clip = self._video_clip_cache.get(video_id)
        per_object = _prompts_by_object(prompts, n_frames=clip.n_frames)
        if not 0 <= frame_idx < clip.n_frames:
            raise ModelInputError(
                message=f"frame_idx {frame_idx} is outside the clip's "
                f"{clip.n_frames} frames."
            )

        masks = {}
        with self._lock:
            for obj_id, object_prompts in per_object.items():
                if frame_idx not in object_prompts:
                    continue

                outputs = self._prompted_session(clip, obj_id, object_prompts)[1]
                masks[str(obj_id)] = self._clip_masks(clip, outputs[frame_idx])[0]

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

        Each object gets its own session: its prompts are applied frame by
        frame in frame order, then it is propagated forward from its first
        prompted frame to the end of the clip and backward from there to the
        start, so every frame is computed once per object.

        Args:
            video_id: Id the clip was embedded under.
            prompts: Point prompts as for ``segment_video_frame``; every object
                needs an include point.
            frame_indices: Frames to return masks for; all frames when None.
                The others are still tracked through.

        Returns:
            One entry per returned frame in frame order: ``frame_idx`` and
            ``masks``, COCO RLE by object id as a string.

        Raises:
            ModelInputError: If the clip is unknown, there are no prompts, a
                prompt or frame index is out of range, or a frame has exclude
                points of an object without an include point.
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
        with self._lock:
            for obj_id, object_prompts in per_object.items():
                session = self._prompted_session(clip, obj_id, object_prompts)[0]
                start = min(object_prompts)
                for reverse in (False, True):
                    for output in self._model.propagate_in_video_iterator(
                        session, start_frame_idx=start, reverse=reverse
                    ):
                        frame = int(output.frame_idx)
                        if frame in wanted:
                            masks[frame][str(obj_id)] = self._clip_masks(
                                clip, output.pred_masks
                            )[0]

        results = [{"frame_idx": i, "masks": masks[i]} for i in sorted(masks)]

        return results

    def _clip_session(self, clip: VideoClip) -> Any:
        from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
            Sam3TrackerVideoInferenceSession,
        )

        session = Sam3TrackerVideoInferenceSession(
            video=clip.pixel_values,
            video_height=clip.video_size[0],
            video_width=clip.video_size[1],
            inference_device=self._device,
            video_storage_device="cpu",
            inference_state_device=self._device,
            dtype=self._dtype,
        )
        # The clip stands in for the session's one-frame feature cache, so
        # every session on the clip reuses the frames already encoded.
        session.cache = clip

        return session

    def _prompted_session(
        self, clip: VideoClip, obj_id: int, object_prompts: ObjectPrompts
    ) -> Tuple[Any, Dict[int, torch.Tensor]]:
        """A fresh session for one object with its prompts applied in frame order,
        and the predicted masks on each prompted frame."""
        session = self._clip_session(clip)
        outputs = {}
        for frame_idx in sorted(object_prompts):
            points = object_prompts[frame_idx]
            self._processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=frame_idx,
                obj_ids=obj_id,
                input_points=[[[[x, y] for x, y, _ in points]]],
                input_labels=[[[label for _, _, label in points]]],
            )
            outputs[frame_idx] = self._model(
                inference_session=session, frame_idx=frame_idx
            ).pred_masks

        return session, outputs

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

    exclude_only = sorted(
        (frame_idx, obj_id)
        for obj_id, frames in grouped.items()
        for frame_idx, points in frames.items()
        if not any(label == 1 for _, _, label in points)
    )
    if exclude_only:
        # Each prompted frame is segmented from its own points, so exclude
        # points alone leave nothing to exclude from.
        raise ModelInputError(
            message="Exclude points need an include point of the same object on "
            f"the same frame: (frame, object) {exclude_only}."
        )

    per_object = {
        obj_id: {frame_idx: tuple(points) for frame_idx, points in frames.items()}
        for obj_id, frames in sorted(grouped.items())
    }

    return per_object
