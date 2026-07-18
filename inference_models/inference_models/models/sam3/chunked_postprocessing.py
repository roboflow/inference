"""Memory-bounded variant of sam3's PostProcessImage for serving.

Upstream `sam3.eval.postprocessors.PostProcessImage` (pinned sam3==0.1.4) has
two serving-hostile behaviors (measured in development/sam3_debug/FINDINGS.md):

1. The `max_dets_per_img` topk runs in `process_results` AFTER every
   above-threshold mask has been interpolated to original resolution and
   (optionally) RLE-encoded, so the cap never bounds peak memory.
2. `_process_masks` interpolates ALL kept masks as one float32 batch
   (k x 1 x H x W); on GPU OOM a try/except silently falls back to CPU and
   materializes that batch in host RAM (+14 GiB observed at k=100 @ 4032²).

This subclass fixes both without modifying the pinned package:

- the detection cap is folded into the `keep` selection inside
  `_process_boxes_and_labels` (applied even when thresholding is disabled),
  so boxes, scores, labels AND masks are truncated to the top
  `max_dets_per_img` detections BEFORE interpolation; capped survivors are
  score-descending, matching the ordering of the parent's late topk (which
  remains as a no-op safety net);
- mask interpolation + RLE encoding run in fixed-size chunks
  (`INFERENCE_MODELS_SAM3_MASK_PROCESSING_CHUNK_SIZE`, default 8), so neither device ever
  holds the full k x H x W float32 batch, and the GPU->CPU fallback is
  bounded to one chunk.

`keep` is a list of index tensors rather than the parent's boolean mask:
uncapped selection is index-of-nonzero in query order (identical output to
the parent), capped selection is topk order (identical to the parent's
late sort).
"""

import torch
from sam3.eval.postprocessors import PostProcessImage
from sam3.model import box_ops
from sam3.model.data_misc import interpolate
from sam3.train.masks_ops import robust_rle_encode

from inference_models.configuration import (
    INFERENCE_MODELS_SAM3_MASK_PROCESSING_CHUNK_SIZE,
)


class ChunkedPostProcessImage(PostProcessImage):

    def __init__(
        self,
        *args,
        mask_chunk_size: int = INFERENCE_MODELS_SAM3_MASK_PROCESSING_CHUNK_SIZE,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._mask_chunk_size = max(1, int(mask_chunk_size))

    def _process_boxes_and_labels(
        self, target_sizes, forced_labels, out_bbox, out_probs
    ):
        # Mirrors the parent implementation; the only addition is folding
        # max_dets_per_img into `keep` so the cap applies before masks are
        # interpolated.
        if out_bbox is None:
            return None, None, None, None
        assert len(out_probs) == len(target_sizes)
        if self.to_cpu:
            out_probs = out_probs.cpu()
        scores, labels = out_probs.max(-1)
        if forced_labels is None:
            labels = torch.ones_like(labels)
        else:
            labels = forced_labels[:, None].expand_as(labels)

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        if self.to_cpu:
            boxes = boxes.cpu()

        keep = None
        if self.detection_threshold > 0 or self.max_dets_per_img > 0:
            if self.detection_threshold > 0:
                candidates = scores > self.detection_threshold
            else:
                # cap requested with thresholding disabled: every query is a
                # candidate, the top-k fold below still bounds the pipeline
                candidates = torch.ones_like(scores, dtype=torch.bool)
            keep = []
            for i in range(len(candidates)):
                kept_indices = candidates[i].nonzero(as_tuple=False).squeeze(1)
                if 0 < self.max_dets_per_img < kept_indices.numel():
                    # topk returns score-descending order — same ordering the
                    # parent's late topk produced, kept for output parity
                    top = torch.topk(
                        scores[i][kept_indices], self.max_dets_per_img
                    ).indices
                    kept_indices = kept_indices[top]
                keep.append(kept_indices)
            assert len(keep) == len(boxes) == len(scores) == len(labels)
            boxes = [b[k.to(b.device)] for b, k in zip(boxes, keep)]
            scores = [s[k.to(s.device)] for s, k in zip(scores, keep)]
            labels = [l[k.to(l.device)] for l, k in zip(labels, keep)]

        return boxes, scores, labels, keep

    def _process_masks(self, target_sizes, pred_masks, consistent=True, keep=None):
        if pred_masks is None or consistent:
            # the consistent branch rejects RLE upstream and interpolates a
            # single uniform batch; leave it to the parent
            return super()._process_masks(
                target_sizes, pred_masks, consistent=consistent, keep=keep
            )
        if self.always_interpolate_masks_on_gpu:
            gpu_device = target_sizes.device
            assert gpu_device.type == "cuda"
            pred_masks = pred_masks.to(device=gpu_device)

        out_masks = [None] * len(pred_masks)
        assert keep is None or len(keep) == len(pred_masks)
        for i, mask in enumerate(pred_masks):
            h, w = target_sizes[i]
            if keep is not None:
                mask = mask[keep[i]]
            pieces = []
            for start in range(0, mask.shape[0], self._mask_chunk_size):
                chunk = mask[start : start + self._mask_chunk_size]
                try:
                    interpolated = (
                        interpolate(
                            chunk.unsqueeze(1),
                            (h, w),
                            mode="bilinear",
                            align_corners=False,
                        ).sigmoid()
                        > 0.5
                    )
                except Exception:
                    chunk_cpu = chunk.cpu()
                    interpolated = (
                        interpolate(
                            chunk_cpu.unsqueeze(1),
                            (h, w),
                            mode="bilinear",
                            align_corners=False,
                        ).sigmoid()
                        > 0.5
                    )
                    interpolated = interpolated.to(chunk.device)
                if self.convert_mask_to_rle:
                    pieces.extend(robust_rle_encode(interpolated.squeeze(1)))
                else:
                    pieces.append(interpolated.cpu() if self.to_cpu else interpolated)
                del interpolated
            if self.convert_mask_to_rle:
                out_masks[i] = pieces
            else:
                if pieces:
                    out_masks[i] = torch.cat(pieces, dim=0)
                else:
                    empty = mask.new_zeros((0, 1, int(h), int(w)), dtype=torch.bool)
                    out_masks[i] = empty.cpu() if self.to_cpu else empty
        return out_masks
