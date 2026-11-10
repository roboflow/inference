"""Deterministic stand-ins for the SAM2 and SAM3 models.

Not model inference; nothing is downloaded. The fakes copy the public call
shapes of ``inference_models`` ``SAM2Torch`` and ``SAM3Torch`` (sam2_torch.py
and sam3_torch.py), including SAM2 dropping the prompt axis for one prompt and
SAM3 returning ``(N, 1, H, W)`` host masks per prompt::

    FakeSam2.segment_images, per image
      no boxes        K proposals (H, W); best = centre rectangle, score 0.8
      box i of n      K proposals; best = the box itself (inclusive pixels),
                      score 0.9 - 0.2 * i; others shrink/grow it by 1 px
      K               3 with multi_mask_output (best is proposal 1), else 1
      logits          +4 inside a proposal, -4 outside

    FakeSam3.segment_with_text_prompts, per image and prompt
      rows of ``table[text]``: (box or None for an empty mask, score),
      kept when score >= output_prob_thresh, like the real model
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from roboflow_workflows.v2.blocks.models.cache import ModelRequest

from inference_models.models.sam2.entities import SAM2ImageEmbeddings, SAM2Prediction

Box = Tuple[int, int, int, int]
SAM3_TABLE: Mapping[Optional[str], Sequence[Tuple[Optional[Box], float]]] = {
    # Two overlapping cars: mask IoU 60/80 = 0.75; NMS at 0.5 keeps the better.
    "car": (((2, 2, 11, 9), 0.9), ((2, 2, 11, 7), 0.7), ((20, 3, 25, 9), 0.4)),
    # The person overlaps the first car exactly: cross-class NMS removes it.
    "person": (((2, 2, 11, 9), 0.6), (None, 0.95)),
    None: (((1, 1, 28, 18), 0.8),),
}


def box_mask(box: Optional[Box], *, size_hw: Tuple[int, int]) -> torch.Tensor:
    """Bool ``(H, W)`` mask of an inclusive ``(x1, y1, x2, y2)`` box; empty for None."""
    mask = torch.zeros(size_hw, dtype=torch.bool)
    if box is None:
        return mask

    x1, y1, x2, y2 = box
    mask[max(y1, 0) : y2 + 1, max(x1, 0) : x2 + 1] = True

    return mask


class FakeSam2:
    """``embed_images`` and ``segment_images`` with SAM2Torch's shapes."""

    def __init__(self, request: Optional[ModelRequest] = None):
        self.request = request
        self.embedded: List[int] = []
        self.embed_options: List[Dict[str, Any]] = []
        # Per segmented image: its embeddings object, box prompt and K choice.
        self.segmented: List[SAM2ImageEmbeddings] = []
        self.prompts: List[Optional[torch.Tensor]] = []
        self.multimask: List[bool] = []

    def embed_images(
        self, images: List[torch.Tensor], **options: Any
    ) -> List[SAM2ImageEmbeddings]:
        self.embedded.append(len(images))
        self.embed_options.append(options)
        embeddings = [
            SAM2ImageEmbeddings(
                image_hash=f"fake-{index}",
                image_size_hw=tuple(image.shape[1:]),
                embeddings=image.float().mean().reshape(1),
                high_resolution_features=[],
            )
            for index, image in enumerate(images)
        ]

        return embeddings

    def segment_images(
        self,
        embeddings: List[SAM2ImageEmbeddings],
        boxes: Optional[List[torch.Tensor]] = None,
        multi_mask_output: bool = True,
        return_logits: bool = False,
    ) -> List[SAM2Prediction]:
        assert return_logits, "the block binarizes logits itself"
        predictions = []
        for index, embedding in enumerate(embeddings):
            prompt = None if boxes is None else boxes[index]
            self.segmented.append(embedding)
            self.prompts.append(prompt)
            self.multimask.append(multi_mask_output)
            predictions.append(
                _proposals(
                    embedding.image_size_hw, prompt=prompt, multimask=multi_mask_output
                )
            )

        return predictions


def _proposals(
    size_hw: Tuple[int, int], *, prompt: Optional[torch.Tensor], multimask: bool
) -> SAM2Prediction:
    height, width = size_hw
    if prompt is None:
        targets = [(width // 4, height // 4, 3 * width // 4 - 1, 3 * height // 4 - 1)]
        best_scores = [0.8]
    else:
        targets = [tuple(int(value) for value in row) for row in prompt.tolist()]
        best_scores = [0.9 - 0.2 * index for index in range(len(targets))]
    shifts = (1, 0, -1) if multimask else (0,)
    masks, scores = [], []
    for (x1, y1, x2, y2), best in zip(targets, best_scores):
        proposals = [
            box_mask((x1 + s, y1 + s, x2 - s, y2 - s), size_hw=size_hw) for s in shifts
        ]
        masks.append(torch.stack(proposals))
        scores.append([best if s == 0 else best - 0.3 for s in shifts])
    logits = torch.stack(masks).float() * 8.0 - 4.0
    scores = torch.tensor(scores, dtype=torch.float32)
    if logits.shape[0] == 1:
        # SAM2Torch returns masks[0] when there is one prompt set.
        logits, scores = logits[0], scores[0]
    prediction = SAM2Prediction(masks=logits, scores=scores, logits=logits)

    return prediction


class FakeSam3:
    """``segment_with_text_prompts`` with SAM3Torch's dense result shape."""

    def __init__(
        self,
        request: Optional[ModelRequest] = None,
        *,
        table: Mapping[Optional[str], Sequence[Tuple[Optional[Box], float]]] = (
            SAM3_TABLE
        ),
    ):
        self.request = request
        self.table = table
        self.calls: List[Dict[str, Any]] = []

    def segment_with_text_prompts(
        self,
        images: List[torch.Tensor],
        prompts: List[Dict],
        output_prob_thresh: float = 0.5,
        max_detections: int = -1,
        mask_format: str = "dense",
    ) -> List[List[Dict[str, Any]]]:
        self.calls.append(
            {
                "images": len(images),
                "prompts": prompts,
                "output_prob_thresh": output_prob_thresh,
                "mask_format": mask_format,
            }
        )
        results = []
        for image in images:
            size_hw = tuple(image.shape[1:])
            per_prompt = []
            for index, prompt in enumerate(prompts):
                rows = [
                    (box, score)
                    for box, score in self.table.get(prompt["text"], ())
                    if score >= output_prob_thresh
                ]
                masks = np.zeros((len(rows), 1, *size_hw), dtype=bool)
                for row, (box, _) in enumerate(rows):
                    masks[row, 0] = box_mask(box, size_hw=size_hw).numpy()
                per_prompt.append(
                    {
                        "prompt_index": index,
                        "masks": masks,
                        "scores": [score for _, score in rows],
                    }
                )
            results.append(per_prompt)

        return results
