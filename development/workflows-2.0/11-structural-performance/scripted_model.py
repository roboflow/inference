"""Deterministic stand-in detector with many, varied detections per frame.

The 10 fake models return one or three boxes. Drawing needs more: empty
frames, dozens of boxes, labels crossing the frame edge, class ids beyond the
palette and tied confidences. Detections depend only on the frame's marker
(pixel ``(0, 0, 0)``, as 10 ``check_parity.marker_frames`` sets it) and size::

    marker m, frame H x W
    count        none when m % 4 == 0, else (m * 7) % 41 boxes
    box i        clamped to the frame; x1/y1 end in .25/.75 (int() truncates)
                 every 5th box starts in the top-left corner (label off-frame)
    class_id     (31 i + m) % 80, int64 (palette has 21 colours)
    confidence   0.40 + 0.05 * ((i + m) % 12), float32: ties in every frame

Like the TRT model, ``pre_process`` takes one tensor or a list and
``forward`` one stacked batch. ``post_process`` keeps detections at or above
``confidence``.
"""

from typing import Any, List, Optional, Tuple

import torch

from inference_models.models.base.object_detection import Detections

CLASS_COUNT = 80
MAX_COUNT = 41


class ScriptedDetectionModel:
    """CPU or GPU stand-in detector; see the module doc for its detections.

    Args:
        device: Device of the returned detections.
    """

    class_names = [f"class-{index}" for index in range(CLASS_COUNT)]

    def __init__(self, *, device: str = "cpu"):
        self._device = torch.device(device)
        self.forward_shapes: List[Tuple[int, ...]] = []

    def pre_process(
        self, images: Any, input_color_format: Optional[str] = None, **kwargs: Any
    ) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        """Stack each image's top-left 8 x 8 pixels; metadata: H, W, marker."""
        images = images if isinstance(images, list) else [images]
        batch = torch.stack([image[:, :8, :8].float() for image in images])
        metadata = [
            (int(image.shape[1]), int(image.shape[2]), int(image[0, 0, 0]))
            for image in images
        ]

        return batch, metadata

    def forward(self, batch: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Record the input shape; return the input."""
        self.forward_shapes.append(tuple(batch.shape))

        return batch

    def post_process(
        self,
        raw: torch.Tensor,
        metadata: List[Tuple[int, int, int]],
        confidence: float = 0.0,
        **kwargs: Any,
    ) -> List[Detections]:
        """One ``Detections`` per image, in batch order."""
        detections = [
            scripted_detections(
                marker,
                height=height,
                width=width,
                confidence=confidence,
                device=self._device,
            )
            for height, width, marker in metadata
        ]

        return detections

    def __call__(self, images: Any, **kwargs: Any) -> List[Detections]:
        network_input, metadata = self.pre_process(images)
        detections = self.post_process(self.forward(network_input), metadata, **kwargs)

        return detections


def scripted_detections(
    marker: int,
    *,
    height: int,
    width: int,
    confidence: float = 0.0,
    device: torch.device = torch.device("cpu"),
) -> Detections:
    """The detections of a frame with ``marker`` and size ``height x width``.

    Args:
        marker: The frame's marker value.
        height: Frame height.
        width: Frame width.
        confidence: Keep detections at or above this confidence.
        device: Device of the returned tensors.

    Returns:
        Detections with float32 ``xyxy``, int64 ``class_id``, float32
        ``confidence``.
    """
    rows = []
    count = 0 if marker % 4 == 0 else (marker * 7) % MAX_COUNT
    for index in range(count):
        box_w = 30 + (index * 13 + marker) % 90
        box_h = 20 + (index * 7 + marker) % 60
        if index % 5 == 0:
            x1, y1 = 0.25, 0.75
        else:
            x1 = (index * 97 + marker * 31) % width + 0.75
            y1 = (index * 53 + marker * 11) % height + 0.25
        x2, y2 = min(x1 + box_w, width - 1), min(y1 + box_h, height - 1)
        score = 0.40 + 0.05 * ((index + marker) % 12)
        rows.append((x1, y1, x2, y2, (31 * index + marker) % CLASS_COUNT, score))
    kept = [row for row in rows if row[5] >= confidence]

    detections = Detections(
        xyxy=torch.tensor(
            [row[:4] for row in kept], dtype=torch.float32, device=device
        ).reshape(-1, 4),
        class_id=torch.tensor(
            [row[4] for row in kept], dtype=torch.int64, device=device
        ),
        confidence=torch.tensor(
            [row[5] for row in kept], dtype=torch.float32, device=device
        ),
    )

    return detections
