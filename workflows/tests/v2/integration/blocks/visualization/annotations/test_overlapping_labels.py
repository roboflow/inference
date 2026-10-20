"""Overlapping labels in compiled workflows equal the r3 merged-patch painter.

The r3 painter merged overlapping label regions pairwise before exchanging
patches (quadratic in the label count). The current painter exchanges one
patch per label, or one enclosing patch. Both must give the same pixels::

    workflow ─▶ label block ─▶ paint_labels            current painter
    workflow ─▶ label block ─▶ r3_paint_labels         oracle, patched in

The scene's labels overlap each other and are clipped at the top border.
Pixel equality with supervision is the unit suite's job
(``test_label_patch_scaling.py``).
"""

from typing import Any, Dict, List, Sequence

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.visualization.annotations.keypoint_visualization import v1 as keypoint
from roboflow_workflows.v2.blocks.visualization.annotations.label_visualization import v2 as label
from roboflow_workflows.v2.blocks.visualization.annotations import text
from roboflow_workflows.v2.blocks.visualization.annotations.text import Label

from tests.v2.integration.blocks.visualization.annotations.scene import (
    changed,
    pixels_of,
    street_image,
)
from tests.v2.integration.blocks.visualization.annotations.test_annotation_workflows import (
    SEG,
    draw,
    run_rows,
)

KEYPOINTS = {
    "type": "roboflow_core/roboflow_keypoint_detection_model@v3",
    "name": "keypoints",
    "image": "$inputs.image",
    "model_id": "shapes-keypoints",
}
CASES = {
    "label": (
        label,
        [
            SEG,
            draw(
                "roboflow_core/label_visualization@v2",
                "drawn",
                "$steps.seg.segmentation",
                text_scale=0.6,
                border_radius=3,
            ),
        ],
    ),
    "vertex_label": (
        keypoint,
        [
            KEYPOINTS,
            draw(
                "roboflow_core/keypoint_visualization@v1",
                "drawn",
                "$steps.keypoints.predictions",
                annotator_type="vertex_label",
                text_scale=0.6,
                text_padding=10,
            ),
        ],
    ),
}


def r3_paint_labels(scene: torch.Tensor, labels: Sequence[Label]) -> None:
    """The r3 painter: regions merged until none overlap, then one exchange."""
    height, width = int(scene.shape[1]), int(scene.shape[2])
    clipped = [
        text._clipped(item.region(), height=height, width=width) for item in labels
    ]
    groups = r3_merge([region for region in clipped if region is not None])
    if not groups:
        return

    host = torch.cat(
        [scene[:, y0:y1, x0:x1].reshape(-1) for x0, y0, x1, y1 in groups]
    ).numpy()
    canvas_height = max(group[3] for group in groups)
    canvas_width = max(group[2] for group in groups)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    offset = 0
    for x0, y0, x1, y1 in groups:
        size = 3 * (y1 - y0) * (x1 - x0)
        patch = host[offset : offset + size].reshape(3, y1 - y0, x1 - x0)
        canvas[y0:y1, x0:x1] = patch.transpose(1, 2, 0)
        offset += size

    for item in labels:
        text._draw(canvas, item)

    for x0, y0, x1, y1 in groups:
        patch = np.ascontiguousarray(canvas[y0:y1, x0:x1].transpose(2, 0, 1))
        scene[:, y0:y1, x0:x1] = torch.from_numpy(patch)


def r3_merge(regions: List[tuple]) -> List[tuple]:
    merged = list(regions)
    changed_any = True
    while changed_any:
        changed_any = False
        result: List[tuple] = []
        for region in merged:
            for index, other in enumerate(result):
                if overlap(region, other):
                    result[index] = (
                        min(region[0], other[0]),
                        min(region[1], other[1]),
                        max(region[2], other[2]),
                        max(region[3], other[3]),
                    )
                    changed_any = True
                    break
            else:
                result.append(region)
        merged = result

    return merged


def overlap(first: tuple, second: tuple) -> bool:
    overlapping = (
        first[0] < second[2]
        and second[0] < first[2]
        and first[1] < second[3]
        and second[1] < first[3]
    )

    return overlapping


def run_with(painter, case: str, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    module, steps = CASES[case]
    calls: List[Sequence[Label]] = []

    def recorded(scene: torch.Tensor, labels: Sequence[Label]) -> None:
        calls.append(list(labels))
        painter(scene, labels)

    with monkeypatch.context() as patch:
        patch.setattr(module, "paint_labels", recorded)
        rows = run_rows(steps, {"drawn": "$steps.drawn.image"}, [street_image()])

    result = {"image": rows[0]["drawn"], "labels": calls[0]}

    return result


@pytest.mark.parametrize("case", sorted(CASES))
def test_overlapping_labels_draw_like_the_r3_merged_painter(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # given: the street scene, whose labels overlap and cross the top border
    image = street_image()

    # when
    current = run_with(text.paint_labels, case, monkeypatch)
    oracle = run_with(r3_paint_labels, case, monkeypatch)

    # then: the scene really overlaps, and the pixels are identical
    regions = [item.region() for item in current["labels"]]
    assert any(
        overlap(first, second)
        for index, first in enumerate(regions)
        for second in regions[index + 1 :]
    )
    assert min(region[1] for region in regions) < 0
    assert changed(image, current["image"]).any()
    assert np.array_equal(pixels_of(current["image"]), pixels_of(oracle["image"]))
