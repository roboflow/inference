"""Move the segmentation prediction classes into workflows; the server re-exports them (Task 11.4).

Creates `inference/core/workflows/core_steps/common/segmentation_entities.py` from
three spans copied character for character:
  - `class Point` .. (up to `class Point3D`) and
    `class InstanceSegmentationBasePrediction` .. (up to `def _mask_to_base64_png`)
    of `inference/core/entities/responses/inference.py`;
  - `class Sam2SegmentationPrediction` .. (up to `class Sam2SegmentationResponse`)
    of `inference/core/entities/responses/sam2.py`;
then replaces each span in its server module with a re-export import placed right
after the module's `from pydantic import ...` line (so `Point3D(Point)`,
`Keypoint(Point)` and the response classes still see the names), and trims the
typing names only the moved span used (`responses/sam2.py`: `Any`, `Dict`,
`Optional`, `Union`; `responses/inference.py` keeps every name - the rest of the
module still uses them). Content-anchored, re-parses every output before
writing, idempotent (second run prints SKIP), aborts on a half-applied state.

Run:  python scripts/phase11_move_segmentation_entities.py [<repo root>]
"""

import ast
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
TARGET = ROOT / "inference/core/workflows/core_steps/common/segmentation_entities.py"
INFERENCE = ROOT / "inference/core/entities/responses/inference.py"
SAM2 = ROOT / "inference/core/entities/responses/sam2.py"
MODULE = "inference.core.workflows.core_steps.common.segmentation_entities"
INFERENCE_IMPORT = (
    f"from {MODULE} import (  # noqa: F401\n"
    "    InstanceSegmentationBasePrediction,\n"
    "    InstanceSegmentationPrediction,\n"
    "    InstanceSegmentationRLEPrediction,\n"
    "    Point,\n"
    ")\n"
)
SAM2_IMPORT = (
    f"from {MODULE} import (  # noqa: F401\n    Sam2SegmentationPrediction,\n)\n"
)
INFERENCE_ANCHOR = "from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_serializer\n"
SAM2_ANCHOR = "from pydantic import BaseModel, Field\n"
HEADER = '''"""Segmentation prediction entities, owned by Workflows.

`Point`, `InstanceSegmentationBasePrediction`, `InstanceSegmentationPrediction`
and `InstanceSegmentationRLEPrediction` (from `inference/core/entities/
responses/inference.py`) and `Sam2SegmentationPrediction` (from
`responses/sam2.py`) were MOVED here verbatim; both server modules re-export
them, so there is exactly ONE class object per name. The blocks build these
from REMOTE responses as well as from their own arithmetic - the SAM 3
interactive parser, the SAM2 remote converter, the SAM3 v1/v2/v3 and
seg-preview remote/proxy paths - so they must keep pydantic validation and
coercion (`"0.9"` -> 0.9, nested `Point` and mask validation, alias `class`).

Pinned by `tests/workflows/unit_tests/core_steps/common/test_segmentation_entities.py`
(identity through the re-export, the remote-parser coercion/rejection matrix
through both remote branches, frozen validation tables).
"""

from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


'''


def _span(source: str, start: str, end: str) -> str:
    first, last = source.index(start), source.index(end)
    assert first < last, (start, end)
    return source[first:last]


def _drop_names(source: str, prefix: str, names: list) -> str:
    lines = source.split("\n")
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    assert len(matches) == 1, (prefix, matches)
    present = [name.strip() for name in lines[matches[0]][len(prefix) :].split(",")]
    for name in names:
        assert name in present, (prefix, name)
    kept = [name for name in present if name not in names]
    lines[matches[0]] = prefix + ", ".join(kept)
    return "\n".join(lines)


def main() -> int:
    inference = INFERENCE.read_text(encoding="utf-8")
    sam2 = SAM2.read_text(encoding="utf-8")
    applied = (INFERENCE_IMPORT in inference, SAM2_IMPORT in sam2, TARGET.exists())
    if all(applied):
        print("SKIP (already applied)")
        return 0
    if any(applied):
        raise SystemExit(
            f"half-applied state {applied}; restore the three files from git first"
        )
    point_span = _span(inference, "class Point(BaseModel):", "class Point3D(Point):")
    base_span = _span(
        inference,
        "class InstanceSegmentationBasePrediction(BaseModel):",
        "def _mask_to_base64_png(",
    )
    sam2_span = _span(
        sam2,
        "class Sam2SegmentationPrediction(BaseModel):",
        "class Sam2SegmentationResponse(",
    )
    module = (
        HEADER
        + point_span.rstrip()
        + "\n\n\n"
        + base_span.rstrip()
        + "\n\n\n"
        + sam2_span.rstrip()
        + "\n"
    )
    ast.parse(module)
    new_inference = inference.replace(point_span, "", 1).replace(base_span, "", 1)
    assert new_inference.count(INFERENCE_ANCHOR) == 1
    new_inference = new_inference.replace(
        INFERENCE_ANCHOR, INFERENCE_ANCHOR + INFERENCE_IMPORT, 1
    )
    new_sam2 = sam2.replace(sam2_span, "", 1)
    assert new_sam2.count(SAM2_ANCHOR) == 1
    new_sam2 = new_sam2.replace(SAM2_ANCHOR, SAM2_ANCHOR + SAM2_IMPORT, 1)
    new_sam2 = _drop_names(
        new_sam2, "from typing import ", ["Any", "Dict", "Optional", "Union"]
    )
    for text in (new_inference, new_sam2):
        ast.parse(text)
    for name in (
        "class Point(",
        "class InstanceSegmentationBasePrediction(",
        "class InstanceSegmentationPrediction(",
        "class InstanceSegmentationRLEPrediction(",
    ):
        assert name not in new_inference, name
    assert "class Sam2SegmentationPrediction(" not in new_sam2
    TARGET.write_text(module, encoding="utf-8")
    INFERENCE.write_text(new_inference, encoding="utf-8")
    SAM2.write_text(new_sam2, encoding="utf-8")
    for path in (TARGET, INFERENCE, SAM2):
        print(f"{path}: {len(path.read_text(encoding='utf-8').splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
