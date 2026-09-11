"""Move the SAM prompt classes into workflows; the server re-exports them (Task 11.13).

Creates `inference/core/workflows/core_steps/models/foundation/segment_anything_common/prompts.py`
from two spans copied character for character - `class Box` .. `to_sam2_inputs`
of `inference/core/entities/requests/sam2.py` and `class Sam3Prompt` ..
`_validate_output_prob_thresh` of `requests/sam3.py` - then replaces each span
in its server module with a re-export import, and trims the typing/pydantic
names only those spans used (`Tuple`, `BaseModel` in sam2.py; `Union`,
`BaseModel` in sam3.py). Content-anchored (never line numbers), re-parses every
output before writing, and idempotent: a second run prints SKIP and changes
nothing. A half-applied state (one of the three files already edited) aborts
with a message instead of guessing.

Run:  python scripts/phase11_move_sam_prompts.py [<repo root>]
"""

import ast
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
TARGET = (
    ROOT
    / "inference/core/workflows/core_steps/models/foundation/segment_anything_common/prompts.py"
)
SAM2 = ROOT / "inference/core/entities/requests/sam2.py"
SAM3 = ROOT / "inference/core/entities/requests/sam3.py"
PROMPTS_MODULE = "inference.core.workflows.core_steps.models.foundation.segment_anything_common.prompts"
SAM2_IMPORT = (
    f"from {PROMPTS_MODULE} import (  # noqa: F401\n"
    "    Box,\n    Point,\n    Sam2Prompt,\n    Sam2PromptSet,\n)\n"
)
SAM3_IMPORT = f"from {PROMPTS_MODULE} import (  # noqa: F401\n    Sam3Prompt,\n)\n"
SAM2_ANCHOR = "from inference.core.env import SAM2_VERSION_ID\n"
SAM3_ANCHOR = "from inference.core.env import SAM3_MAX_PROMPT_BATCH_SIZE\n"
HEADER = '''"""SAM prompt value objects, owned by Workflows.

`Box`, `Point`, `Sam2Prompt`, `Sam2PromptSet` (from `inference/core/entities/
requests/sam2.py`) and `Sam3Prompt` (from `requests/sam3.py`) were MOVED here
verbatim; both server modules now re-export them, so there is exactly ONE class
object per name. That is what keeps `isinstance(raw_point, Point)` in the two
SAM 3 interactive blocks true for a `Point` built through either import path,
lets the server request classes accept prompt sets the blocks build, and keeps
the `ValidationError` a bad prompt raises the same object either way.

Pinned by `tests/workflows/unit_tests/core_steps/models/foundation/test_sam_prompts.py`
(identity through the re-export, the accepted-input matrix of `_as_sam2_points`,
frozen `to_sam2_inputs()` / payload / validation tables).
"""

from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator


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
    lines[matches[0]] = prefix + " ".join(", ".join(kept).split())
    return "\n".join(lines)


def main() -> int:
    sam2, sam3 = SAM2.read_text(encoding="utf-8"), SAM3.read_text(encoding="utf-8")
    applied = (SAM2_IMPORT in sam2, SAM3_IMPORT in sam3, TARGET.exists())
    if all(applied):
        print("SKIP (already applied)")
        return 0
    if any(applied):
        raise SystemExit(
            f"half-applied state {applied}; restore the three files from git first"
        )
    sam2_span = _span(sam2, "class Box(BaseModel):", "class Sam2SegmentationRequest(")
    sam3_span = _span(
        sam3, "class Sam3Prompt(BaseModel):", "class Sam3InferenceRequest("
    )
    module = HEADER + sam2_span.rstrip() + "\n\n\n" + sam3_span.rstrip() + "\n"
    ast.parse(module)
    new_sam2 = sam2.replace(sam2_span, "", 1)
    assert new_sam2.count(SAM2_ANCHOR) == 1
    new_sam2 = new_sam2.replace(SAM2_ANCHOR, SAM2_ANCHOR + SAM2_IMPORT, 1)
    new_sam2 = _drop_names(new_sam2, "from typing import ", ["Tuple"])
    new_sam2 = _drop_names(new_sam2, "from pydantic import ", ["BaseModel"])
    new_sam3 = sam3.replace(sam3_span, "", 1)
    assert new_sam3.count(SAM3_ANCHOR) == 1
    new_sam3 = new_sam3.replace(SAM3_ANCHOR, SAM3_ANCHOR + SAM3_IMPORT, 1)
    new_sam3 = _drop_names(new_sam3, "from typing import ", ["Union"])
    new_sam3 = _drop_names(new_sam3, "from pydantic import ", ["BaseModel"])
    for text in (new_sam2, new_sam3):
        ast.parse(text)
        for name in (
            "class Box(",
            "class Point(",
            "class Sam2Prompt(",
            "class Sam2PromptSet(",
            "class Sam3Prompt(",
        ):
            assert name not in text, name
    TARGET.write_text(module, encoding="utf-8")
    SAM2.write_text(new_sam2, encoding="utf-8")
    SAM3.write_text(new_sam3, encoding="utf-8")
    for path in (TARGET, SAM2, SAM3):
        print(f"{path}: {len(path.read_text(encoding='utf-8').splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
