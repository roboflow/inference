"""What Phase 6 removed, asserted by row and by AST - never by grep.

Two natural-looking greps cannot return zero and would make a false gate: both
SAM `v1_tensor.py` siblings mention `usage_collector` in prose comments
explaining why they are *not* metered. So the statements below are (a) these
exact eight `(path, module)` pairs are absent from the lint baseline, and (b)
no call under `execution_engine/` passes the three `usage_*` keyword arguments
any more - `usage_workflow_id` legitimately survives as a local variable in
`v1/core.py`, so only the *keyword* can be checked.

This file deliberately asserts **nothing about rows other phases own**. An
earlier draft required `executor/core.py`'s `inference.core.env` row to be
present; Phase 5 removes that row, so the assertion would have made one phase
order or the other fail. "Nothing else was swept up" is checked instead by the
shared baseline-diff checker in Global Constraints, which compares against this
phase's own start commit and is therefore order-independent.
"""

import ast
from pathlib import Path

import pytest

# tests/workflows/unit_tests/<this file> -> three levels up is the repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE = (
    REPO_ROOT / "tests" / "workflows" / "unit_tests" / "decontamination_baseline.txt"
)
ENGINE_ROOT = REPO_ROOT / "inference" / "core" / "workflows" / "execution_engine"

ROWS_OWNED_BY_PHASE_6 = [
    (
        "inference/core/workflows/core_steps/models/foundation/segment_anything2_video/v1.py",
        "inference.usage_tracking.collector",
    ),
    (
        "inference/core/workflows/core_steps/models/foundation/segment_anything3_video/v1.py",
        "inference.usage_tracking.collector",
    ),
    (
        "inference/core/workflows/execution_engine/v1/dynamic_blocks/block_scaffolding.py",
        "inference.usage_tracking.block_execution",
    ),
    (
        "inference/core/workflows/execution_engine/v1/dynamic_blocks/block_scaffolding.py",
        "inference.usage_tracking.collector",
    ),
    (
        "inference/core/workflows/execution_engine/v1/dynamic_blocks/modal_executor.py",
        "inference.usage_tracking.block_execution",
    ),
    (
        "inference/core/workflows/execution_engine/v1/executor/core.py",
        "inference.core.telemetry",
    ),
    (
        "inference/core/workflows/execution_engine/v1/executor/core.py",
        "inference.usage_tracking.collector",
    ),
    (
        "inference/core/workflows/execution_engine/v1/executor/core.py",
        "inference.usage_tracking.stream_session",
    ),
]

REMOVED_USAGE_KEYWORDS = {"usage_fps", "usage_workflow_id", "usage_workflow_preview"}


def _baseline_rows() -> set:
    return set(BASELINE.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize("path, module", ROWS_OWNED_BY_PHASE_6)
def test_phase_6_row_is_gone_from_the_baseline(path: str, module: str) -> None:
    assert f"{path}\t{module}" not in _baseline_rows()


def test_the_engine_no_longer_passes_usage_keyword_arguments() -> None:
    offenders = []
    for path in sorted(ENGINE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg in REMOVED_USAGE_KEYWORDS:
                    offenders.append(
                        (str(path.relative_to(REPO_ROOT)), node.lineno, keyword.arg)
                    )
    assert offenders == []
