"""Static audit: the literal string `"CustomPythonBlockTimeout"` MUST live in
exactly one place — the definition of `MODAL_TIMEOUT_ERROR_TYPE` in
`inference/core/workflows/execution_engine/v1/dynamic_blocks/constants.py`.

Both `modal/modal_app.py` and `modal_executor.py` reference the constant by
import, never by literal. This test catches accidental string duplication in
PRs that would silently break the wire contract with the CS-237 classifier.
"""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
LITERAL = '"CustomPythonBlockTimeout"'


def _python_files_to_scan():
    """Scan `inference/` and `modal/` source, excluding tests and build output."""
    roots = [REPO_ROOT / "inference", REPO_ROOT / "modal"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            parts = set(path.parts)
            if "build" in parts or ".venv" in parts or "__pycache__" in parts:
                continue
            yield path


def test_wire_constant_literal_appears_in_one_place() -> None:
    matches = []
    for path in _python_files_to_scan():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if LITERAL in text:
            matches.append(path)

    rel_matches = [p.relative_to(REPO_ROOT) for p in matches]
    expected = Path(
        "inference/core/workflows/execution_engine/v1/dynamic_blocks/constants.py"
    )

    assert (
        rel_matches == [expected]
    ), (
        "The wire-contract literal 'CustomPythonBlockTimeout' must appear ONLY in "
        f"{expected}, but it appears in: {rel_matches}. Reference the constant "
        "MODAL_TIMEOUT_ERROR_TYPE from that module instead of using the literal."
    )
