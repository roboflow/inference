"""AST-level guard: no call under `execution_engine/` passes the three legacy
`usage_*` keyword arguments any more.

A natural-looking grep for `usage_collector`/`usage_tracking` cannot be used
here: both SAM `v1_tensor.py` siblings mention `usage_collector` in prose
comments explaining why they are *not* metered. `usage_workflow_id`
legitimately survives as a local variable in `v1/core.py`, so only the
*keyword argument* form can be checked, not the name.
"""

import ast
from pathlib import Path

# tests/workflows/unit_tests/<this file> -> three levels up is the repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "roboflow_workflows" / "execution_engine"
assert ENGINE_ROOT.is_dir(), f"missing engine source: {ENGINE_ROOT}"
assert any(ENGINE_ROOT.rglob("*.py")), f"empty engine source: {ENGINE_ROOT}"

REMOVED_USAGE_KEYWORDS = {"usage_fps", "usage_workflow_id", "usage_workflow_preview"}


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
                        (str(path.relative_to(PROJECT_ROOT)), node.lineno, keyword.arg)
                    )
    assert offenders == []
