"""Locks in Workflows decontamination progress.

`inference/core/workflows` must stop importing the `inference` server package
(it stays in place - see DECONTAMINATION.PLAN.MD). The remaining violations
are listed in `decontamination_baseline.txt`; this test fails if a new one
appears, and also if a listed one disappears without the baseline being
updated - so the list can only shrink.

This test is necessary but not sufficient: the string scan only sees
single-line quoted `from|import inference...` literals, so a triple-quoted
multi-line code template is invisible to it, and so is
`importlib.import_module(<computed name>)`. The Phase 13 isolation probe is
the backstop for what this lint misses.

Create the baseline the first time, and regenerate it after removing
violations, with the same command:
    UPDATE_DECONTAMINATION_BASELINE=1 pytest \
        tests/workflows/unit_tests/test_decontamination_lint.py

Once a baseline exists, the regeneration path REFUSES to write when new
violations are present, so it cannot be used to bless a regression. Never
delete the baseline to "reset" it - regenerating from a missing baseline
recreates it from whatever is present, silently absorbing any regression.
"""

import ast
import os
import re
from pathlib import Path
from typing import Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_ROOT = REPO_ROOT / "inference" / "core" / "workflows"
BASELINE_PATH = Path(__file__).parent / "decontamination_baseline.txt"

# `inference_models` and `inference_sdk` are separately published
# distributions, not part of the server package - they are allowed.
ALLOWED_PREFIX = "inference.core.workflows"

# Imports hidden inside string literals that are later exec()'d into
# dynamically assembled blocks. `ast` sees a string, not an import, so they
# have to be matched textually. This is not a corner case:
# `dynamic_blocks/block_scaffolding.py:92` smuggles
# `"from inference.core.env import WORKFLOWS_IMAGE_TENSOR_DEVICE"` this way,
# and `modal/modal_app.py` mirrors the same list into the Modal sandbox.
_STRING_IMPORT = re.compile(
    # The negative lookahead is load-bearing: without it, the string
    # "from inference_models.models.base..." matches the bare `inference`
    # alternative and is reported as a forbidden import. Verified: it produced
    # a false positive on block_scaffolding.py lines 93-96.
    r"""["'](?:from|import)\s+(inference(?![A-Za-z0-9_])(?:\.[A-Za-z0-9_]+)*)"""
)


def _is_forbidden(module: str) -> bool:
    if module == "inference":  # bare `import inference` pulls the package in
        return True
    if not module.startswith("inference."):
        return False
    return module != ALLOWED_PREFIX and not module.startswith(ALLOWED_PREFIX + ".")


def _resolve_relative(module: str, level: int, path: Path) -> str:
    """Turn a relative import into its absolute dotted path.

    A `from ...core.env import X` inside workflows climbs out of the package
    and is exactly as contaminating as the absolute form, so it must resolve
    to the same string the baseline records.
    """
    pkg_parts = path.relative_to(REPO_ROOT).with_suffix("").parts[:-1]
    base = pkg_parts[: len(pkg_parts) - (level - 1)] if level > 1 else pkg_parts
    return ".".join([*base, *(module.split(".") if module else [])])


def _imported_modules(tree: ast.AST, path: Path) -> Set[str]:
    # ast.walk (not tree.body) so function-local imports are caught too -
    # `dynamic_blocks/modal_executor.py` hides four of them.
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    modules.add(node.module)
            else:
                modules.add(_resolve_relative(node.module or "", node.level, path))
    return modules


def collect_violations() -> Set[Tuple[str, str]]:
    violations = set()
    for path in sorted(WORKFLOWS_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPO_ROOT).as_posix()
        for module in _imported_modules(ast.parse(source, filename=str(path)), path):
            if _is_forbidden(module):
                violations.add((relative, module))
        for module in _STRING_IMPORT.findall(source):
            if _is_forbidden(module):
                violations.add((relative, f"{module} (exec'd string)"))
    return violations


def _updating() -> bool:
    return os.getenv("UPDATE_DECONTAMINATION_BASELINE", "").strip().lower() in {
        "1",
        "true",
    }


def _read_baseline() -> Optional[Set[Tuple[str, str]]]:
    # `None` (absent) is distinct from an empty set (nothing left to remove):
    # an absent baseline must be creatable, an empty one must be enforced.
    if not BASELINE_PATH.exists():
        return None
    declared_count: Optional[int] = None
    entries = set()
    for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            count_match = re.match(r"#\s*Count:\s*(\d+)", line)
            if count_match:
                declared_count = int(count_match.group(1))
            continue
        fields = line.split("\t")
        assert (
            len(fields) == 2
        ), f"Malformed baseline row (expected 'path<TAB>module'): {line!r}"
        path, module = fields
        entries.add((path, module))
    if declared_count is not None and not _updating():
        assert declared_count == len(entries), (
            f"{BASELINE_PATH} header declares Count: {declared_count} but "
            f"{len(entries)} rows were parsed - regenerate with "
            "UPDATE_DECONTAMINATION_BASELINE=1"
        )
    return entries


def _write_baseline(violations: Set[Tuple[str, str]]) -> None:
    lines = [f"{path}\t{module}" for path, module in sorted(violations)]
    header = (
        "# Remaining `inference.*` imports inside inference/core/workflows.\n"
        "# This list may only shrink. See DECONTAMINATION.PLAN.MD.\n"
        f"# Count: {len(lines)}\n"
    )
    BASELINE_PATH.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def test_no_new_inference_imports_in_workflows() -> None:
    actual = collect_violations()
    baseline = _read_baseline()
    updating = _updating()
    if baseline is None:
        # First run: nothing to compare against, so creation is the only
        # sensible action - but only when asked for explicitly, so a deleted
        # baseline cannot be silently re-blessed by an ordinary test run.
        assert updating, (
            f"{BASELINE_PATH} is missing. Create it with "
            "UPDATE_DECONTAMINATION_BASELINE=1 and commit it."
        )
        _write_baseline(actual)
        return
    added = sorted(actual - baseline)
    removed = sorted(baseline - actual)
    if updating:
        # Regeneration is for recording progress, never for absorbing a
        # regression - refuse to write while anything new is present.
        assert not added, (
            "Refusing to regenerate the baseline: these are NEW violations, "
            "not resolved ones:\n  " + "\n  ".join(f"{p} -> {m}" for p, m in added)
        )
        _write_baseline(actual)
        return
    assert (
        not added
    ), "New `inference.*` imports inside inference/core/workflows:\n  " + "\n  ".join(
        f"{p} -> {m}" for p, m in added
    )
    assert not removed, (
        "These baseline entries are gone - good. Regenerate the baseline with "
        "UPDATE_DECONTAMINATION_BASELINE=1 and commit it:\n  "
        + "\n  ".join(f"{p} -> {m}" for p, m in removed)
    )
