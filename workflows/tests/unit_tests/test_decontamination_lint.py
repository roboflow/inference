"""Locks in Workflows decontamination progress.

`roboflow_workflows` must not import the `inference` server package. This
scans every `*.py` under `workflows/roboflow_workflows` for `inference.*`
imports (server side of the compat boundary), including relative imports and
function-local imports, and asserts there are none left.

This test is necessary but not sufficient: the string scan only sees
single-line quoted `from|import inference...` literals, so a triple-quoted
multi-line code template is invisible to it, and so is
`importlib.import_module(<computed name>)`. The Phase 13 isolation probe is
the backstop for what this lint misses.
"""

import ast
import re
from pathlib import Path
from typing import Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_ROOT = PROJECT_ROOT / "roboflow_workflows"
assert WORKFLOWS_ROOT.is_dir(), f"missing package source: {WORKFLOWS_ROOT}"
assert any(WORKFLOWS_ROOT.rglob("*.py")), f"empty package source: {WORKFLOWS_ROOT}"

# `inference_models` and `inference_sdk` are separately published
# distributions, not part of the server package - they are allowed.
ALLOWED_PREFIX = "roboflow_workflows"

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
    to the same string as an absolute-form violation.
    """
    pkg_parts = path.relative_to(PROJECT_ROOT).with_suffix("").parts[:-1]
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
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        for module in _imported_modules(ast.parse(source, filename=str(path)), path):
            if _is_forbidden(module):
                violations.add((relative, module))
        for module in _STRING_IMPORT.findall(source):
            if _is_forbidden(module):
                violations.add((relative, f"{module} (exec'd string)"))
    return violations


def test_no_inference_imports_in_workflows() -> None:
    violations = collect_violations()
    assert not violations, (
        "`roboflow_workflows` must not import the `inference` server "
        "package:\n  " + "\n  ".join(f"{p} -> {m}" for p, m in sorted(violations))
    )
