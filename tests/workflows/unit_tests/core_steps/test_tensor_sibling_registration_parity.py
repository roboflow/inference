"""Structural parity checks for tensor-native block siblings.

Under ``ENABLE_TENSOR_DATA_REPRESENTATION`` the loader swaps each block's
numpy ``vN.py`` for its ``vN_tensor.py`` sibling. A sibling that exists on
disk but is not imported in the **flag-on** branch of ``loader.py`` is
silently never loaded, so the tensor path would keep executing the numpy
implementation. These tests pin that structural invariant with an AST-only
walk (no block imports), so they run in any CI job without pulling in the
full model dependency set.

Scope: file existence + import-module-string parity. This does NOT pin
``describe_outputs()`` / manifest / runtime semantic parity between
siblings — that is a separate, larger invariant.
"""

import ast
from pathlib import Path
from typing import Dict, List, Set

CORE_STEPS_ROOT = Path(__file__).parents[4] / "inference/core/workflows/core_steps"
LOADER_PATH = CORE_STEPS_ROOT / "loader.py"
MODULE_PREFIX = "inference.core.workflows.core_steps."

_OPPOSITE_BRANCH = {"numpy": "tensor", "tensor": "numpy"}


def _module_name(path: Path) -> str:
    relative = path.relative_to(CORE_STEPS_ROOT)
    return MODULE_PREFIX + ".".join(relative.with_suffix("").parts)


def _flag_branch(test: ast.expr) -> str:
    """``numpy`` for a negated flag check, ``tensor`` for a bare one, else ``""``."""
    if (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and isinstance(test.operand, ast.Name)
        and test.operand.id == "ENABLE_TENSOR_DATA_REPRESENTATION"
    ):
        return "numpy"
    if isinstance(test, ast.Name) and test.id == "ENABLE_TENSOR_DATA_REPRESENTATION":
        return "tensor"
    return ""


def _import_branches(tree: ast.Module) -> Dict[str, Set[str]]:
    """Map a module's imports to their ``ENABLE_TENSOR_DATA_REPRESENTATION`` branch.

    The loader uses two shapes: ``if not ENABLE_TENSOR_DATA_REPRESENTATION:
    ... else: ...`` and ``if ENABLE_TENSOR_DATA_REPRESENTATION: ... else:
    ...``. Whichever branch tests the flag directly, its ``orelse`` is the
    opposite branch — not the branch this ``if`` itself was nested in.
    """
    numpy_branch: Set[str] = set()
    tensor_branch: Set[str] = set()
    unconditional: Set[str] = set()

    def _walk(statements: List[ast.stmt], branch: str) -> None:
        for statement in statements:
            if isinstance(statement, ast.ImportFrom) and statement.module:
                if branch == "numpy":
                    numpy_branch.add(statement.module)
                elif branch == "tensor":
                    tensor_branch.add(statement.module)
                else:
                    unconditional.add(statement.module)
            elif isinstance(statement, ast.If):
                test_branch = _flag_branch(statement.test)
                _walk(statement.body, test_branch or branch)
                _walk(statement.orelse, _OPPOSITE_BRANCH.get(test_branch, branch))

    _walk(tree.body, "")

    return {
        "numpy": numpy_branch,
        "tensor": tensor_branch,
        "unconditional": unconditional,
    }


def _loader_import_branches() -> Dict[str, Set[str]]:
    return _import_branches(ast.parse(LOADER_PATH.read_text()))


def test_every_tensor_sibling_has_a_numpy_sibling() -> None:
    # Invariant: every tensor-native block has a numpy fallback. The loader
    # swaps vN.py → vN_tensor.py under ENABLE_TENSOR_DATA_REPRESENTATION, so
    # a tensor sibling without a numpy counterpart would break the flag-off
    # path. This test prevents silent orphaning when a new tensor sibling is
    # added without its numpy base.
    tensor_files = sorted(CORE_STEPS_ROOT.rglob("v*_tensor.py"))
    assert tensor_files, "expected tensor-native siblings under core_steps"

    orphans = [
        str(path)
        for path in tensor_files
        if not path.with_suffix("").with_name(path.name.replace("_tensor", "")).exists()
    ]
    assert orphans == [], f"tensor siblings without a numpy sibling: {orphans}"


def test_every_tensor_sibling_is_registered_in_loader_flag_branch() -> None:
    """Each ``vN_tensor.py`` must be imported ONLY in the flag-on branch.

    A tensor sibling imported unconditionally or in the ``if not FLAG``
    (numpy) branch is the worst registration bug: the tensor implementation
    would load when the flag is off, or the numpy one when the flag is on.
    """
    branches = _loader_import_branches()

    missing = []
    contaminated = []
    for path in sorted(CORE_STEPS_ROOT.rglob("v*_tensor.py")):
        module = _module_name(path)
        if module not in branches["tensor"]:
            missing.append(module)
        if module in branches["numpy"] or module in branches["unconditional"]:
            contaminated.append(module)

    assert missing == [], (
        "tensor-native siblings are invisible until loader.py imports them "
        f"under the ENABLE_TENSOR_DATA_REPRESENTATION branch: {missing}"
    )
    assert contaminated == [], (
        "tensor-native siblings must NOT be imported in the numpy or "
        f"unconditional branch of loader.py: {contaminated}"
    )


def test_loader_import_branches_rejects_tensor_module_in_numpy_branch() -> None:
    """Negative fixture: a tensor module only in the numpy branch must fail
    the flag-on check. Proves the walker is not silently permissive.
    """
    fake_loader = ast.parse(
        "from inference.core.workflows.core_steps.models.roboflow."
        "object_detection.v3_tensor import ObjectDetectionBlockV3\n"
        "if not ENABLE_TENSOR_DATA_REPRESENTATION:\n"
        "    from inference.core.workflows.core_steps.models.roboflow."
        "object_detection.v3_tensor import ObjectDetectionBlockV3\n"
    )
    branches = _import_branches(fake_loader)

    tensor_module = (
        "inference.core.workflows.core_steps.models.roboflow."
        "object_detection.v3_tensor"
    )
    assert tensor_module not in branches["tensor"]
    assert tensor_module in branches["numpy"]


def test_loader_import_branches_labels_bare_flag_else_as_numpy() -> None:
    """Regression fixture: ``if FLAG: ... else: ...`` (no leading ``not``)
    must label its ``else`` as the numpy branch, not whatever branch the
    ``if`` itself was nested in. A walker that inherited the outer branch
    here would let a tensor sibling's numpy fallback go unregistered
    whenever loader.py uses this shape instead of ``if not FLAG``.
    """
    fake_loader = ast.parse(
        "if ENABLE_TENSOR_DATA_REPRESENTATION:\n"
        "    from pkg.tensor_impl import Block\n"
        "else:\n"
        "    from pkg.numpy_impl import Block\n"
    )
    branches = _import_branches(fake_loader)

    assert "pkg.tensor_impl" in branches["tensor"]
    assert "pkg.numpy_impl" in branches["numpy"]
    assert "pkg.numpy_impl" not in branches["unconditional"]


def test_numpy_siblings_of_tensor_blocks_are_registered_without_flag() -> None:
    branches = _loader_import_branches()

    missing = []
    for path in sorted(CORE_STEPS_ROOT.rglob("v*_tensor.py")):
        numpy_path = path.with_name(path.name.replace("_tensor", ""))
        module = _module_name(numpy_path)
        registered_without_flag = (
            module in branches["numpy"] or module in branches["unconditional"]
        )
        if not registered_without_flag:
            missing.append(module)

    assert missing == [], (
        "numpy siblings must stay importable when "
        f"ENABLE_TENSOR_DATA_REPRESENTATION is off: {missing}"
    )
