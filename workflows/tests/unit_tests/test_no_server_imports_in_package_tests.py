"""Package-side lint: no test/fixture/helper under workflows/tests/ may
resolve a symbol from the `inference` server package. The standalone job
installs no server; a server import here would either fail collection or
leak coverage into a suite meant to prove roboflow-workflows works alone.

Detects (via AST):

* `import inference...` / `from inference... import ...`
* `importlib.import_module("inference...")`
* `mock.patch("inference...")`, `patch("inference...")`, `patch.object` first arg
* `sys.modules["inference..."]` subscript

Deliberate negative assertions on module NAMES (e.g. `assert "inference.core.roboflow_api" not in sys.modules`) survive because their string is not a load target - it never reaches the four call/subscript sites above. `inference_sdk` and `inference_models` are separate distributions and are allowed.

Scope of string resolution: module-level `NAME = "inference..."` and
`Constant + Constant` concatenation are folded when the result is the FIRST
POSITIONAL of a load call (so `patch(MODULE)` and `patch("inference." +
"core")` are caught). Anything more (function-scope binds, f-strings,
attribute lookups, list joins) is deliberately out of scope: the lint is a
guard, not an abstract interpreter. Ambiguous computed targets fall through
silently; the isolation probe's runtime ServerImportBlocker is the belt to
this lint's braces.
"""

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TESTS_ROOT.parent
SELF_PATH = Path(__file__).resolve()

_LOAD_CALL_TARGETS = {
    ("importlib", "import_module"),
    ("mock", "patch"),
    ("unittest.mock", "patch"),
    ("patch",),
}


def _is_forbidden_module_name(name: str) -> bool:
    return name == "inference" or name.startswith("inference.")


def _callable_dotted_name(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _is_load_call(callable_name: str) -> bool:
    if callable_name.endswith(".import_module") and callable_name.startswith(
        "importlib"
    ):
        return True
    if callable_name == "importlib.import_module":
        return True
    if callable_name in {"mock.patch", "unittest.mock.patch", "patch"}:
        return True
    if callable_name.endswith(".patch") and any(
        part in callable_name for part in ("mock", "unittest.mock")
    ):
        return True
    if callable_name in {
        "patch.object",
        "mock.patch.object",
        "unittest.mock.patch.object",
    }:
        return True
    return False


def _module_string_constants(tree: ast.AST) -> dict:
    """Collect module-level `NAME = "..."` string bindings.

    Only assignments at the top level of the file are recorded; function-
    or class-scope bindings are intentionally ignored. If the same name is
    rebound, the last binding wins (matches Python's actual behaviour).
    """
    binds: dict = {}
    for node in tree.body if isinstance(tree, ast.Module) else ():
        if not isinstance(node, ast.Assign):
            continue
        folded = _fold_string(node.value, {})
        if folded is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                binds[target.id] = folded
    return binds


def _fold_string(node: ast.AST, binds: dict) -> str | None:
    """Return the string value of `node` if it is a literal, a module-level
    string bind, or a `Constant + Constant` concatenation of those. Return
    None for anything else - the lint refuses to guess at computed targets.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return binds.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _fold_string(node.left, binds)
        right = _fold_string(node.right, binds)
        if left is not None and right is not None:
            return left + right
    return None


def _string_arg(call: ast.Call, index: int, binds: dict) -> str:
    if index < len(call.args):
        value = _fold_string(call.args[index], binds)
        if value is not None:
            return value
    return ""


def _forbidden_from_call(
    node: ast.Call, violations: list, source: str, binds: dict
) -> None:
    callable_name = _callable_dotted_name(node.func)
    if not callable_name:
        return
    if _is_load_call(callable_name):
        target = _string_arg(node, 0, binds)
        if _is_forbidden_module_name(target):
            violations.append((source, f"{callable_name}({target!r})"))
    if callable_name.endswith(".patch.object") or callable_name == "patch.object":
        # patch.object(target, attribute) - target is an object, not a name;
        # only flag if the first positional is a literal server module string.
        target = _string_arg(node, 0, binds)
        if _is_forbidden_module_name(target):
            violations.append((source, f"{callable_name}({target!r}, ...)"))


def _forbidden_from_subscript(
    node: ast.Subscript, violations: list, source: str
) -> None:
    value = node.value
    if not (isinstance(value, ast.Attribute) and value.attr == "modules"):
        return
    slice_node = node.slice
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        if _is_forbidden_module_name(slice_node.value):
            violations.append((source, f"sys.modules[{slice_node.value!r}]"))


def test_no_test_file_loads_the_server_package() -> None:
    files = sorted(TESTS_ROOT.rglob("*.py"))
    assert files, f"no test files found under {TESTS_ROOT}"
    violations = []
    for path in files:
        if path.resolve() == SELF_PATH:
            continue  # this scanner names forbidden modules for regex/AST use
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as error:
            violations.append(
                (str(path.relative_to(PROJECT_ROOT)), f"syntax error: {error}")
            )
            continue
        source = str(path.relative_to(PROJECT_ROOT))
        binds = _module_string_constants(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden_module_name(alias.name):
                        violations.append((source, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if _is_forbidden_module_name(node.module):
                    violations.append((source, f"from {node.module}"))
            elif isinstance(node, ast.Call):
                _forbidden_from_call(node, violations, source, binds)
            elif isinstance(node, ast.Subscript):
                _forbidden_from_subscript(node, violations, source)
    assert not violations, violations
