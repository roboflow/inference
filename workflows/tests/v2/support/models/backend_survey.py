"""Source survey of the ``post_process`` keywords of the task backends.

Reads ``inference_models`` source with ``ast`` and imports nothing from it.
A backend is a class that inherits, by class name and transitively, one of
the six task base classes. For each backend's own ``post_process`` it reports::

    named         keywords in the signature
    named_unused  named keywords whose name never appears in the body
    kwargs_reads  keys read from **kwargs, also inside a self.<method> that
                  receives **kwargs (one level, same class)

Limits: a forward to a callee outside the class is reported as a read of
``"<unfollowed>"``, so a test fails rather than trusting it; dynamic access
(``getattr``, ``locals()``) is not seen.
"""

import ast
import importlib.util
import warnings
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

TASK_BASES = (
    "ObjectDetectionModel",
    "ClassificationModel",
    "MultiLabelClassificationModel",
    "KeyPointsDetectionModel",
    "InstanceSegmentationModel",
    "SemanticSegmentationModel",
)
UNFOLLOWED = "<unfollowed>"


class PostProcess(NamedTuple):
    """One backend's own ``post_process``, as written in its source."""

    qualified_name: str
    named: List[str]
    named_unused: List[str]
    kwargs_reads: List[str]


def inference_models_root() -> Path:
    """The ``inference_models`` package directory, found without importing it."""
    spec = importlib.util.find_spec("inference_models")
    root = Path(list(spec.submodule_search_locations)[0])

    return root


def survey(root: Optional[Path] = None) -> List[PostProcess]:
    """Every task backend that defines ``post_process``, in path order."""
    root = inference_models_root() if root is None else root
    classes: Dict[str, ast.ClassDef] = {}
    modules: Dict[str, str] = {}
    for path in sorted((root / "models").rglob("*.py")):
        module = ".".join(
            ("inference_models", *path.relative_to(root).with_suffix("").parts)
        )
        with warnings.catch_warnings():
            # Backend sources hold invalid escapes in plain strings.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, node)
                modules.setdefault(node.name, module)

    found = []
    for name, node in classes.items():
        if name in TASK_BASES or not _is_backend(name, classes, ()):
            continue
        methods = {
            item.name: item for item in node.body if isinstance(item, ast.FunctionDef)
        }
        post = methods.get("post_process")
        if post is None:
            continue
        named = [arg.arg for arg in post.args.args[1:] + post.args.kwonlyargs]
        found.append(
            PostProcess(
                qualified_name=f"{modules[name]}.{name}.post_process",
                named=named,
                named_unused=_named_unused(post, named),
                kwargs_reads=_kwargs_reads(post, methods),
            )
        )

    return found


def _is_backend(name: str, classes: Dict[str, ast.ClassDef], seen: tuple) -> bool:
    if name in TASK_BASES:
        return True

    node = classes.get(name)
    if node is None or name in seen:
        return False

    backend = any(
        _is_backend(base, classes, seen + (name,)) for base in _base_names(node)
    )

    return backend


def _base_names(node: ast.ClassDef) -> List[str]:
    names = []
    for base in node.bases:
        target = base.value if isinstance(base, ast.Subscript) else base
        if isinstance(target, ast.Attribute):
            names.append(target.attr)
        elif isinstance(target, ast.Name):
            names.append(target.id)

    return names


def _named_unused(function: ast.FunctionDef, named: List[str]) -> List[str]:
    used = {
        node.id
        for statement in function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    unused = [name for name in named if name not in used]

    return unused


def _kwargs_reads(
    function: ast.FunctionDef, methods: Dict[str, ast.FunctionDef], depth: int = 0
) -> List[str]:
    kwarg = function.args.kwarg.arg if function.args.kwarg else None
    if kwarg is None:
        return []

    reads = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Subscript) and _is_name(node.value, kwarg):
            if isinstance(node.slice, ast.Constant):
                reads.add(node.slice.value)
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("get", "pop")
            and _is_name(func.value, kwarg)
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            reads.add(node.args[0].value)
        if any(k.arg is None and _is_name(k.value, kwarg) for k in node.keywords):
            callee = _self_method(func, methods)
            if callee is None or depth > 0:
                reads.add(UNFOLLOWED)
            else:
                reads.update(_kwargs_reads(callee, methods, depth + 1))

    return sorted(reads)


def _self_method(
    func: ast.expr, methods: Dict[str, ast.FunctionDef]
) -> Optional[ast.FunctionDef]:
    if isinstance(func, ast.Attribute) and _is_name(func.value, "self"):
        return methods.get(func.attr)

    return None


def _is_name(node: ast.expr, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name
