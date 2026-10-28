"""Regenerate the reviewed inventory from current registry registration source.

Run from the repository root:
    python tests/inference/unit_tests/models/generate_registry_inventory.py

Only registration metadata executes: model imports become controlled symbols and
lazy references remain paths. Neither optional dependencies nor git history are
needed. Review the generated JSON diff together with model-registration changes.
"""

import ast
import json
from dataclasses import dataclass
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[4] / "inference/models/utils.py"
_INVENTORY = Path(__file__).with_name("registry_inventory.json")
REGENERATE_COMMAND = (
    "python tests/inference/unit_tests/models/generate_registry_inventory.py"
)


@dataclass
class _ModelReference:
    path: str
    optional: bool = False
    warning_message: str = None
    warning_category: type = Warning

    def __post_init__(self):
        if not isinstance(self.path, str) or self.path.count(":") != 1:
            raise ValueError(f"Expected a module:class reference, got {self.path!r}")


class _RegistrationRegistry(dict):
    @property
    def _entries(self):
        return self

    def set_adapter(self, key, adapter):
        if not isinstance(adapter, _ModelReference):
            raise TypeError("Adapter registration must use a lazy model reference")
        self[key] = adapter


class _ControlledImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if not node.module.startswith("inference."):
            return node

        statements = [
            ast.Assign(
                targets=[ast.Name(id=alias.asname or alias.name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="_symbol", ctx=ast.Load()),
                    args=[ast.Constant(node.module), ast.Constant(alias.name)],
                    keywords=[],
                ),
            )
            for alias in node.names
        ]
        return statements


def _coverage_flags(source_path):
    flags = {
        alias.name: alias.name
        not in {"CORE_MODEL_OWLV2_ENABLED", "SAM3_3D_OBJECTS_ENABLED"}
        for node in ast.walk(ast.parse(source_path.read_text()))
        if isinstance(node, ast.ImportFrom) and node.module == "inference.core.env"
        for alias in node.names
        if alias.name.endswith("_ENABLED")
    }
    return flags


def _registry_paths(*, flags, use_adapters, use_proxy, source_path=_SOURCE):
    """Evaluate current registration with every declared implementation available."""

    def _symbol(module, name):
        if module == "inference.core.env":
            return {**flags, "USE_INFERENCE_MODELS": use_adapters, "API_KEY": None}[
                name
            ]
        if module == "inference.core.registries.lazy":
            return {
                "_LazyModelClass": _ModelReference,
                "_LazyModelRegistry": _RegistrationRegistry,
            }[name]
        if module == "inference.models.vllm_proxy":
            return use_proxy
        if name == "LOCAL_INFERENCE_MODELS_MODEL_TYPE":
            return "inference-models-local"
        if module == "inference.core.warnings":
            return Warning

        return type(name, (), {})

    tree = _ControlledImports().visit(ast.parse(source_path.read_text()))
    ast.fix_missing_locations(tree)
    namespace = {"_symbol": _symbol}
    exec(compile(tree, str(source_path), "exec"), namespace)
    registry = namespace["ROBOFLOW_MODEL_TYPES"]
    paths = {}
    for key, reference in registry.items():
        if not isinstance(reference, _ModelReference):
            raise TypeError(f"Registry key {key!r} must use a lazy model reference")
        paths[key] = reference.path
    return paths


def _generate_inventory(flags=None, *, source_path=_SOURCE):
    current_flags = _coverage_flags(source_path)
    if flags is not None:
        current_flags.update({key: flags[key] for key in current_flags if key in flags})
    cases = {}
    for use_adapters in (False, True):
        for use_proxy in (False, True):
            registry = _registry_paths(
                flags=current_flags,
                use_adapters=use_adapters,
                use_proxy=use_proxy,
                source_path=source_path,
            )
            grouped = {}
            for key, path in sorted(registry.items()):
                grouped.setdefault(path, []).append(list(key))
            cases[f"adapters={use_adapters},proxy={use_proxy}"] = grouped
    inventory = {
        "source": "inference/models/utils.py",
        "cases": cases,
        "flags": current_flags,
    }
    return inventory


if __name__ == "__main__":
    _INVENTORY.write_text(json.dumps(_generate_inventory(), indent=2) + "\n")
