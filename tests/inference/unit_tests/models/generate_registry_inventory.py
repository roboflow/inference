"""Regenerate the registry inventory from the last eager implementation.

Run from the repository root:
    python tests/inference/unit_tests/models/generate_registry_inventory.py

The source is read with git show, never from the current registry. Model imports
become controlled classes, while the original conditionals and registration
statements execute unchanged. No optional model dependencies are needed.
"""

import ast
import json
import subprocess
import warnings
from functools import lru_cache
from pathlib import Path

SOURCE_REVISION = "fdd13b32c6f55f91d0a800277564bb40067c533b"
_INVENTORY = Path(__file__).with_name("registry_inventory.json")


@lru_cache
def _source(path: str) -> str:
    source = subprocess.check_output(
        ["git", "show", f"{SOURCE_REVISION}:{path}"], text=True
    )
    return source


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


def _original_registry(*, flags, use_adapters, use_proxy):
    """Execute historical registration with controlled imports."""
    model_tree = ast.parse(_source("inference/models/__init__.py"))
    core_models = next(
        node.value
        for node in model_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CORE_MODELS"
            for target in node.targets
        )
    )
    optional_models = next(
        ast.literal_eval(node.value)
        for node in model_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "OPTIONAL_MODELS"
            for target in node.targets
        )
    )
    core_flags = {
        key.value: value.elts[1].id
        for key, value in zip(core_models.keys, core_models.values)
    }
    classes = {}

    def _symbol(module, name):
        if module == "inference.core.env":
            return {**flags, "USE_INFERENCE_MODELS": use_adapters, "API_KEY": None}[
                name
            ]
        if module == "inference.models.vllm_proxy":
            return use_proxy
        if name == "LOCAL_INFERENCE_MODELS_MODEL_TYPE":
            return "inference-models-local"
        if module == "inference.core.warnings":
            return Warning
        if module == "inference.models" and name in core_flags:
            available = (
                flags[core_flags[name]]
                if flags["CORE_MODELS_ENABLED"]
                else name in optional_models
            )
            if not available:
                raise ImportError(f"Disabled core model: {name}")

        path = f"{module}:{name}"
        model_class = classes.setdefault(path, type(name, (), {"registry_path": path}))
        return model_class

    tree = _ControlledImports().visit(ast.parse(_source("inference/models/utils.py")))
    ast.fix_missing_locations(tree)
    namespace = {"_symbol": _symbol}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(tree, "historical_registry", "exec"), namespace)
    registry = namespace["ROBOFLOW_MODEL_TYPES"]
    paths = {key: value.registry_path for key, value in registry.items()}
    return paths


def _generate_inventory(flags):
    cases = {}
    for use_adapters in (False, True):
        for use_proxy in (False, True):
            registry = _original_registry(
                flags=flags, use_adapters=use_adapters, use_proxy=use_proxy
            )
            grouped = {}
            for key, path in sorted(registry.items()):
                grouped.setdefault(path, []).append(list(key))
            cases[f"adapters={use_adapters},proxy={use_proxy}"] = grouped
    inventory = {"source_revision": SOURCE_REVISION, "cases": cases, "flags": flags}
    return inventory


if __name__ == "__main__":
    flags = json.loads(_INVENTORY.read_text())["flags"]
    _INVENTORY.write_text(json.dumps(_generate_inventory(flags), indent=2) + "\n")
