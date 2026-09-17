"""The inventory of engine construction sites outside workflows.

This is the ONE inventory guard: it *discovers* `ExecutionEngine.init` sites
across the production tree (server, CLI, and the two maintained direct-caller
scripts) rather than trusting a hard-coded list, so a new root in a new file
cannot go unnoticed - it fails the count below.

What each site binds is proven at runtime, not from source text: the four
server/CLI roots in `test_image_codec_binding.py` (codec, configuration,
observer, platform objects, models provider, step error handler), the two
scripts in `test_direct_caller_bindings.py`.
"""

import ast
from pathlib import Path

# tests/inference/unit_tests/core/interfaces/<file> -> parents[5] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[5]
SEARCH_ROOTS = ("inference", "inference_cli", "development", "examples")
# The engine itself constructs engines (nested workflows, tests of the engine);
# only *server* construction sites are composition roots.
EXCLUDED_SUBTREE = REPO_ROOT / "inference" / "core" / "workflows"

EXPECTED_ROOTS = {
    "inference/core/interfaces/http/http_api.py": 2,
    "inference/core/interfaces/stream/inference_pipeline.py": 1,
    "inference_cli/lib/workflows/local_image_adapter.py": 1,
    "development/stream_interface/benchmark_engine_throughput.py": 1,
    "examples/run_perspective_correction.py": 1,
}


def _python_files():
    for root in SEARCH_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if EXCLUDED_SUBTREE in path.parents:
                continue
            yield path


def _is_engine_init(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "init"
        and getattr(node.func.value, "id", None) == "ExecutionEngine"
    )


def _discovered_engine_init_counts() -> dict:
    counts = {}
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        n = sum(1 for node in ast.walk(tree) if _is_engine_init(node))
        if n:
            counts[str(path.relative_to(REPO_ROOT))] = n
    return counts


def test_the_set_of_composition_roots_is_exactly_the_expected_one() -> None:
    assert _discovered_engine_init_counts() == EXPECTED_ROOTS
