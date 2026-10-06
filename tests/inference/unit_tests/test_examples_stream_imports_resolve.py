"""Stream-related `inference` imports used by shipped examples/docs must resolve.

Walks example scripts and notebooks (examples/**/*.py, examples/**/*.ipynb,
docs/**/*.ipynb) for imports of the stream/camera/stream_manager interfaces,
then imports each one to catch a rename or removal the docs were not updated
for.
"""

import ast
import importlib
import json
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

_STREAM_MODULE_PREFIXES = (
    "inference.core.interfaces.stream",
    "inference.core.interfaces.camera",
    "inference.core.interfaces.stream_manager",
    "inference.core.interfaces.webrtc_worker",
)
_TOP_LEVEL_STREAM_NAMES = ("InferencePipeline", "Stream")

_Pair = Tuple[str, Optional[str]]
# (notebook path, cell index) for cells that failed to parse but mention "inference".
_UNPARSEABLE_CELLS_WITH_INFERENCE: List[Tuple[Path, int]] = []


def _is_stream_module(module: str) -> bool:
    is_match = any(module.startswith(prefix) for prefix in _STREAM_MODULE_PREFIXES)

    return is_match


def _collect_from_tree(tree: ast.AST) -> Set[_Pair]:
    pairs: Set[_Pair] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_stream_module(alias.name):
                    pairs.add((alias.name, None))
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if _is_stream_module(node.module):
                pairs.update((node.module, alias.name) for alias in node.names)
            elif node.module == "inference":
                pairs.update(
                    (node.module, alias.name)
                    for alias in node.names
                    if alias.name in _TOP_LEVEL_STREAM_NAMES
                )

    return pairs


def _collect_from_py(path: Path) -> Set[_Pair]:
    tree = ast.parse(path.read_text(), filename=str(path))
    pairs = _collect_from_tree(tree)

    return pairs


def _collect_from_notebook(path: Path) -> Set[_Pair]:
    notebook = json.loads(path.read_text())
    pairs: Set[_Pair] = set()
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        raw_source = cell.get("source", "")
        # Notebook cell "source" is a str or a list of lines depending on tool.
        source = "".join(raw_source) if isinstance(raw_source, list) else raw_source
        code = "".join(
            line
            for line in source.splitlines(keepends=True)
            if not line.lstrip().startswith(("!", "%")) and " = !" not in line
        )
        try:
            tree = ast.parse(code, filename=str(path))
        except SyntaxError:
            if "inference" in source:
                _UNPARSEABLE_CELLS_WITH_INFERENCE.append((path, cell_index))
            continue
        pairs |= _collect_from_tree(tree)

    return pairs


def _collect_all_pairs() -> Set[_Pair]:
    pairs: Set[_Pair] = set()
    for path in REPO_ROOT.joinpath("examples").rglob("*.py"):
        pairs |= _collect_from_py(path)
    for path in REPO_ROOT.joinpath("examples").rglob("*.ipynb"):
        pairs |= _collect_from_notebook(path)
    for path in REPO_ROOT.joinpath("docs").rglob("*.ipynb"):
        pairs |= _collect_from_notebook(path)

    return pairs


def _pair_id(pair: _Pair) -> str:
    module, name = pair
    if name is None:
        return module

    pair_id = f"{module}.{name}"

    return pair_id


_COLLECTED_PAIRS = sorted(_collect_all_pairs(), key=_pair_id)


def test_collected_pairs_are_non_empty() -> None:
    assert _COLLECTED_PAIRS


def test_no_unparseable_cells_reference_inference() -> None:
    failures = ", ".join(
        f"{path}:cell[{cell_index}]"
        for path, cell_index in _UNPARSEABLE_CELLS_WITH_INFERENCE
    )

    assert not _UNPARSEABLE_CELLS_WITH_INFERENCE, failures


@pytest.mark.parametrize(
    "module, name", _COLLECTED_PAIRS, ids=[_pair_id(pair) for pair in _COLLECTED_PAIRS]
)
def test_stream_import_resolves(
    module: str, name: Optional[str], stub_ultralytics_if_missing
) -> None:
    module_obj = importlib.import_module(module)
    if name is None:
        return

    try:
        getattr(module_obj, name)
    except AttributeError:
        importlib.import_module(f"{module}.{name}")
