"""Additive edit of `inference/core/workflows/prototypes/models_provider.py` (Task 11.1).

Anchor-based, never a whole-file replacement, so whatever another phase has
already appended to the module survives - Phase 9's `CORE_MODEL_ENDPOINT_TYPE`
constant (Task 9.2) and its `load_action_recognition_model` declaration
(Task 9.9) in particular. It:

  1. rewrites the `from typing import ...` line to carry Dict and Union as well
     (Tasks 11.8/11.9 add annotations that need them);
  2. deletes the `__getitem__` declaration;
  3. inserts the five first-class model-access declarations before
     `def __contains__`;
  4. replaces the "PROVISIONAL MEMBERS." docstring paragraph with the
     single-member form Task 11.15 later deletes.

Re-parses before writing and verifies the post-state. Idempotent: a second run
reports SKIP and changes nothing.

Run:  python scripts/phase11_port_edit.py inference/core/workflows/prototypes/models_provider.py
"""

import ast
import re
import sys
from pathlib import Path

TYPING_NAMES = ("Any", "Dict", "List", "Optional", "Protocol", "Union")

GETITEM_DECL = "    def __getitem__(self, key: str) -> Any: ...\n"

NEW_MEMBERS = """    def get_keypoints_classes(self, model_id: str) -> List[List[str]]: ...

    def model_supports_stream_pipeline(self, model_id: str) -> bool: ...

    def get_model_pipeline_depth(self, model_id: str) -> int: ...

    def flush_model_stream_pipeline(self, model_id: str) -> Optional[List[Any]]: ...

    def shutdown_model_stream_pipeline(self, model_id: str) -> None: ...

"""

CONTAINS_ANCHOR = "    def __contains__(self, model_id: str) -> bool: ...\n"

NEW_PARAGRAPH = """    The stream-pipeline members are prefixed because ``flush_stream_pipeline``,
    ``stream_pipeline_depth``, ``close_stream_pipeline`` and
    ``is_stream_pipelined`` are already a *block*-level duck-typed protocol that
    the executor and the server's stream handler call on step instances.

    PROVISIONAL MEMBER. ``infer_from_request_sync`` takes a pydantic request
    object built by the caller from ``inference.core.entities`` - it is the
    method Phase 11 removes entirely. Do not build new code against it.
"""


def edit(source: str) -> str:
    if "def get_keypoints_classes" in source and "__getitem__" not in source:
        return source  # already applied
    # 1. typing import: union of what is there and what Tasks 11.8/11.9 need.
    match = re.search(r"^from typing import ([^\n]+)\n", source, re.MULTILINE)
    if match is None:
        raise SystemExit("no `from typing import` line found")
    present = {name.strip() for name in match.group(1).split(",")}
    names = sorted(present | set(TYPING_NAMES))
    source = source.replace(
        match.group(0), f"from typing import {', '.join(names)}\n", 1
    )
    # 2. drop __getitem__.
    if source.count(GETITEM_DECL) != 1:
        raise SystemExit("expected exactly one __getitem__ declaration")
    source = source.replace(GETITEM_DECL, "", 1)
    # 3. insert the five members before __contains__.
    if source.count(CONTAINS_ANCHOR) != 1:
        raise SystemExit("expected exactly one __contains__ declaration")
    source = source.replace(CONTAINS_ANCHOR, NEW_MEMBERS + CONTAINS_ANCHOR, 1)
    # 4. the docstring paragraph: from "    PROVISIONAL MEMBERS." to the closing quotes.
    paragraph = re.search(
        r"    PROVISIONAL MEMBERS\..*?(?=\n    \"\"\"\n)", source, re.DOTALL
    )
    if paragraph is None:
        raise SystemExit("PROVISIONAL MEMBERS paragraph not found")
    source = (
        source[: paragraph.start()]
        + NEW_PARAGRAPH.rstrip("\n")
        + source[paragraph.end() :]
    )
    source = re.sub(r"\n{3,}(    def __contains__)", r"\n\n\1", source)
    return source


def verify(source: str, original: str) -> None:
    tree = ast.parse(source)
    declared = {
        node.name
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "ModelsProvider"
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in (
        "get_keypoints_classes",
        "model_supports_stream_pipeline",
        "get_model_pipeline_depth",
        "flush_model_stream_pipeline",
        "shutdown_model_stream_pipeline",
        "__contains__",
        "add_model",
        "run_tensor_native_inference",
        "get_class_names",
        "infer_from_request_sync",
    ):
        assert name in declared, f"missing {name}"
    assert "__getitem__" not in declared
    # Whatever other phases put here must survive untouched.
    for preserved in ("CORE_MODEL_ENDPOINT_TYPE", "load_action_recognition_model"):
        assert (preserved in source) == (preserved in original), preserved
    assert "PROVISIONAL MEMBERS" not in source and "PROVISIONAL MEMBER." in source


def main(paths):
    for raw in paths:
        path = Path(raw)
        original = path.read_text(encoding="utf-8")
        updated = edit(original)
        if updated == original:
            print(f"SKIP (already applied): {path}")
            continue
        verify(updated, original)
        path.write_text(updated, encoding="utf-8")
        print(f"edited: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
