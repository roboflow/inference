"""The copied `merge_window_segments` must BEHAVE identically to the server's.

Copied rather than injected because it is pure logic over
`ActionRecognitionPrediction` (re-exported by execution_engine/entities/base.py)
and `inference_models`' `merge_segment` - an allowed dependency. Source
equality alone never calls either implementation, so both are driven over the
same inputs, including the in-place multi-window case.
"""

import ast
import inspect
import pathlib
from types import SimpleNamespace

import pytest

from inference.core.models.action_recognition import (
    merge_window_segments as server_implementation,
)
from inference.core.workflows.utils.action_recognition import merge_window_segments


def test_signature_matches_the_server_implementation() -> None:
    assert inspect.signature(merge_window_segments) == inspect.signature(
        server_implementation
    )


def _segment(class_name, start, end, confidence=0.9):
    return SimpleNamespace(
        class_name=class_name,
        start_frame_idx=start,
        end_frame_idx=end,
        confidence=confidence,
    )


CASES = [
    ([0, 1, 2, 3], [_segment("walk", 0, 2)], ["walk", "run"], 1.0, None),
    ([0, 1, 2, 3], [_segment("run", 1, 3)], ["walk", "run"], 0.5, None),
    ([0, 1, 2, 3], [_segment("fly", 0, 1)], ["walk", "run"], 1.0, None),
    ([0, 1, 2, 3], [_segment("walk", 0, 2)], ["walk"], 1.0, ["run"]),
    ([0, 1, 2, 3], [_segment("walk", 3, 0)], ["walk"], 1.0, None),
    ([0, 1, 2, 3], [_segment("walk", -5, 99)], ["walk"], 1.0, None),
    ([], [_segment("walk", 0, 1)], ["walk"], 1.0, None),
    ([0, 1], [], ["walk"], 1.0, None),
    ([0, 1, 2], [_segment("walk", 0, 1)], None, 1.0, None),
]


def _dump(timeline):
    return [p.model_dump() for p in timeline]


@pytest.mark.parametrize("frame_numbers,segments,vocabulary,stride,class_filter", CASES)
def test_behaviour_matches_the_server_implementation(
    frame_numbers, segments, vocabulary, stride, class_filter
) -> None:
    ours, theirs = [], []
    for function, timeline in (
        (merge_window_segments, ours),
        (server_implementation, theirs),
    ):
        function(
            timeline=timeline,
            frame_numbers=frame_numbers,
            segments=segments,
            id_vocabulary=vocabulary,
            stride=stride,
            class_filter=class_filter,
        )
    assert _dump(ours) == _dump(theirs)


def test_repeated_calls_union_into_the_same_timeline_identically() -> None:
    # The function mutates `timeline` in place across windows; a copy that
    # merges differently on the second window would pass a single-call test.
    ours, theirs = [], []
    for window in ([_segment("walk", 0, 1)], [_segment("walk", 1, 2)]):
        for function, timeline in (
            (merge_window_segments, ours),
            (server_implementation, theirs),
        ):
            function(
                timeline=timeline,
                frame_numbers=[0, 1, 2],
                segments=window,
                id_vocabulary=["walk"],
                stride=1.0,
            )
    assert _dump(ours) == _dump(theirs)


def test_the_module_imports_only_allowed_packages() -> None:
    path = (
        pathlib.Path(__file__).resolve().parents[4]
        / "inference/core/workflows/utils/action_recognition.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
        elif isinstance(node, ast.Import):
            module = node.names[0].name
        if module and module.startswith("inference."):
            assert module.startswith("inference.core.workflows"), module


def test_syntactic_parser_returns_a_plain_dict() -> None:
    from inference.core.workflows.execution_engine.v1.compiler.syntactic_parser import (
        get_workflow_schema,
    )

    schema = get_workflow_schema()
    assert isinstance(schema, dict)
    assert "$defs" in schema or "definitions" in schema


def test_syntactic_parser_does_not_import_server_entities() -> None:
    path = (
        pathlib.Path(__file__).resolve().parents[4]
        / "inference/core/workflows/execution_engine/v1/compiler/syntactic_parser.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
    }
    assert not any(m.startswith("inference.core.entities") for m in modules)


def test_models_provider_declares_the_action_recognition_loader() -> None:
    from inference.core.workflows.prototypes.models_provider import ModelsProvider

    assert hasattr(ModelsProvider, "load_action_recognition_model")


def test_model_manager_implements_it() -> None:
    from inference.core.managers.base import ModelManager

    assert hasattr(ModelManager, "load_action_recognition_model")


def test_model_manager_decorator_forwards_it() -> None:
    """`ModelManagerDecorator` has no `__getattr__` and overrides `_models`
    to raise, so an inherited method would not reach the inner manager - the
    forwarder must be explicit (Phase 11's D2 makes the same point)."""
    from unittest.mock import MagicMock

    from inference.core.managers.decorators.base import ModelManagerDecorator

    inner = MagicMock()
    decorator = ModelManagerDecorator(model_manager=inner)
    decorator.load_action_recognition_model(model_id="ar/1", api_key="k")
    inner.load_action_recognition_model.assert_called_once_with(
        model_id="ar/1", api_key="k"
    )


def test_the_action_recognition_block_loads_through_the_port() -> None:
    from inference.core.workflows.core_steps.common.entities import StepExecutionMode
    from inference.core.workflows.core_steps.models.roboflow.action_recognition.v1 import (
        ActionRecognitionModelBlockV1,
    )

    class _Manager:
        def __init__(self):
            self.calls = []

        def load_action_recognition_model(self, model_id, api_key=None, **kwargs):
            self.calls.append((model_id, api_key))
            return object()

    manager = _Manager()
    block = ActionRecognitionModelBlockV1(
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
        model_manager=manager,
    )
    assert block._get_model("cosmos-3-edge") is not None
    assert manager.calls == [("cosmos-3-edge", "k")]
