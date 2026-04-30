"""
Unit tests for ``inference.core.workflows.execution_engine.v1.inner_workflow.inline``.

These focus on small, composable helpers so behavior stays obvious when reading or changing
``inline.py``; one test exercises ``inline_inner_workflow_steps`` on a minimal graph.
"""

from __future__ import annotations

import copy
from unittest import mock

import pytest

from inference.core.workflows.execution_engine.introspection import blocks_loader
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.errors import (
    InnerWorkflowInliningStructureError,
    InnerWorkflowInvalidStepEntryError,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.inline import (
    _collect_step_names_at_level,
    _contains_inner_workflow_step,
    _deep_map_leaves,
    _defaults_for_unbound_workflow_parameters,
    _expand_leaf_inner_at_index,
    _inline_one_inner_workflow_leaf,
    _replace_bare_child_step_refs_in_string,
    _replace_inner_step_control_and_output_refs_in_object,
    _replace_inputs_in_string,
    _replace_step_prefixes_in_string,
    _rewrite_inner_scalar,
    _unique_prefixed_step_name,
    inline_inner_workflow_steps,
)

_SCALAR_ONLY_ECHO_PLUGIN = (
    "tests.workflows.integration_tests.execution.stub_plugins.scalar_only_block_plugin"
)


# --- _contains_inner_workflow_step ---


def test_contains_inner_workflow_step_true_when_present() -> None:
    steps = [
        {"name": "a", "type": "some/block@v1"},
        {"name": "b", "type": USE_INNER_WORKFLOW_BLOCK_TYPE},
    ]
    assert _contains_inner_workflow_step(steps) is True


def test_contains_inner_workflow_step_false_when_absent() -> None:
    steps = [{"name": "a", "type": "some/block@v1"}]
    assert _contains_inner_workflow_step(steps) is False


def test_contains_inner_workflow_step_false_for_non_list() -> None:
    assert _contains_inner_workflow_step("not-a-list") is False  # type: ignore[arg-type]


def test_contains_inner_workflow_step_false_when_any_step_not_dict() -> None:
    assert (
        _contains_inner_workflow_step([{"name": "ok", "type": "x"}, None])  # type: ignore[list-item]
        is False
    )


# --- _collect_step_names_at_level ---


def test_collect_step_names_empty_for_non_list() -> None:
    assert _collect_step_names_at_level(None) == set()
    assert _collect_step_names_at_level("x") == set()


def test_collect_step_names_collects_string_names_only() -> None:
    steps = [
        {"name": "a", "type": "t"},
        {"name": 123},
        "skip",
        {"type": "t"},
    ]
    assert _collect_step_names_at_level(steps) == {"a"}


# --- _unique_prefixed_step_name ---


def test_unique_prefixed_step_name_base_form_and_collision() -> None:
    used: set[str] = {"outer__pick", "other"}
    assert _unique_prefixed_step_name("outer", "echo", used) == "outer__echo"
    assert "outer__echo" in used
    assert _unique_prefixed_step_name("outer", "pick", used) == "outer__pick__2"
    assert _unique_prefixed_step_name("outer", "pick", used) == "outer__pick__3"


# --- _defaults_for_unbound_workflow_parameters ---


def test_defaults_for_unbound_skips_bound_and_non_parameter_inputs() -> None:
    inner_inputs = [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "alpha",
            "default_value": {"k": 1},
        },
        {
            "type": "WorkflowParameter",
            "name": "beta",
            "default_value": ["beta-default"],
        },
    ]
    bindings = {"alpha": "$inputs.x"}
    out = _defaults_for_unbound_workflow_parameters(inner_inputs, bindings)
    assert set(out) == {"beta"}
    assert out["beta"] == ["beta-default"]
    assert out["beta"] is not inner_inputs[2]["default_value"]


def test_defaults_for_unbound_skips_none_default_value() -> None:
    inner_inputs = [
        {"type": "WorkflowParameter", "name": "z", "default_value": None},
    ]
    assert _defaults_for_unbound_workflow_parameters(inner_inputs, {}) == {}


def test_defaults_for_unbound_rejects_non_list_inputs() -> None:
    with pytest.raises(InnerWorkflowInvalidStepEntryError, match="`inputs` list"):
        _defaults_for_unbound_workflow_parameters({}, {})  # type: ignore[arg-type]


def test_defaults_for_unbound_rejects_non_dict_spec() -> None:
    with pytest.raises(InnerWorkflowInvalidStepEntryError, match="inputs` list entry"):
        _defaults_for_unbound_workflow_parameters([None], {})  # type: ignore[list-item]


# --- _replace_inputs_in_string ---


def test_replace_inputs_whole_string_returns_binding_value_preserve_type() -> None:
    assert _replace_inputs_in_string(
        "$inputs.flag",
        bindings={},
        input_defaults={"flag": [1, 2]},
    ) == [1, 2]
    assert (
        _replace_inputs_in_string(
            "$inputs.x",
            bindings={"x": "$steps.a.out"},
            input_defaults={},
        )
        == "$steps.a.out"
    )


def test_replace_inputs_embedded_longest_key_first() -> None:
    s = _replace_inputs_in_string(
        "a$inputs.image_sizeb$inputs.imagec",
        bindings={"image": "X", "image_size": "Y"},
        input_defaults={},
    )
    assert s == "aYbXc"


# --- _replace_step_prefixes_in_string ---


def test_replace_step_prefixes_applies_pairs_in_order() -> None:
    s = _replace_step_prefixes_in_string(
        "$steps.a.x $steps.b.y",
        old_to_new=[("a", "A"), ("b", "B")],
    )
    assert s == "$steps.A.x $steps.B.y"


# --- _replace_bare_child_step_refs_in_string ---


def test_replace_bare_child_step_refs_does_not_touch_dotted_refs() -> None:
    s = _replace_bare_child_step_refs_in_string(
        "$steps.detect.predictions",
        step_pairs=[("detect", "inner__detect")],
    )
    assert s == "$steps.detect.predictions"


def test_replace_bare_child_step_refs_rewrites_json_list_of_step_tokens() -> None:
    s = _replace_bare_child_step_refs_in_string(
        '["$steps.detect", "$steps.other"]',
        step_pairs=[("detect", "inner__detect"), ("other", "inner__other")],
    )
    assert "$steps.inner__detect" in s and "$steps.inner__other" in s


def test_replace_bare_child_step_refs_longer_old_name_first_avoids_partial_match() -> (
    None
):
    s = _replace_bare_child_step_refs_in_string(
        "$steps.det",
        step_pairs=[("det", "p__det"), ("d", "p__d")],
    )
    assert s == "$steps.p__det"


# --- _rewrite_inner_scalar ---


def test_rewrite_inner_scalar_non_string_unchanged() -> None:
    assert _rewrite_inner_scalar(
        {"nested": True},
        step_pairs=[("a", "p__a")],
        bindings={},
        input_defaults={},
    ) == {"nested": True}


def test_rewrite_inner_scalar_pipeline_inputs_then_dotted_then_bare() -> None:
    out = _rewrite_inner_scalar(
        "$inputs.msg and $steps.echo.out and $steps.echo",
        step_pairs=[("echo", "inner__echo")],
        bindings={"msg": "hello"},
        input_defaults={},
    )
    assert out == "hello and $steps.inner__echo.out and $steps.inner__echo"


def test_rewrite_inner_scalar_whole_input_non_string_skips_step_rewrite() -> None:
    sentinel = {"selector": "$steps.echo.x"}
    out = _rewrite_inner_scalar(
        "$inputs.obj",
        step_pairs=[("echo", "inner__echo")],
        bindings={"obj": sentinel},
        input_defaults={},
    )
    assert out is sentinel


# --- _deep_map_leaves ---


def test_deep_map_leaves_applies_fn_to_nested_scalars_only() -> None:
    obj = {"a": [1, {"b": "x"}], "c": 3}

    def double_ints(v):
        if isinstance(v, int):
            return v * 2
        return v

    got = _deep_map_leaves(obj, double_ints)
    assert got == {"a": [2, {"b": "x"}], "c": 6}


# --- _replace_inner_step_control_and_output_refs_in_object ---


def test_replace_inner_step_control_rewrites_output_tokens_longest_out_name_first() -> (
    None
):
    obj = {
        "a": "$steps.i.echo_extra",
        "b": "$steps.i.echo",
    }
    out = _replace_inner_step_control_and_output_refs_in_object(
        obj,
        inner_step_name="i",
        output_name_to_selector={
            "echo": "$steps.i__pick.out",
            "echo_extra": "$steps.i__pick.extra",
        },
        first_inlined_step_name="i__pick",
    )
    assert out["b"] == "$steps.i__pick.out"
    assert out["a"] == "$steps.i__pick.extra"


def test_replace_inner_step_control_bare_inner_step_routes_to_first_inlined() -> None:
    obj = {"next": ["$steps.wrapper", "$steps.wrapper__already"]}
    out = _replace_inner_step_control_and_output_refs_in_object(
        obj,
        inner_step_name="wrapper",
        output_name_to_selector={},
        first_inlined_step_name="wrapper__first",
    )
    assert out["next"][0] == "$steps.wrapper__first"
    assert "$steps.wrapper__already" in out["next"][1]


def test_replace_inner_step_control_non_string_leaves_unchanged() -> None:
    assert _replace_inner_step_control_and_output_refs_in_object(
        {"n": 42, "s": "$steps.i.x"},
        inner_step_name="i",
        output_name_to_selector={"x": "y"},
        first_inlined_step_name="f",
    ) == {"n": 42, "s": "y"}


# --- inline_inner_workflow_steps (minimal) ---


def _minimal_echo_inner_workflow_definition() -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "child_msg",
                "default_value": "default-child",
            },
        ],
        "steps": [
            {
                "type": "scalar_only_echo",
                "name": "pick",
                "value": "$inputs.child_msg",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "echo",
                "selector": "$steps.pick.output",
            },
        ],
    }


def test_inline_inner_workflow_steps_expands_leaf_and_preserves_parent_inputs() -> None:
    raw = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "parent_msg", "default_value": "p"},
        ],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "child",
                "workflow_definition": _minimal_echo_inner_workflow_definition(),
                "parameter_bindings": {"child_msg": "$inputs.parent_msg"},
            },
        ],
        "outputs": [],
    }
    original_pick = raw["steps"][0]["workflow_definition"]["steps"][0]
    init_parameters = {"workflows_core.api_key": None}

    with mock.patch.object(
        blocks_loader,
        "get_plugin_modules",
        return_value=[_SCALAR_ONLY_ECHO_PLUGIN],
    ):
        blocks_loader.clear_caches()
        try:
            from inference.core.workflows.execution_engine.v1.inner_workflow.compiler_bridge import (
                validate_inner_workflow_composition_from_raw_workflow_definition,
            )
            from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
                normalize_inner_workflow_references_in_definition,
            )

            normalized = normalize_inner_workflow_references_in_definition(
                workflow_definition=copy.deepcopy(raw),
                init_parameters=init_parameters,
            )
            validate_inner_workflow_composition_from_raw_workflow_definition(normalized)
            available_blocks = load_workflow_blocks(
                execution_engine_version=None,
                profiler=None,
            )
            inlined = inline_inner_workflow_steps(
                copy.deepcopy(normalized),
                available_blocks=available_blocks,
                profiler=None,
            )
        finally:
            blocks_loader.clear_caches()

    assert raw["steps"][0]["workflow_definition"]["steps"][0] == original_pick
    assert len(inlined["steps"]) == 1
    step = inlined["steps"][0]
    assert step["name"] == "child__pick"
    assert step["type"] == "scalar_only_echo"
    assert step["value"] == "$inputs.parent_msg"


def test_inline_inner_workflow_steps_raises_when_inlining_makes_no_progress() -> None:
    """``InnerWorkflowInliningStructureError`` when the outer graph still has inner steps but no leaf expands."""
    raw = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "parent_msg", "default_value": "p"},
        ],
        "steps": [
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "child",
                "workflow_definition": _minimal_echo_inner_workflow_definition(),
                "parameter_bindings": {"child_msg": "$inputs.parent_msg"},
            },
        ],
        "outputs": [],
    }
    init_parameters = {"workflows_core.api_key": None}

    with mock.patch.object(
        blocks_loader,
        "get_plugin_modules",
        return_value=[_SCALAR_ONLY_ECHO_PLUGIN],
    ):
        blocks_loader.clear_caches()
        try:
            from inference.core.workflows.execution_engine.v1.inner_workflow.compiler_bridge import (
                validate_inner_workflow_composition_from_raw_workflow_definition,
            )
            from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
                normalize_inner_workflow_references_in_definition,
            )

            normalized = normalize_inner_workflow_references_in_definition(
                workflow_definition=copy.deepcopy(raw),
                init_parameters=init_parameters,
            )
            validate_inner_workflow_composition_from_raw_workflow_definition(normalized)
            available_blocks = load_workflow_blocks(
                execution_engine_version=None,
                profiler=None,
            )
            with mock.patch(
                "inference.core.workflows.execution_engine.v1.inner_workflow.inline._inline_one_inner_workflow_leaf",
                return_value=False,
            ):
                with pytest.raises(
                    InnerWorkflowInliningStructureError, match="Could not inline"
                ):
                    inline_inner_workflow_steps(
                        copy.deepcopy(normalized),
                        available_blocks=available_blocks,
                        profiler=None,
                    )
        finally:
            blocks_loader.clear_caches()


# --- _inline_one_inner_workflow_leaf ---


def test_inline_one_inner_workflow_leaf_false_when_no_inner_step() -> None:
    wf = {
        "version": "1.0",
        "inputs": [],
        "steps": [{"name": "a", "type": "scalar_only_echo"}],
        "outputs": [],
    }
    with mock.patch.object(
        blocks_loader,
        "get_plugin_modules",
        return_value=[_SCALAR_ONLY_ECHO_PLUGIN],
    ):
        blocks_loader.clear_caches()
        try:
            available_blocks = load_workflow_blocks(None, None)
            assert (
                _inline_one_inner_workflow_leaf(
                    wf,
                    available_blocks=available_blocks,
                    profiler=None,
                )
                is False
            )
        finally:
            blocks_loader.clear_caches()


# --- _expand_leaf_inner_at_index ---


def test_expand_leaf_inner_at_index_mutates_workflow_in_place() -> None:
    inner = _minimal_echo_inner_workflow_definition()
    wf = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "parent_msg", "default_value": "p"},
        ],
        "steps": [
            {"name": "before", "type": "scalar_only_echo", "value": "x"},
            {
                "type": USE_INNER_WORKFLOW_BLOCK_TYPE,
                "name": "child",
                "workflow_definition": inner,
                "parameter_bindings": {"child_msg": "$inputs.parent_msg"},
            },
            {"name": "after", "type": "scalar_only_echo", "value": "y"},
        ],
        "outputs": [],
    }
    init_parameters = {"workflows_core.api_key": None}

    with mock.patch.object(
        blocks_loader,
        "get_plugin_modules",
        return_value=[_SCALAR_ONLY_ECHO_PLUGIN],
    ):
        blocks_loader.clear_caches()
        try:
            from inference.core.workflows.execution_engine.v1.inner_workflow.compiler_bridge import (
                validate_inner_workflow_composition_from_raw_workflow_definition,
            )
            from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
                normalize_inner_workflow_references_in_definition,
            )

            normalized = normalize_inner_workflow_references_in_definition(
                copy.deepcopy(wf),
                init_parameters=init_parameters,
            )
            validate_inner_workflow_composition_from_raw_workflow_definition(normalized)
            available_blocks = load_workflow_blocks(None, None)
            _expand_leaf_inner_at_index(
                normalized,
                1,
                available_blocks=available_blocks,
                profiler=None,
            )
        finally:
            blocks_loader.clear_caches()

    names = [s["name"] for s in normalized["steps"]]
    assert names == ["before", "child__pick", "after"]
