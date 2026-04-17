"""
Compile-time expansion of ``roboflow_core/inner_workflow@v1`` into ordinary steps.

Runs after reference normalization and composition validation on the pre-inline definition.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.execution_engine.profiling.core import WorkflowsProfiler
from inference.core.workflows.execution_engine.v1.compiler.syntactic_parser import (
    parse_workflow_definition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.compiler_bridge import (
    validate_parameter_bindings_against_child,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)

_INPUT_REF_PATTERN = re.compile(r"^\$inputs\.(?P<name>[A-Za-z0-9_\-]+)$")


def _contains_inner_workflow_step(steps: Any) -> bool:
    if not isinstance(steps, list):
        return False
    for step in steps:
        if isinstance(step, dict) and step.get("type") == USE_INNER_WORKFLOW_BLOCK_TYPE:
            return True
    return False


def _collect_step_names_at_level(steps: Any) -> Set[str]:
    names: Set[str] = set()
    if not isinstance(steps, list):
        return names
    for step in steps:
        if isinstance(step, dict):
            n = step.get("name")
            if isinstance(n, str):
                names.add(n)
    return names


def _unique_prefixed_step_name(inner_name: str, child_step_name: str, used: Set[str]) -> str:
    base = f"{inner_name}__{child_step_name}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}__{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _defaults_for_unbound_workflow_parameters(
    child_inputs: Any, bindings: Dict[str, Any]
) -> Dict[str, Any]:
    """Literal defaults for child WorkflowParameter inputs omitted from ``parameter_bindings``."""
    result: Dict[str, Any] = {}
    if not isinstance(child_inputs, list):
        return result
    for spec in child_inputs:
        if not isinstance(spec, dict):
            continue
        if spec.get("type") not in {"WorkflowParameter", "InferenceParameter"}:
            continue
        name = spec.get("name")
        if not isinstance(name, str) or name in bindings:
            continue
        if "default_value" not in spec:
            continue
        dv = spec.get("default_value")
        if dv is None:
            continue
        result[name] = copy.deepcopy(dv)
    return result


def _replace_inputs_in_string(
    s: str,
    *,
    bindings: Dict[str, str],
    input_defaults: Dict[str, Any],
) -> str | Any:
    m = _INPUT_REF_PATTERN.match(s)
    if m:
        name = m.group("name")
        if name in bindings:
            return bindings[name]
        if name in input_defaults:
            return copy.deepcopy(input_defaults[name])
    out = s
    for key in sorted(bindings.keys(), key=len, reverse=True):
        out = out.replace(f"$inputs.{key}", str(bindings[key]))
    return out


def _replace_step_prefixes_in_string(s: str, old_to_new: List[Tuple[str, str]]) -> str:
    out = s
    for old, new in old_to_new:
        out = out.replace(f"$steps.{old}.", f"$steps.{new}.")
    return out


def _replace_bare_child_step_refs_in_string(s: str, step_pairs: List[Tuple[str, str]]) -> str:
    """
    Rewrite ``$steps.child_step`` (no property) used e.g. by ``continue_if.next_steps`` inside the
    child workflow to ``$steps.{inner}__child_step``.
    """
    out = s
    for old, new in sorted(step_pairs, key=lambda p: len(p[0]), reverse=True):
        pattern = rf"\$steps\.{re.escape(old)}(?!\.)(?!__)"
        out = re.sub(pattern, f"$steps.{new}", out)
    return out


def _rewrite_child_scalar(
    value: Any,
    *,
    step_pairs: List[Tuple[str, str]],
    bindings: Dict[str, str],
    input_defaults: Dict[str, Any],
) -> Any:
    if not isinstance(value, str):
        return value
    after_inputs = _replace_inputs_in_string(value, bindings=bindings, input_defaults=input_defaults)
    if not isinstance(after_inputs, str):
        return after_inputs
    dotted = _replace_step_prefixes_in_string(after_inputs, step_pairs)
    return _replace_bare_child_step_refs_in_string(dotted, step_pairs)


def _deep_map_leaves(obj: Any, fn) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_map_leaves(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_map_leaves(v, fn) for v in obj]
    return fn(obj)


def _replace_inner_step_control_and_output_refs_in_object(
    obj: Any,
    *,
    inner_step_name: str,
    output_name_to_selector: Dict[str, str],
    first_inlined_step_name: str,
) -> Any:
    if isinstance(obj, dict):
        return {
            k: _replace_inner_step_control_and_output_refs_in_object(
                v,
                inner_step_name=inner_step_name,
                output_name_to_selector=output_name_to_selector,
                first_inlined_step_name=first_inlined_step_name,
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [
            _replace_inner_step_control_and_output_refs_in_object(
                v,
                inner_step_name=inner_step_name,
                output_name_to_selector=output_name_to_selector,
                first_inlined_step_name=first_inlined_step_name,
            )
            for v in obj
        ]
    if not isinstance(obj, str):
        return obj
    s = obj
    for out_name in sorted(output_name_to_selector.keys(), key=len, reverse=True):
        token = f"$steps.{inner_step_name}.{out_name}"
        s = s.replace(token, output_name_to_selector[out_name])
    # Control flow may reference the inner step as a whole, e.g. ``next_steps: ["$steps.inner"]``.
    # Inlined steps are named ``{inner}__{child_step}``; route bare references to the first.
    bare = re.compile(
        rf"\$steps\.{re.escape(inner_step_name)}(?!\.)"
        # Do not match the prefix of ``$steps.inner__child`` (inlined step name).
        rf"(?!__)"
    )
    s = bare.sub(f"$steps.{first_inlined_step_name}", s)
    return s


def _expand_leaf_inner_at_index(
    workflow: Dict[str, Any],
    step_index: int,
    *,
    available_blocks: Any,
    profiler: Optional[WorkflowsProfiler],
) -> None:
    steps = workflow.get("steps")
    if not isinstance(steps, list) or step_index >= len(steps):
        return
    inner_step = steps[step_index]
    if not isinstance(inner_step, dict) or inner_step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
        raise AssertionError("expected inner_workflow step")
    inner_name = inner_step.get("name")
    if not isinstance(inner_name, str) or not inner_name:
        raise WorkflowDefinitionError(
            public_message="inner_workflow step requires a non-empty string `name`.",
            context="workflow_compilation | inner_workflow_inlining",
        )
    child = inner_step.get("workflow_definition")
    if not isinstance(child, dict):
        raise WorkflowDefinitionError(
            public_message=f"inner_workflow step `{inner_name}` requires `workflow_definition` object.",
            context="workflow_compilation | inner_workflow_inlining",
        )
    bindings_raw = inner_step.get("parameter_bindings") or {}
    if not isinstance(bindings_raw, dict):
        raise WorkflowDefinitionError(
            public_message=f"inner_workflow step `{inner_name}` requires `parameter_bindings` object.",
            context="workflow_compilation | inner_workflow_inlining",
        )
    bindings = {str(k): str(v) for k, v in bindings_raw.items()}

    child_parsed = parse_workflow_definition(
        raw_workflow_definition=child,
        available_blocks=available_blocks,
        profiler=profiler,
    )
    validate_parameter_bindings_against_child(
        bindings=bindings,
        child_parsed=child_parsed,
        step_name=inner_name,
    )

    input_defaults = _defaults_for_unbound_workflow_parameters(
        child.get("inputs"),
        bindings,
    )

    used_names = _collect_step_names_at_level(steps)
    used_names.discard(inner_name)

    child_steps = child.get("steps") or []
    if not isinstance(child_steps, list):
        raise WorkflowDefinitionError(
            public_message=f"inner_workflow `{inner_name}` child workflow_definition.steps must be a list.",
            context="workflow_compilation | inner_workflow_inlining",
        )

    old_to_new: List[Tuple[str, str]] = []
    for cs in child_steps:
        if not isinstance(cs, dict):
            continue
        old = cs.get("name")
        if not isinstance(old, str) or not old:
            raise WorkflowDefinitionError(
                public_message="Each step in an inner workflow must have a non-empty string `name`.",
                context="workflow_compilation | inner_workflow_inlining",
            )
        new_name = _unique_prefixed_step_name(inner_name, old, used_names)
        old_to_new.append((old, new_name))

    step_pairs = sorted(old_to_new, key=lambda p: len(p[0]), reverse=True)

    def rewrite_leaf(x: Any) -> Any:
        return _rewrite_child_scalar(
            x,
            step_pairs=step_pairs,
            bindings=bindings,
            input_defaults=input_defaults,
        )

    inlined_steps: List[Dict[str, Any]] = []
    for cs in child_steps:
        if not isinstance(cs, dict):
            continue
        old = cs.get("name")
        if not isinstance(old, str):
            continue
        new_name = next(nn for oo, nn in old_to_new if oo == old)
        cloned = copy.deepcopy(cs)
        cloned["name"] = new_name
        inlined_steps.append(_deep_map_leaves(cloned, rewrite_leaf))

    if not inlined_steps:
        raise WorkflowDefinitionError(
            public_message=f"inner_workflow `{inner_name}` child has no steps to inline.",
            context="workflow_compilation | inner_workflow_inlining",
        )
    first_inlined_step_name = str(inlined_steps[0]["name"])

    output_name_to_selector: Dict[str, str] = {}
    for out in child.get("outputs") or []:
        if not isinstance(out, dict) or out.get("type") != "JsonField":
            continue
        oname = out.get("name")
        sel = out.get("selector")
        if not isinstance(oname, str) or not isinstance(sel, str):
            continue
        rewritten = _rewrite_child_scalar(
            sel,
            step_pairs=step_pairs,
            bindings=bindings,
            input_defaults=input_defaults,
        )
        if not isinstance(rewritten, str):
            raise WorkflowDefinitionError(
                public_message=(
                    f"inner_workflow `{inner_name}` child output `{oname}` selector must rewrite "
                    f"to a string selector after inlining."
                ),
                context="workflow_compilation | inner_workflow_inlining",
            )
        output_name_to_selector[oname] = rewritten

    patched_workflow = _replace_inner_step_control_and_output_refs_in_object(
        workflow,
        inner_step_name=inner_name,
        output_name_to_selector=output_name_to_selector,
        first_inlined_step_name=first_inlined_step_name,
    )
    workflow.clear()
    workflow.update(patched_workflow)

    steps = workflow.get("steps")
    assert isinstance(steps, list)
    new_steps = steps[:step_index] + inlined_steps + steps[step_index + 1 :]
    workflow["steps"] = new_steps


def _inline_one_inner_workflow_leaf(workflow: Dict[str, Any], *, available_blocks, profiler) -> bool:
    steps = workflow.get("steps")
    if not isinstance(steps, list):
        return False
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
            continue
        child = step.get("workflow_definition")
        if not isinstance(child, dict):
            continue
        child_steps = child.get("steps") or []
        if _contains_inner_workflow_step(child_steps):
            if _inline_one_inner_workflow_leaf(child, available_blocks=available_blocks, profiler=profiler):
                return True
            continue
        _expand_leaf_inner_at_index(
            workflow,
            i,
            available_blocks=available_blocks,
            profiler=profiler,
        )
        return True
    return False


def fully_inline_inner_workflow_steps(
    workflow_definition: Dict[str, Any],
    *,
    available_blocks: Any,
    profiler: Optional[WorkflowsProfiler] = None,
) -> Dict[str, Any]:
    """
    Return a copy of ``workflow_definition`` with all ``inner_workflow`` steps expanded.

    Composition and reference normalization must already be applied to the input.
    """
    root = copy.deepcopy(workflow_definition)
    while _contains_inner_workflow_step(root.get("steps")):
        if not _inline_one_inner_workflow_leaf(root, available_blocks=available_blocks, profiler=profiler):
            raise WorkflowDefinitionError(
                public_message=(
                    "Could not inline inner_workflow steps (unexpected nested structure). "
                    "Ensure inner workflow composition is valid."
                ),
                context="workflow_compilation | inner_workflow_inlining",
            )
    return root
