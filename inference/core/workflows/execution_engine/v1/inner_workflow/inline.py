"""
Compile-time expansion of ``roboflow_core/inner_workflow@v1`` into ordinary steps.

Runs after reference normalization and composition validation on the pre-inline definition.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
from inference.core.workflows.execution_engine.v1.inner_workflow.errors import (
    InnerWorkflowInliningStructureError,
    InnerWorkflowInvalidStepEntryError,
)


# Requires format: $inputs.<name>
#
# Examples:
# $inputs.username
# $inputs.file_1
# $inputs.var-name
# $inputs.ABC123
_INPUT_REF_PATTERN = re.compile(r"^\$inputs\.(?P<name>[A-Za-z0-9_\-]+)$")


def _contains_inner_workflow_step(steps: List[Dict[str, Any]]) -> bool:
    """Return True if any step in ``steps`` is an ``inner_workflow`` block.

    Returns False if ``steps`` is not a list or any entry is not a dict.
    """
    if not isinstance(steps, list):
        return False

    for step in steps:
        if not isinstance(step, dict):
            return False

        if step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
            continue

        return True
    return False


def _collect_step_names_at_level(steps: Any) -> Set[str]:
    """Collect string ``name`` fields from each dict step at one workflow level.

    Non-lists or non-dict steps are skipped without error so callers can probe loosely-typed
    structures; use ``_contains_inner_workflow_step`` when strict validation is required.
    """
    names: Set[str] = set()
    if not isinstance(steps, list):
        return names
    for step in steps:
        if isinstance(step, dict):
            n = step.get("name")
            if isinstance(n, str):
                names.add(n)
    return names


def _unique_prefixed_step_name(
    inner_name: str,
    inner_step_name: str,
    used: Set[str],
) -> str:
    """Pick a globally unique step name for an inlined child: ``{inner}__{child}`` with optional suffix.

    Reserves the chosen name in ``used`` (mutates the set) so later children do not collide
    with siblings or existing parent steps.
    """
    base = f"{inner_name}__{inner_step_name}"
    candidate = base
    suffix = 2

    while candidate in used:
        candidate = f"{base}__{suffix}"
        suffix += 1

    used.add(candidate)
    return candidate


def _defaults_for_unbound_workflow_parameters(
    inner_inputs: List[Dict[str, Any]],
    bindings: Dict[str, Any],
) -> Dict[str, Any]:
    """Build default values for child parameters that have no ``parameter_bindings`` entry.

    Only considers ``WorkflowParameter`` and ``InferenceParameter`` entries in the child
    ``inputs`` list that define a non-``None`` ``default_value``. Each value is deep-copied.

    Raises:
        InnerWorkflowInvalidStepEntryError: If ``inner_inputs`` is not a list or an entry is invalid.
    """
    result: Dict[str, Any] = {}

    if not isinstance(inner_inputs, list):
        raise InnerWorkflowInvalidStepEntryError(
            "inner_workflow step `{inner_name}` requires `inputs` list, got "
            f"{type(inner_inputs).__name__}."
        )

    for spec in inner_inputs:
        if not isinstance(spec, dict):
            raise InnerWorkflowInvalidStepEntryError(
                "inner_workflow step `{inner_name}` `inputs` list entry must be a JSON object (dict), got "
                f"{type(spec).__name__}."
            )

        if spec.get("type") not in {"WorkflowParameter", "InferenceParameter"}:
            continue

        name = spec.get("name")
        if name in bindings:
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
    """Resolve ``$inputs.<name>`` placeholders inside a single string leaf.

    If ``s`` is *exactly* ``$inputs.<name>`` (see ``_INPUT_REF_PATTERN``), the result is the
    bound value from ``bindings`` or a deep copy of ``input_defaults[name]``. That preserves
    non-string values (selectors, numbers, etc.) for fields that store a whole reference.

    Otherwise, every occurrence of each literal substring ``$inputs.<key>`` is replaced with
    ``str(bindings[key])``. Keys are applied longest-first so a name like ``image`` does not
    corrupt a longer token such as ``$inputs.image_size``.

    Examples:
        >>> _replace_inputs_in_string(
        ...     "$inputs.threshold",
        ...     bindings={"threshold": "0.5"},
        ...     input_defaults={},
        ... )
        '0.5'

        >>> _replace_inputs_in_string(
        ...     "prefix-$inputs.image_size-suffix",
        ...     bindings={"image_size": "42", "image": "x"},
        ...     input_defaults={},
        ... )
        'prefix-42-suffix'

        >>> _replace_inputs_in_string(
        ...     "$inputs.flag",
        ...     bindings={},
        ...     input_defaults={"flag": {"nested": True}},
        ... )
        {'nested': True}
    """
    matched = _INPUT_REF_PATTERN.match(s)

    if matched:
        name = matched.group("name")
        if name in bindings:
            return bindings[name]
        if name in input_defaults:
            return copy.deepcopy(input_defaults[name])

    out = s
    for key in sorted(bindings.keys(), key=len, reverse=True):
        out = out.replace(f"$inputs.{key}", str(bindings[key]))

    return out


def _replace_step_prefixes_in_string(
    s: str,
    *,
    old_to_new: List[Tuple[str, str]],
) -> str:
    """Rewrite dotted step references: ``$steps.<old>.`` → ``$steps.<new>.`` for each mapping pair.

    Pairs are applied in list order; callers typically pass ``old_to_new`` sorted by descending
    old-name length so longer step ids are substituted before shorter prefixes overlap.
    """
    out = s
    for old, new in old_to_new:
        out = out.replace(f"$steps.{old}.", f"$steps.{new}.")
    return out


def _replace_bare_child_step_refs_in_string(
    s: str,
    *,
    step_pairs: List[Tuple[str, str]],
) -> str:
    """Rewrite bare child step tokens ``$steps.<child>`` to inlined names ``$steps.<new>``.

    Some fields (for example ``continue_if.next_steps``) reference a step by id only, without a
    property segment after the step name. Those strings must be rewritten when the child graph
    is inlined so they point at ``$steps.{inner_step}__{child_step}``.

    Tokens of the form ``$steps.<child>.<output>`` are left to ``_replace_step_prefixes_in_string``
    and are not matched here: the pattern requires no ``.`` immediately after the step name and
    skips names already followed by ``__`` (already prefixed).

    Examples:
        >>> _replace_bare_child_step_refs_in_string(
        ...     '["$steps.detect"]',
        ...     step_pairs=[("detect", "inner__detect")],
        ... )
        '["$steps.inner__detect"]'

        >>> _replace_bare_child_step_refs_in_string(
        ...     "$steps.detect.predictions",
        ...     step_pairs=[("detect", "inner__detect")],
        ... )
        '$steps.detect.predictions'
    """
    out = s
    for old, new in sorted(step_pairs, key=lambda p: len(p[0]), reverse=True):
        pattern = rf"\$steps\.{re.escape(old)}(?!\.)(?!__)"
        out = re.sub(pattern, f"$steps.{new}", out)
    return out


def _rewrite_inner_scalar(
    value: Any,
    *,
    step_pairs: List[Tuple[str, str]],
    bindings: Dict[str, str],
    input_defaults: Dict[str, Any],
) -> Any:
    """Rewrite one JSON scalar from the child workflow for inlining under ``inner_step``.

    Non-strings are returned unchanged. For strings, replacements run in order:

    1. ``_replace_inputs_in_string`` — workflow inputs and defaults.
    2. If that step produced a non-string (whole ``$inputs.*`` reference), it is returned as-is;
       dotted step references are not applied to non-strings.
    3. ``_replace_step_prefixes_in_string`` — ``$steps.<child>.`` → ``$steps.<new>.``.
    4. ``_replace_bare_child_step_refs_in_string`` — bare ``$steps.<child>`` tokens.

    ``step_pairs`` maps each original child step name to its unique name after inlining
    (typically ``f"{inner_name}__{child_step}"``).

    Examples:
        >>> _rewrite_inner_scalar(
        ...     123,
        ...     step_pairs=[("a", "inner__a")],
        ...     bindings={},
        ...     input_defaults={},
        ... )
        123

        >>> _rewrite_inner_scalar(
        ...     "$steps.detect.predictions",
        ...     step_pairs=[("detect", "inner__detect")],
        ...     bindings={},
        ...     input_defaults={},
        ... )
        '$steps.inner__detect.predictions'

        >>> _rewrite_inner_scalar(
        ...     "$inputs.model_id",
        ...     step_pairs=[],
        ...     bindings={"model_id": "my-model"},
        ...     input_defaults={},
        ... )
        'my-model'
    """
    if not isinstance(value, str):
        return value

    after_inputs = _replace_inputs_in_string(
        value,
        bindings=bindings,
        input_defaults=input_defaults,
    )

    if not isinstance(after_inputs, str):
        return after_inputs

    dotted = _replace_step_prefixes_in_string(
        after_inputs,
        old_to_new=step_pairs,
    )

    out = _replace_bare_child_step_refs_in_string(
        dotted,
        step_pairs=step_pairs,
    )

    return out


def _deep_map_leaves(obj: Any, fn: Callable[[Any], Any]) -> Any:
    """Recursively walk dicts and lists; apply ``fn`` to every non-container leaf value."""
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
    """Recursively rebuild ``obj`` with parent references to a composite ``inner_workflow`` step rewritten.

    For each declared child output name, replaces ``$steps.{inner_step_name}.{out_name}`` with
    the corresponding inlined selector string. Bare ``$steps.{inner_step_name}`` (no property,
    not already ``...__``) becomes ``$steps.{first_inlined_step_name}`` so control-flow lists
    still target a concrete step after the block is removed.

    Non-string leaves are returned unchanged. Dict keys are not passed through ``fn``.
    """
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
    """Inline one leaf ``inner_workflow`` step into the parent ``workflow`` (mutates in place).

    The step at ``workflow["steps"][step_index]`` must be a nested workflow whose own ``steps``
    contain **no** further ``inner_workflow`` blocks (callers recurse first).

    Parses and validates the child definition, renames child steps to unique parent names,
    rewrites all string leaves in cloned child steps (inputs and ``$steps`` references), builds
    a map from child output names to rewritten selectors, patches the entire parent workflow so
    ``$steps.<inner_name>`` references target real steps, then replaces the composite step with
    the inlined step list.

    Raises:
        InnerWorkflowInvalidStepEntryError: Invalid step shape, bindings, empty child steps, or
            a child output selector that does not rewrite to a string.
    """
    steps = workflow.get("steps")

    inner_step = steps[step_index]
    inner_name = inner_step.get("name")

    if not isinstance(inner_name, str) or not inner_name:
        raise InnerWorkflowInvalidStepEntryError(
            "inner_workflow step requires a non-empty string `name`.",
        )

    inner = inner_step.get("workflow_definition")
    raw_bindings = inner_step.get("parameter_bindings") or {}

    if not isinstance(raw_bindings, dict):
        raise InnerWorkflowInvalidStepEntryError(
            f"inner_workflow step `{inner_name}` requires `parameter_bindings` object.",
        )

    inner_parsed = parse_workflow_definition(
        raw_workflow_definition=inner,
        available_blocks=available_blocks,
        profiler=profiler,
    )

    bindings = {str(k): str(v) for k, v in raw_bindings.items()}
    validate_parameter_bindings_against_child(
        bindings=bindings,
        child_parsed=inner_parsed,
        step_name=inner_name,
    )

    inner_input_defaults = _defaults_for_unbound_workflow_parameters(
        inner.get("inputs"),
        bindings,
    )

    used_names = _collect_step_names_at_level(steps)
    used_names.discard(inner_name)

    inner_steps = inner.get("steps") or []

    old_to_new: List[Tuple[str, str]] = []
    for inner_step in inner_steps:
        if not isinstance(inner_step, dict):
            raise InnerWorkflowInvalidStepEntryError(
                "inner_workflow step `{inner_name}` `steps` list entry must be a JSON object (dict), got "
                f"{type(inner_step).__name__}."
            )

        old_name = inner_step.get("name")
        if not isinstance(old_name, str) or not old_name:
            raise InnerWorkflowInvalidStepEntryError(
                "inner_workflow step `{inner_name}` `steps` list entry must have a non-empty string `name`."
            )

        new_name = _unique_prefixed_step_name(inner_name, old_name, used_names)
        old_to_new.append((old_name, new_name))

    old_to_new_sorted = sorted(old_to_new, key=lambda p: len(p[0]), reverse=True)

    def rewrite_leaf(leaf: Any) -> Any:
        return _rewrite_inner_scalar(
            leaf,
            step_pairs=old_to_new_sorted,
            bindings=bindings,
            input_defaults=inner_input_defaults,
        )

    inlined_steps: List[Dict[str, Any]] = []
    for inner_step in inner_steps:
        old_name = inner_step.get("name")
        new_name = next(nname for oname, nname in old_to_new if oname == old_name)
        cloned = copy.deepcopy(inner_step)
        cloned["name"] = new_name
        inlined_steps.append(_deep_map_leaves(cloned, rewrite_leaf))

    if not inlined_steps:
        raise InnerWorkflowInvalidStepEntryError(
            f"inner_workflow step `{inner_name}` has no steps to inline.",
        )

    first_inlined_step_name = str(inlined_steps[0]["name"])

    inner_output_name_to_selector: Dict[str, str] = {}
    for inner_output in inner.get("outputs") or []:
        if not isinstance(inner_output, dict) or inner_output.get("type") != "JsonField":
            # Here we continue as a output definition can hold None values.
            continue

        inner_output_name = inner_output.get("name")
        inner_output_selector = inner_output.get("selector")
        if not isinstance(inner_output_name, str) or not isinstance(inner_output_selector, str):
            raise InnerWorkflowInvalidStepEntryError(
                f"inner_workflow `{inner_name}` child output `{inner_output_name}` selector must be a string, got "
                f"{type(inner_output_selector).__name__}."
            )

        rewritten = _rewrite_inner_scalar(
            inner_output_selector,
            step_pairs=old_to_new_sorted,
            bindings=bindings,
            input_defaults=inner_input_defaults,
        )

        if not isinstance(rewritten, str):
            raise InnerWorkflowInvalidStepEntryError(
                f"inner_workflow `{inner_name}` child output `{inner_output_name}` selector must rewrite "
                f"to a string selector after inlining, got {type(rewritten).__name__}."
            )

        inner_output_name_to_selector[inner_output_name] = rewritten

    patched_workflow = _replace_inner_step_control_and_output_refs_in_object(
        workflow,
        inner_step_name=inner_name,
        output_name_to_selector=inner_output_name_to_selector,
        first_inlined_step_name=first_inlined_step_name,
    )

    workflow.clear()
    workflow.update(patched_workflow)

    steps = workflow.get("steps")

    new_steps = steps[:step_index] + inlined_steps + steps[step_index + 1 :]
    workflow["steps"] = new_steps


def _inline_one_inner_workflow_leaf(workflow: Dict[str, Any], *, available_blocks, profiler) -> bool:
    """Find the first ``inner_workflow`` step in ``workflow`` and inline one nested layer.

    If that step's nested ``steps`` still contain an ``inner_workflow``, recurses into the
    nested definition until a leaf inner is found and expanded in the **parent** ``workflow``,
    then returns ``True``. If there is no ``inner_workflow`` step at this level, returns ``False``.

    Mutates ``workflow`` when a leaf inner is expanded.

    Raises:
        InnerWorkflowInvalidStepEntryError: Invalid ``steps`` list or inner workflow shape.
    """
    steps = workflow.get("steps")

    if not isinstance(steps, list):
        raise InnerWorkflowInvalidStepEntryError(
            "Invalid workflow steps definition: must be a list of JSON objects (dicts), got "
            f"{type(steps).__name__}."
        )

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise InnerWorkflowInvalidStepEntryError(
                "Invalid workflow step: must be a JSON object (dict), got "
                f"{type(step).__name__}."
            )

        if step.get("type") != USE_INNER_WORKFLOW_BLOCK_TYPE:
            continue

        inner = step.get("workflow_definition")
        if not isinstance(inner, dict):
            raise InnerWorkflowInvalidStepEntryError(
                "Invalid workflow step: must be a JSON object (dict), got "
                f"{type(inner).__name__}."
            )

        inner_steps = inner.get("steps")
        if not isinstance(inner_steps, list):
            raise InnerWorkflowInvalidStepEntryError(
                "inner_workflow step `{inner_name}` requires `steps` list, got "
                f"{type(inner_steps).__name__}."
            )

        if _contains_inner_workflow_step(inner_steps):
            if _inline_one_inner_workflow_leaf(inner, available_blocks=available_blocks, profiler=profiler):
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


def inline_inner_workflow_steps(
    workflow_definition: Dict[str, Any],
    *,
    available_blocks: Any,
    profiler: Optional[WorkflowsProfiler] = None,
) -> Dict[str, Any]:
    """Return a deep copy of ``workflow_definition`` with every ``inner_workflow`` step inlined.

    Repeatedly expands innermost nested workflows until no ``roboflow_core/inner_workflow@v1``
    steps remain at any depth. The original dict is not modified.

    Args:
        workflow_definition: Parsed workflow JSON (root ``steps`` list).
        available_blocks: Block registry passed to the child workflow parser.
        profiler: Optional compiler profiler.

    Raises:
        InnerWorkflowInliningStructureError: If inlining cannot make progress (unexpected graph).

    Note:
        Composition and reference normalization must already be applied to the input.
    """
    root = copy.deepcopy(workflow_definition)
    while _contains_inner_workflow_step(root.get("steps")):
        if not _inline_one_inner_workflow_leaf(root, available_blocks=available_blocks, profiler=profiler):
            raise InnerWorkflowInliningStructureError(
                public_message=(
                    "Could not inline inner_workflow steps (unexpected nested structure). "
                    "Ensure inner workflow composition is valid."
                ),
            )
    return root
