"""Compile-time checks for video clip anchors (`video_clip_anchor` kind).

A `ClipAnchor` emitted by `roboflow_core/video_clip_recorder@v1` is an in-process
reference to frames the recorder holds. The checks below run once the execution
graph is built, when every step's inputs, outputs and control-flow branches are
known:

* binding an anchor into a step that executes outside this process (an inner
  workflow with `remote_dispatch`, Custom Python running remotely, a Roboflow model
  step running remotely) is an error, because the anchor cannot be deserialised on
  the other side;
* exposing an anchor as a workflow output is an error, because only its metadata
  would leave the process;
* a recorder that runs under flow control only sees the frames on which its branch
  executes, so clips lose pre-roll; that is a warning, not an error.

Anchor tracking is conservative. A step output declaring the anchor kind carries an
anchor. A step that receives an anchor through an input which does not itself
declare the anchor kind (Frame Delay, Buffer, Dimension Collapse, Expression, ...)
may pass it on in any output able to hold an arbitrary value: a wildcard (`*`),
`list_of_values` or `dictionary` output, or one without declared kinds. The
compiler cannot tell what such an output contains, so it is treated as carrying an
anchor; scalar outputs of such a step (a boolean, a number, a string) are not.
Known false positive: an Expression-style block that only reads metadata from an
anchor (e.g. its `availability`) and outputs a plain value through a wildcard
output is still treated as carrying one, so binding that value into a remote step
or a workflow output is rejected. Steps that declare the anchor kind on their input
(the Vision Events sink) are its intended consumers and never pass it through.
"""

import warnings
from typing import Dict, Iterable, List, Optional, Set, Type, Union

import networkx as nx
from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.environment import (
    WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
    WORKFLOWS_STEP_EXECUTION_MODE,
)
from roboflow_workflows.errors import ExecutionGraphStructureError
from roboflow_workflows.execution_engine.constants import (
    NODE_COMPILATION_OUTPUT_PROPERTY,
)
from roboflow_workflows.execution_engine.entities.types import (
    DICTIONARY_KIND,
    LIST_OF_VALUES_KIND,
    VIDEO_CLIP_ANCHOR_KIND,
    WILDCARD_KIND,
)
from roboflow_workflows.execution_engine.introspection.schema_parser import (
    parse_block_manifest,
)
from roboflow_workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    OutputNode,
    StepInputDefinition,
    StepNode,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.entities import (
    BLOCK_SOURCE as DYNAMIC_BLOCK_SOURCE,
)
from roboflow_workflows.execution_engine.v1.inner_workflow.constants import (
    INNER_WORKFLOW_EXECUTION_MODE_REMOTE_DISPATCH,
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)

logger = get_logger(__name__)

RECORDER_TYPE_PREFIX = "roboflow_core/video_clip_recorder@"
ROBOFLOW_MODEL_TYPE_PREFIX = "roboflow_core/roboflow_"
ROBOFLOW_MODEL_TYPE_MARKER = "_model@"
LOCAL_EXECUTION_MODE = "local"
REMOTE_EXECUTION_MODE = "remote"
ERROR_CONTEXT = "workflow_compilation | video_clip_anchor_boundary"
# Output kinds able to hold an anchor received through an input of another kind.
CONTAINER_KIND_NAMES = frozenset(
    {WILDCARD_KIND.name, LIST_OF_VALUES_KIND.name, DICTIONARY_KIND.name}
)


class VideoClipRecorderUnderFlowControlWarning(UserWarning):
    """A video clip recorder placed under flow control loses pre-roll."""


def verify_video_clip_anchors(
    execution_graph: nx.DiGraph,
    *,
    available_blocks: Iterable[BlockSpecification],
    step_execution_mode: Union[str, object, None] = None,
    custom_python_execution_mode: Optional[str] = None,
) -> List[str]:
    """Run every video clip anchor check on a compiled execution graph.

    This is the single entry point the compiler calls. It raises on the first
    anchor that would leave the process and warns about recorders under flow
    control. The compiler runs it before caching a compiled graph, and the cache
    key does not include init parameters, so a per-engine `step_execution_mode`
    override only takes effect for definitions not compiled before.

    Args:
        execution_graph: The execution graph built by the compiler.
        available_blocks: Every block the workflow may use, static and dynamic.
        step_execution_mode: The engine's step execution mode (`local` or
            `remote`, as a string or `StepExecutionMode`); the process default
            when None.
        custom_python_execution_mode: The Custom Python execution mode (`local`
            or `modal`); the process default when None.

    Returns:
        Names of the recorder steps that were warned about for flow control.

    Raises:
        ExecutionGraphStructureError: If an anchor is bound into a step that
            executes outside this process, or exposed as a workflow output.
    """
    tainted = find_anchor_carrying_selectors(execution_graph=execution_graph)
    if tainted:
        verify_video_clip_anchor_boundaries(
            execution_graph,
            available_blocks=available_blocks,
            tainted_selectors=tainted,
            step_execution_mode=step_execution_mode,
            custom_python_execution_mode=custom_python_execution_mode,
        )
        verify_video_clip_anchors_not_in_outputs(
            execution_graph,
            tainted_selectors=tainted,
        )
    warned_recorders = warn_on_conditional_video_clip_recorders(execution_graph)

    return warned_recorders


def verify_video_clip_anchor_boundaries(
    execution_graph: nx.DiGraph,
    *,
    available_blocks: Iterable[BlockSpecification],
    tainted_selectors: Set[str],
    step_execution_mode: Union[str, object, None] = None,
    custom_python_execution_mode: Optional[str] = None,
) -> None:
    """Reject anchors bound into steps that execute outside this process.

    Args:
        execution_graph: The execution graph built by the compiler.
        available_blocks: Every block the workflow may use, static and dynamic.
        tainted_selectors: Selectors that may carry an anchor, from
            `find_anchor_carrying_selectors`.
        step_execution_mode: The engine's step execution mode; the process
            default when None.
        custom_python_execution_mode: The Custom Python execution mode; the
            process default when None.

    Raises:
        ExecutionGraphStructureError: If an anchor reaches a remote boundary.
    """
    sources = {spec.manifest_class: spec.block_source for spec in available_blocks}
    step_mode = _normalise_mode(
        step_execution_mode, default=WORKFLOWS_STEP_EXECUTION_MODE
    )
    python_mode = _normalise_mode(
        custom_python_execution_mode, default=WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE
    )
    for step_node in _iterate_step_nodes(execution_graph=execution_graph):
        boundary = _remote_boundary(
            step_node=step_node,
            sources=sources,
            step_execution_mode=step_mode,
            custom_python_execution_mode=python_mode,
        )
        if boundary is None:
            continue

        for definition in _iterate_input_definitions(step_node=step_node):
            selector = getattr(definition, "selector", None)
            if not _is_tainted(selector=selector, tainted_selectors=tainted_selectors):
                continue

            parameter = definition.parameter_specification.parameter_name
            raise ExecutionGraphStructureError(
                public_message=(
                    f"Step `{step_node.name}` binds `{selector}` to `{parameter}`, "
                    f"which may carry a `{VIDEO_CLIP_ANCHOR_KIND.name}`, but the step "
                    f"runs {boundary}. A video clip anchor refers to frames held by a "
                    "Video Clip Recorder in this process and cannot cross that "
                    "boundary: attach the clip with a Vision Events sink in this "
                    "workflow instead."
                ),
                context=ERROR_CONTEXT,
            )


def verify_video_clip_anchors_not_in_outputs(
    execution_graph: nx.DiGraph,
    *,
    tainted_selectors: Set[str],
) -> None:
    """Reject workflow outputs that may expose an anchor.

    Args:
        execution_graph: The execution graph built by the compiler.
        tainted_selectors: Selectors that may carry an anchor, from
            `find_anchor_carrying_selectors`.

    Raises:
        ExecutionGraphStructureError: If a workflow output is bound to a selector
            that may carry an anchor.
    """
    for node_name in execution_graph.nodes:
        output_node = execution_graph.nodes[node_name].get(
            NODE_COMPILATION_OUTPUT_PROPERTY
        )
        if not isinstance(output_node, OutputNode):
            continue

        selector = output_node.output_manifest.selector
        if not _is_tainted(selector=selector, tainted_selectors=tainted_selectors):
            continue

        raise ExecutionGraphStructureError(
            public_message=(
                f"Workflow output `{output_node.output_manifest.name}` is bound to "
                f"`{selector}`, which may carry a `{VIDEO_CLIP_ANCHOR_KIND.name}`. A "
                "video clip anchor is an in-process reference that cannot be "
                "returned from a workflow: feed it into a Vision Events sink and "
                "output the sink's results instead."
            ),
            context=ERROR_CONTEXT,
        )


def warn_on_conditional_video_clip_recorders(execution_graph: nx.DiGraph) -> List[str]:
    """Warn when a video clip recorder executes only on some frames.

    Args:
        execution_graph: The execution graph built by the compiler.

    Returns:
        Names of the recorder steps that were warned about.
    """
    warned = []
    for step_node in _iterate_step_nodes(execution_graph=execution_graph):
        if not step_node.step_manifest.type.startswith(RECORDER_TYPE_PREFIX):
            continue

        if not step_node.execution_branches_impacting_inputs:
            continue

        message = (
            f"Step `{step_node.name}` (Video Clip Recorder) runs under flow control "
            f"({sorted(step_node.execution_branches_impacting_inputs)}). It only "
            "buffers frames on which it executes, so clips lose pre-roll from frames "
            "where the condition was false. Place the recorder on the unconditional "
            "path and gate the sink instead."
        )
        logger.warning(message)
        warnings.warn(message, VideoClipRecorderUnderFlowControlWarning)
        warned.append(step_node.name)

    return warned


def find_anchor_carrying_selectors(execution_graph: nx.DiGraph) -> Set[str]:
    """Return the step output selectors that may carry an anchor.

    A step output declaring the anchor kind carries one. So does any container
    output (wildcard, `list_of_values`, `dictionary`, or no declared kind) of a
    step that receives an anchor-carrying selector through an input that does
    not declare the anchor kind, since the compiler cannot tell what such an
    output contains (see the module docstring for the known false positive).

    Args:
        execution_graph: The execution graph built by the compiler.

    Returns:
        Selectors such as `$steps.recorder.anchor`.
    """
    tainted: Set[str] = set()
    for node_name in nx.topological_sort(execution_graph):
        step_node = _step_node(execution_graph=execution_graph, node_name=node_name)
        if step_node is None:
            continue

        passes_anchor_through = _receives_anchor_as_opaque_input(
            step_node=step_node, tainted_selectors=tainted
        )
        for output in step_node.step_manifest.get_actual_outputs():
            kind_names = _kind_names(kinds=output.kind)
            selector = f"{step_node.selector}.{output.name}"
            if VIDEO_CLIP_ANCHOR_KIND.name in kind_names:
                tainted.add(selector)
            elif passes_anchor_through and _may_hold_any_value(kind_names):
                tainted.add(selector)

    return tainted


def _may_hold_any_value(kind_names: Set[str]) -> bool:
    may_hold_any_value = not kind_names or bool(kind_names & CONTAINER_KIND_NAMES)

    return may_hold_any_value


def _receives_anchor_as_opaque_input(
    step_node: StepNode, tainted_selectors: Set[str]
) -> bool:
    anchor_parameters = _parameters_declaring_anchor_kind(
        manifest_class=type(step_node.step_manifest)
    )
    for definition in _iterate_input_definitions(step_node=step_node):
        selector = getattr(definition, "selector", None)
        if not _is_tainted(selector=selector, tainted_selectors=tainted_selectors):
            continue

        if definition.parameter_specification.parameter_name in anchor_parameters:
            continue

        return True

    return False


def _parameters_declaring_anchor_kind(manifest_class: Type) -> Set[str]:
    try:
        manifest_metadata = parse_block_manifest(manifest_type=manifest_class)
    except Exception:
        # Dynamic or unusual manifests that cannot be introspected are treated as
        # pass-through, which keeps the check conservative.
        return set()

    parameters = set()
    for name, selector_definition in manifest_metadata.selectors.items():
        for reference in selector_definition.allowed_references:
            if VIDEO_CLIP_ANCHOR_KIND.name in _kind_names(kinds=reference.kind):
                parameters.add(name)

    return parameters


def _remote_boundary(
    step_node: StepNode,
    sources: Dict[Type, str],
    step_execution_mode: str,
    custom_python_execution_mode: str,
) -> Optional[str]:
    manifest = step_node.step_manifest
    if manifest.type == USE_INNER_WORKFLOW_BLOCK_TYPE:
        execution_mode = getattr(manifest, "execution_mode", None)
        if execution_mode == INNER_WORKFLOW_EXECUTION_MODE_REMOTE_DISPATCH:
            return "on a remote server (inner workflow with `remote_dispatch`)"

        return None

    is_custom_python = sources.get(type(manifest)) == DYNAMIC_BLOCK_SOURCE
    if is_custom_python and custom_python_execution_mode != LOCAL_EXECUTION_MODE:
        return (
            "remotely (Custom Python with execution mode "
            f"`{custom_python_execution_mode}`)"
        )

    is_roboflow_model = manifest.type.startswith(ROBOFLOW_MODEL_TYPE_PREFIX) and (
        ROBOFLOW_MODEL_TYPE_MARKER in manifest.type
    )
    if is_roboflow_model and step_execution_mode == REMOTE_EXECUTION_MODE:
        return "on a remote inference server (step execution mode `remote`)"

    return None


def _normalise_mode(value: Union[str, object, None], default: str) -> str:
    if value is None:
        value = default
    value = getattr(value, "value", value)
    normalised = str(value).strip().lower()

    return normalised


def _is_tainted(selector: Optional[str], tainted_selectors: Set[str]) -> bool:
    if not selector or not selector.startswith("$steps."):
        return False

    if selector in tainted_selectors:
        return True

    step_selector, _, output_name = selector.rpartition(".")
    if output_name != "*":
        return False

    prefix = f"{step_selector}."
    whole_step_tainted = any(
        tainted.startswith(prefix) for tainted in tainted_selectors
    )

    return whole_step_tainted


def _iterate_step_nodes(execution_graph: nx.DiGraph) -> Iterable[StepNode]:
    for node_name in execution_graph.nodes:
        step_node = _step_node(execution_graph=execution_graph, node_name=node_name)
        if step_node is not None:
            yield step_node


def _step_node(execution_graph: nx.DiGraph, node_name: str) -> Optional[StepNode]:
    definition = execution_graph.nodes[node_name].get(NODE_COMPILATION_OUTPUT_PROPERTY)
    if isinstance(definition, StepNode):
        return definition

    return None


def _iterate_input_definitions(step_node: StepNode) -> Iterable[StepInputDefinition]:
    for definition in step_node.input_data.values():
        if definition.is_compound_input():
            yield from definition.iterate_through_definitions()
        else:
            yield definition


def _kind_names(kinds: Iterable) -> Set[str]:
    names = {getattr(kind, "name", kind) for kind in kinds}

    return names
