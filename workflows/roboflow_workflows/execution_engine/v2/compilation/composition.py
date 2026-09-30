"""Compose nested workflows into scopes and resolve selectors through them.

A nested workflow step does not become a plan step. It opens a child scope
whose block steps get tuple paths under the nested step's name::

    root scope ()                               child scope ("child",)
      $inputs.values       [inputs]               $inputs.x ──alias──▶ $steps.expand.child
      $steps.expand.child  [inputs, expand:child] $inputs.k ──default─▶ Constant(100)
      $steps.child.y ◀────────── child output y ◀── $steps.scale.scaled
                                                   (plan path ("child", "scale"))

A child input is a boundary, not a step: selectors reaching it resolve to the
parent's source (the origin), so the parent's lineage is kept and the child's
declared dimensionality is not enforced (V1 behaviour). A literal binding or a
child default is a ``Constant`` origin. Consumers read the value through the
innermost child input's ``ChildInputPort``, which checks (and for a constant
decodes) it once per run against the child's declared kinds. A control target
naming a nested step expands to every block step inside it.

Saved children are fetched through the caller's resolver once per compile and
copied for each use, so neither the caller's nor the resolver's data is
mutated. A reference repeating on its ancestor path is a cycle; two uses of
one child (a diamond) are valid. The cycle, depth and count checks run before
a fetch, as in V1.
"""

import copy
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from roboflow_workflows.execution_engine.v2.compilation.definition import (
    BlockStepDeclaration,
    NestedStepDeclaration,
    WorkflowDeclaration,
    WorkflowInputDeclaration,
    WorkflowOutputDeclaration,
    WorkflowReference,
    is_selector_text,
    parse_workflow,
    require_data_selector,
)
from roboflow_workflows.execution_engine.v2.declaration import parse_selector
from roboflow_workflows.execution_engine.v2.errors import (
    CycleError,
    FieldPath,
    NestedWorkflowError,
    SelectorError,
    StepPath,
    WorkflowCompileError,
    format_step_path,
)
from roboflow_workflows.execution_engine.v2.plan import (
    ChildInputPort,
    ChildOutputPort,
    CompileOptions,
    Constant,
    InputPort,
    Source,
    StepPort,
)

ReferenceResolver = Callable[[WorkflowReference], Mapping[str, Any]]
"""Caller function returning the definition of a saved workflow."""


@dataclass(frozen=True)
class Hop:
    """One child boundary passed on the way from an origin to its consumer.

    Args:
        port: ``ChildInputPort`` of a child input, or ``ChildOutputPort`` of
            a child output that does not come from a child step.
        declaration: The child's input or output declaration.
    """

    port: Union[ChildInputPort, ChildOutputPort]
    declaration: Union[WorkflowInputDeclaration, WorkflowOutputDeclaration]


@dataclass(frozen=True)
class Resolution:
    """Where the value of a data selector comes from.

    Args:
        origin: Root input, step output or compile-time constant holding the
            value.
        hops: Child boundaries the value passes, from the origin outwards to
            the consumer. A constant origin always starts with the child input
            whose literal or default it is.
    """

    origin: Source
    hops: Tuple[Hop, ...] = ()

    @property
    def port(self) -> Source:
        """What a consumer reads: the last boundary passed, else the origin."""
        if not self.hops:
            return self.origin

        return self.hops[-1].port

    def boundaries(self) -> Iterator[Tuple[Hop, Source]]:
        """Yield each boundary passed, origin side first, with what it reads.

        Yields:
            The hop and its source: the origin for the first hop, the previous
            hop's port otherwise.
        """
        source = self.origin
        for hop in self.hops:
            yield hop, source
            source = hop.port

    def then(self, hop: Hop) -> "Resolution":
        """Return this resolution continued through one more boundary."""
        continued = Resolution(origin=self.origin, hops=self.hops + (hop,))

        return continued


@dataclass
class Scope:
    """One workflow of the composition tree.

    Args:
        path: ``()`` for the root, otherwise the nested step's path.
        workflow: The parsed workflow.
        parent: Enclosing scope; ``None`` for the root.
        bindings: For a child: child input name to the parent's raw binding,
            a selector string or a literal.
        children: Nested step name to its child scope.
    """

    path: StepPath
    workflow: WorkflowDeclaration
    parent: Optional["Scope"] = None
    bindings: Mapping[str, Any] = field(default_factory=dict)
    children: Dict[str, "Scope"] = field(default_factory=dict)
    _constants: Dict[str, Constant] = field(default_factory=dict, repr=False)

    def step_path(self, name: str) -> StepPath:
        """Return the plan path of a step declared in this scope.

        Args:
            name: Step name in this scope.

        Returns:
            The scope path followed by ``name``.
        """
        return self.path + (name,)

    def block_steps(self) -> Iterator[Tuple["Scope", BlockStepDeclaration]]:
        """Yield every block step, expanding children where they are declared.

        Yields:
            The declaring scope and the step.
        """
        for step in self.workflow.steps:
            if isinstance(step, NestedStepDeclaration):
                yield from self.children[step.name].block_steps()
                continue
            yield self, step

    def resolve_data(
        self,
        selector: str,
        *,
        location: str,
        step_path: StepPath,
        field_path: FieldPath = (),
        _visiting: Optional[Set[Tuple[StepPath, str]]] = None,
    ) -> Resolution:
        """Resolve a data selector written in this scope.

        Args:
            selector: ``$inputs.<name>``, ``$steps.<step>.<output>`` or
                ``$steps.<step>.*``.
            location: Where the selector is written, for messages.
            step_path: Step (or nested workflow) holding the selector,
                reported by errors.
            field_path: Field path of the selector in that step.

        Returns:
            The resolved source and the child inputs passed on the way.

        Raises:
            SelectorError: For an unknown input, step or child output, or a
                wildcard over a nested workflow.
            CycleError: When child bindings and outputs refer to each other.
        """
        visiting = _visiting if _visiting is not None else set()
        if (self.path, selector) in visiting:
            raise CycleError(
                f"{location}: {selector!r} refers back to itself through nested "
                "workflow bindings and outputs",
                step_path=step_path,
                field_path=field_path,
            )
        visiting.add((self.path, selector))

        parsed = parse_selector(selector)
        if parsed.target == "input":
            resolution = self._resolve_input(
                parsed.name,
                location=location,
                step_path=step_path,
                field_path=field_path,
                visiting=visiting,
            )
            return resolution

        step = self._find_step(
            parsed.name,
            selector=selector,
            location=location,
            step_path=step_path,
            field_path=field_path,
        )
        if isinstance(step, BlockStepDeclaration):
            port = StepPort(step=self.step_path(step.name), output=parsed.output)
            return Resolution(origin=port)

        child = self.children[step.name]
        child_outputs = [output.name for output in child.workflow.outputs]
        if parsed.output == "*":
            raise SelectorError(
                f"{location}: {selector!r} selects every output of nested workflow "
                f"{format_step_path(child.path)}; select its outputs "
                f"{child_outputs} by name",
                step_path=step_path,
                field_path=field_path,
            )
        output = next(
            (item for item in child.workflow.outputs if item.name == parsed.output),
            None,
        )
        if output is None:
            raise SelectorError(
                f"{location}: nested workflow {format_step_path(child.path)} has no "
                f"output {parsed.output!r}; its outputs are {child_outputs}",
                step_path=step_path,
                field_path=field_path,
            )
        inner = child.resolve_data(
            output.selector,
            location=output.location,
            step_path=child.path,
            field_path=("outputs", output.name),
            _visiting=visiting,
        )
        if not inner.hops:
            # A child step's output: the step itself carries the child's gates.
            return inner

        # A forwarded input, literal or default, or a deeper such output,
        # passes no child step; its own boundary applies the child's gates.
        port = ChildOutputPort(scope=child.path, name=output.name)
        resolution = inner.then(Hop(port=port, declaration=output))

        return resolution

    def resolve_targets(
        self,
        selector: str,
        *,
        location: str,
        step_path: StepPath,
        field_path: FieldPath,
    ) -> Tuple[StepPath, ...]:
        """Resolve a control target ``$steps.<name>`` written in this scope.

        Args:
            selector: Step selector.
            location: Where it is written, for messages.
            step_path: Controller step, reported by errors.
            field_path: Field path of the target in the controller.

        Returns:
            The governed plan paths: the step itself, or every block step of
            a nested workflow (decision 007).

        Raises:
            SelectorError: When the step does not exist.
        """
        name = parse_selector(selector).name
        step = self._find_step(
            name,
            selector=selector,
            location=location,
            step_path=step_path,
            field_path=field_path,
        )
        if isinstance(step, BlockStepDeclaration):
            return (self.step_path(name),)

        targets = tuple(
            scope.step_path(block.name)
            for scope, block in self.children[name].block_steps()
        )

        return targets

    def _find_step(
        self,
        name: str,
        *,
        selector: str,
        location: str,
        step_path: StepPath,
        field_path: FieldPath,
    ) -> Any:
        step = self.workflow.step(name)
        if step is None:
            raise SelectorError(
                f"{location} references unknown step {name!r} via {selector!r}; "
                f"steps here: {[item.name for item in self.workflow.steps]}",
                step_path=step_path,
                field_path=field_path,
            )

        return step

    def _resolve_input(
        self,
        name: str,
        *,
        location: str,
        step_path: StepPath,
        field_path: FieldPath,
        visiting: Set[Tuple[StepPath, str]],
    ) -> Resolution:
        declaration = self.workflow.inputs.get(name)
        if declaration is None:
            raise SelectorError(
                f"{location} references unknown workflow input {name!r}; inputs "
                f"here: {list(self.workflow.inputs)}",
                step_path=step_path,
                field_path=field_path,
            )
        if self.parent is None:
            return Resolution(origin=InputPort(name=name))

        hop = Hop(
            port=ChildInputPort(scope=self.path, name=name), declaration=declaration
        )
        bound = self.bindings.get(name, declaration.default)
        if name not in self.bindings or not is_selector_text(bound):
            return Resolution(origin=self._constant(name, bound), hops=(hop,))

        outer = self.parent.resolve_data(
            bound,
            location=f"{format_step_path(self.path)} parameter_bindings.{name}",
            step_path=self.path,
            field_path=("parameter_bindings", name),
            _visiting=visiting,
        )
        resolution = outer.then(hop)

        return resolution

    def _constant(self, name: str, value: Any) -> Constant:
        # One Constant per child input: every resolution through this input
        # yields the same boundary source, whichever consumer asked.
        if name not in self._constants:
            self._constants[name] = Constant(value=value)

        return self._constants[name]


@dataclass(frozen=True)
class DynamicDefinition:
    """One collected ``dynamic_blocks_definitions`` entry.

    Args:
        definition: The raw entry.
        location: Structural definition path, e.g.
            ``"steps[1].workflow_definition.dynamic_blocks_definitions[0]"``.
    """

    definition: Any
    location: str


@dataclass(frozen=True)
class Composition:
    """A composed definition.

    Args:
        root: Root scope.
        dynamic_definitions: Dynamic block definitions of every scope, root
            first, after the duplicate policy.
        warnings: Non-fatal findings, e.g. conflicting dynamic duplicates.
    """

    root: Scope
    dynamic_definitions: Tuple[DynamicDefinition, ...]
    warnings: Tuple[str, ...]


def compose_workflow(
    definition: Mapping[str, Any],
    *,
    options: CompileOptions,
    reference_resolver: Optional[ReferenceResolver],
) -> Composition:
    """Parse a definition and compose its nested workflows into scopes.

    Dynamic block definitions are collected root first, then each child where
    it is declared. When several definitions share a ``manifest.block_type``,
    the first one wins (V1 policy); an identical later copy is dropped
    silently and a different one is dropped with a warning naming both
    locations.

    Args:
        definition: Root workflow definition; it is not modified.
        options: Nesting depth and count limits.
        reference_resolver: Returns saved workflow definitions; needed only
            when the definition references saved workflows.

    Returns:
        The scope tree, dynamic definitions and warnings.

    Raises:
        WorkflowCompileError: On a malformed definition.
        NestedWorkflowError: On invalid bindings, exceeded limits, an empty
            child or a missing resolver.
        CycleError: When saved references form a cycle.
    """
    composer = _Composer(options=options, resolver=reference_resolver)
    root_declaration = parse_workflow(_detached(definition, what="definition"))
    root = composer.compose(root_declaration, path=(), ancestors=(), depth=0)
    dynamic_definitions, warnings = _apply_duplicate_policy(
        composer.dynamic_definitions
    )

    composition = Composition(
        root=root,
        dynamic_definitions=dynamic_definitions,
        warnings=warnings,
    )

    return composition


class _Composer:
    def __init__(
        self,
        *,
        options: CompileOptions,
        resolver: Optional[ReferenceResolver],
    ):
        self._options = options
        self._resolver = resolver
        self._fetched: Dict[WorkflowReference, Mapping[str, Any]] = {}
        self._nested_count = 0
        self.dynamic_definitions: List[DynamicDefinition] = []

    def compose(
        self,
        declaration: WorkflowDeclaration,
        *,
        path: StepPath,
        ancestors: Tuple[WorkflowReference, ...],
        depth: int,
        parent: Optional[Scope] = None,
        bindings: Optional[Mapping[str, Any]] = None,
    ) -> Scope:
        scope = Scope(
            path=path, workflow=declaration, parent=parent, bindings=bindings or {}
        )
        self.dynamic_definitions.extend(
            DynamicDefinition(
                definition=entry,
                location=f"{declaration.location}dynamic_blocks_definitions[{index}]",
            )
            for index, entry in enumerate(declaration.dynamic_blocks)
        )
        for step in declaration.steps:
            if isinstance(step, NestedStepDeclaration):
                scope.children[step.name] = self._compose_child(
                    step, parent=scope, ancestors=ancestors, depth=depth
                )

        return scope

    def _compose_child(
        self,
        step: NestedStepDeclaration,
        *,
        parent: Scope,
        ancestors: Tuple[WorkflowReference, ...],
        depth: int,
    ) -> Scope:
        path = parent.step_path(step.name)
        where = format_step_path(path)
        if step.reference is not None and step.reference in ancestors:
            chain = " -> ".join(
                item.describe() for item in ancestors + (step.reference,)
            )
            raise CycleError(
                f"{where}: saved workflow references form a cycle: {chain}",
                step_path=path,
            )
        if depth + 1 > self._options.max_nested_depth:
            raise NestedWorkflowError(
                f"{where}: nested workflow depth {depth + 1} exceeds the limit of "
                f"{self._options.max_nested_depth}",
                step_path=path,
            )
        self._nested_count += 1
        if self._nested_count > self._options.max_nested_count:
            raise NestedWorkflowError(
                f"{where}: {self._nested_count} nested workflow steps exceed the "
                f"limit of {self._options.max_nested_count}",
                step_path=path,
            )

        if step.reference is not None:
            child_ancestors = ancestors + (step.reference,)
            raw_child = self._fetch(step.reference, path=path)
            location = f"{step.location}<{step.reference.describe()}>."
        else:
            # A caller may reuse one child object for several steps; each use
            # gets its own copy so no literal or default is shared by accident.
            child_ancestors = ancestors
            raw_child = _detached(step.definition, what=f"{where} workflow_definition")
            location = f"{step.location}.workflow_definition."

        declaration = parse_workflow(raw_child, location=location)
        if not declaration.steps:
            raise NestedWorkflowError(
                f"{where}: nested workflow has no steps", step_path=path
            )
        _check_bindings(step, declaration=declaration, path=path)

        child = self.compose(
            declaration,
            path=path,
            ancestors=child_ancestors,
            depth=depth + 1,
            parent=parent,
            bindings=step.bindings,
        )

        return child

    def _fetch(
        self, reference: WorkflowReference, *, path: StepPath
    ) -> Mapping[str, Any]:
        what = f"saved workflow {reference.describe()}"
        if reference not in self._fetched:
            if self._resolver is None:
                raise NestedWorkflowError(
                    f"{format_step_path(path)} references {what}, but compile_workflow "
                    "received no reference_resolver",
                    step_path=path,
                )
            try:
                fetched = self._resolver(reference)
            except WorkflowCompileError:
                raise
            except Exception as error:
                raise NestedWorkflowError(
                    f"{format_step_path(path)}: resolving {what} failed: "
                    f"{type(error).__name__}: {error}",
                    step_path=path,
                ) from error
            self._fetched[reference] = _detached(fetched, what=what)

        definition = _detached(self._fetched[reference], what=what)

        return definition


def _check_bindings(
    step: NestedStepDeclaration, *, declaration: WorkflowDeclaration, path: StepPath
) -> None:
    where = format_step_path(path)
    unknown = sorted(set(step.bindings) - set(declaration.inputs))
    if unknown:
        raise NestedWorkflowError(
            f"{where}: parameter_bindings name unknown child inputs {unknown}; the "
            f"child declares {list(declaration.inputs)}",
            step_path=path,
        )

    missing = [
        name
        for name, child_input in declaration.inputs.items()
        if name not in step.bindings and child_input.default is None
    ]
    if missing:
        raise NestedWorkflowError(
            f"{where}: parameter_bindings miss required child inputs {missing}; only "
            "child inputs with a non-null default_value may be omitted",
            step_path=path,
        )

    for name, value in step.bindings.items():
        if is_selector_text(value):
            require_data_selector(
                value,
                location=f"{where} parameter_bindings.{name}",
                step_path=path,
                field_path=("parameter_bindings", name),
            )


def _apply_duplicate_policy(
    collected: List[DynamicDefinition],
) -> Tuple[Tuple[DynamicDefinition, ...], Tuple[str, ...]]:
    kept: List[DynamicDefinition] = []
    first_by_type: Dict[str, DynamicDefinition] = {}
    warnings: List[str] = []
    for item in collected:
        block_type = _dynamic_block_type(item.definition)
        first = first_by_type.setdefault(block_type, item) if block_type else item
        if first is item:
            kept.append(item)
        elif _canonical(first.definition) != _canonical(item.definition):
            warnings.append(
                f"dynamic block definition at {item.location} redefines the block "
                f"type defined at {first.location}; the first definition is used"
            )

    return tuple(kept), tuple(warnings)


def _dynamic_block_type(definition: Any) -> Optional[str]:
    manifest = definition.get("manifest") if isinstance(definition, Mapping) else None
    block_type = manifest.get("block_type") if isinstance(manifest, Mapping) else None
    if isinstance(block_type, str) and block_type:
        return block_type

    return None


def _canonical(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=repr)

    return text


def _detached(value: Any, *, what: str) -> Any:
    try:
        copied = copy.deepcopy(value)
    except Exception as error:
        raise WorkflowCompileError(f"{what} cannot be copied: {error}") from error

    return copied
