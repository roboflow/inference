"""Dynamic blocks: V1-shaped custom Python definitions as ordinary V2 blocks.

A workflow may embed ``dynamic_blocks_definitions`` in V1's JSON shape: a
``DynamicBlockDefinition`` holding a ``ManifestDescription`` and ``PythonCode``.
The compiler collects them (root first, one definition per block type) and
calls ``build_dynamic_catalogue``, which returns a catalogue of generated
``Block`` subclasses in the ``dynamic_workflows_blocks`` namespace::

    definition --validate--> manifest --map--> Params, outputs, flags --+
               \\-> code --ast.parse--> run/init signatures, imports ---+--> class
    session:  __init__(shared_state, representation_policy)
                --> policy.check_compatibility(manifest)
                --> exec imports + run + init in a private module (globals = shared_state)
                --> init() once; its result is self._init_results
    each run: run(**kwargs) --> policy.prepare_inputs --> submitted run(self, **inputs)
                            --> policy.prepare_outputs --> V1 shapes adapted

Building classes, compiling and inspecting never execute submitted code: the
source is only parsed. The code runs when an execution session constructs the
step, and only when the catalogue was built with ``allow_local_code=True``.
It then runs in this process with its permissions; it is not sandboxed.
Remote execution services (V1's Modal mode) are a host transport outside this
engine.

Constructor resources (resolved like any block resource, namespace
``dynamic_workflows_blocks``):

* ``shared_state``: one mapping shared by every dynamic step of a session,
  nested copies included. The catalogue provides ``Factory(dict,
  scope="session")``, so a new session starts empty. A caller who wants state
  to outlive sessions passes its own mapping as
  ``resources={"dynamic_workflows_blocks.shared_state": mapping}``. Submitted
  code sees it as ``globals`` (V1's idiom; it shadows the builtin function)
  and as ``self.shared_state``. ``self`` and ``init`` state stay per step.
* ``representation_policy``: a ``RepresentationPolicy`` adapting payloads
  around the submitted code. The default passes payloads unchanged and
  rejects ``tensor_compatibility="tensor_native"``; a host supplies a capable
  policy to run such blocks.

``self.get_workflow_context()`` describes the step and run currently
executing (see ``context``).

Manifest mapping (V1 field -> V2 declaration):

====================================  ==========================================
``value_types``                       literal alternatives of the field
``selector_types``/``selector_data_kind``  one ``Ref``/``Group`` leaf, union of kinds
``is_optional``/``has_default_value``  ``Optional[...]`` / default (compound copied)
batch flags of the manifest           per-leaf batch mode (``_batch_mode``)
``dimensionality_offset`` + reference  ``Group`` for inputs one level below the
                                      reference input, ``Ref`` otherwise
``output_dimensionality_offset=1``    every output expands one new child axis
``output_dimensionality_offset=-1``   batch-capable inputs become ``Group`` fields
``accepts_empty_values``              ``accepts_empty``
``description``, ``ui_manifest``, ...  docstring and ``metadata``
====================================  ==========================================

Submitted code sees small standard-library helpers (``typing`` names, ``math``,
``time``, ``json``, ``os``) plus V2's ``Batch``. Unlike V1, NumPy, OpenCV,
Supervision, Requests and Shapely are not pre-imported: list them in
``code.imports``. Each step instance loads its own module, so ``init`` state is
per step and per session. V1's ``globals`` was one process-wide dictionary by
default; here it is per session by default and shared further only when the
caller passes a mapping.
"""

import ast
import copy
import json
import linecache
import math
import os
import time
import types
import typing
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from pydantic import Field, ValidationError, create_model
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    DiscoveryProblem,
    DiscoveryProblemCode,
    RuntimeRestriction,
    Severity,
    WorkOperation,
    incomplete_discovery,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.entities import (
    BLOCK_SOURCE,
    DynamicBlockDefinition,
    DynamicInputDefinition,
    ManifestDescription,
    PythonCode,
    SelectorType,
    TensorCompatibility,
    ValueType,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.context import get_execution_context
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    RESERVED_PARAM_NAMES,
    Block,
    BlockParams,
    DependentResource,
    Group,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    WorkflowCompileError,
    WorkflowExecutionError,
)
from roboflow_workflows.execution_engine.v2.kinds import BUILTIN_KINDS, Kind
from roboflow_workflows.execution_engine.v2.resources import Factory

__all__ = [
    "BLOCK_SOURCE",
    "LEGACY_REPRESENTATION",
    "SHARED_STATE_FACTORY",
    "DynamicBlock",
    "DynamicBlockError",
    "DynamicCodeError",
    "LocalCodeNotAllowedError",
    "RepresentationError",
    "RepresentationPolicy",
    "build_dynamic_catalogue",
]

# One object for every dynamic catalogue, so merged catalogues never conflict.
SHARED_STATE_FACTORY = Factory(dict, scope="session")

EXPANDED_AXIS_KEY = "children"
ENGINE_COMPATIBILITY = ">=2.0.0,<3.0.0"

PYTHON_TYPES: Mapping[ValueType, Any] = {
    ValueType.ANY: Any,
    ValueType.INTEGER: int,
    ValueType.FLOAT: float,
    ValueType.BOOLEAN: bool,
    ValueType.DICT: dict,
    ValueType.LIST: list,
    ValueType.STRING: str,
}

# V1 accepts string literals for these kinds even without a "string" value type.
STRING_LITERAL_KINDS: FrozenSet[str] = frozenset({"rgb_color"})

IMAGE_SELECTOR_TYPES = (SelectorType.INPUT_IMAGE, SelectorType.STEP_OUTPUT_IMAGE)
IMAGE_KIND_NAME = "image"

BlockResult = Union[Dict[str, Any], List[Dict[str, Any]]]

PRELOADED_GLOBALS: Mapping[str, Any] = {
    "Any": typing.Any,
    "Dict": typing.Dict,
    "List": typing.List,
    "Optional": typing.Optional,
    "Set": typing.Set,
    "math": math,
    "time": time,
    "json": json,
    "os": os,
    "Batch": Batch,
    "BlockResult": BlockResult,
}

# Names V1 pre-imported into every dynamic module, and how to import them now.
V1_IMPLICIT_IMPORTS: Mapping[str, str] = {
    "np": "import numpy as np",
    "sv": "import supervision as sv",
    "cv2": "import cv2",
    "requests": "import requests",
    "shapely": "import shapely",
    "WorkflowImageData": "an import of the image payload class your catalogue uses",
}

CUSTOM_PYTHON_RESTRICTIONS: Tuple[RuntimeRestriction, ...] = (
    RuntimeRestriction(
        severity=Severity.HARD,
        code="custom_python_local_code_disallowed",
        note=(
            "Submitted Python runs only when the caller allows local code "
            "(allow_local_code=True); otherwise creating an execution session fails."
        ),
        applies_to_configuration={"allow_local_code": False},
    ),
    RuntimeRestriction(
        severity=Severity.SOFT,
        code="custom_python_not_sandboxed",
        note=(
            "Submitted Python runs inside the engine's own process with its "
            "permissions; it is not sandboxed."
        ),
    ),
    RuntimeRestriction(
        severity=Severity.HARD,
        code="custom_python_remote_transport_unavailable",
        note=(
            "This engine only runs submitted Python locally; remote execution "
            "(V1's Modal mode) is a host transport it does not provide."
        ),
        applies_to_configuration={"custom_python_execution_mode": "remote"},
    ),
)


class DynamicBlockError(WorkflowCompileError):
    """A dynamic block definition is invalid.

    Args:
        message: Human-readable explanation.
        block_type: Declared block type, when known.
        location: Part of the definition at fault, e.g. ``"inputs.threshold"``
            or ``"code.run_function_code line 3"``.
    """

    def __init__(
        self,
        message: str,
        *,
        block_type: Optional[str] = None,
        location: Optional[str] = None,
    ):
        subject = (
            f"Dynamic block {block_type!r}"
            if block_type
            else "Dynamic block definition"
        )
        where = f" ({location})" if location else ""
        super().__init__(f"{subject}{where}: {message}")
        self.block_type = block_type
        self.location = location


class LocalCodeNotAllowedError(WorkflowExecutionError):
    """A dynamic block was constructed although local code execution is off."""


class DynamicCodeError(WorkflowExecutionError):
    """Submitted code failed to load, failed in ``init`` or returned a bad shape."""


class RepresentationError(WorkflowExecutionError):
    """A representation policy rejected a block or failed to adapt its payloads."""


TENSOR_NATIVE_RESTRICTION = RuntimeRestriction(
    severity=Severity.HARD,
    code="custom_python_tensor_native_requires_policy",
    note=(
        "The block declares tensor_compatibility=tensor_native; it runs only with "
        "a representation_policy that supports it. The default policy rejects it."
    ),
)


class RepresentationPolicy:
    """Boundary between engine payloads and what submitted code expects.

    The engine calls a policy around every dynamic block:

    1. ``check_compatibility(manifest)`` once, when the step is constructed,
       before the code's imports and ``init`` run.
    2. ``prepare_inputs(inputs, manifest=manifest)`` before each call of the
       submitted ``run``; its mapping is what ``run`` receives.
    3. ``prepare_outputs(result, manifest=manifest)`` on the submitted
       ``run``'s raw result, before V1 result shapes are adapted.

    This base class is the default legacy policy: payloads pass unchanged,
    keeping their identity, and blocks declaring
    ``tensor_compatibility="tensor_native"`` are rejected. A host that bridges
    representations subclasses it, overrides the adaptation methods and
    accepts ``tensor_native`` in ``check_compatibility``. Supply it as the
    ``representation_policy`` resource; the step identity is available from
    ``get_execution_context()``.
    """

    def check_compatibility(self, manifest: ManifestDescription) -> None:
        """Reject a block this policy cannot serve.

        Args:
            manifest: The block's V1-shaped manifest.

        Raises:
            RepresentationError: For ``tensor_native`` blocks.
        """
        if manifest.tensor_compatibility is TensorCompatibility.TENSOR_NATIVE:
            raise RepresentationError(
                f"Dynamic block {manifest.block_type!r} declares "
                "tensor_compatibility=tensor_native, but the legacy representation "
                "policy passes payloads unchanged. Provide a representation_policy "
                "that supports it (resource "
                f"'{BLOCK_SOURCE}.representation_policy')."
            )

    def prepare_inputs(
        self, inputs: Dict[str, Any], *, manifest: ManifestDescription
    ) -> Dict[str, Any]:
        """Adapt the input payloads of one call; identity by default.

        Args:
            inputs: One value per manifest input, as the engine delivers them.
            manifest: The block's V1-shaped manifest.

        Returns:
            The keyword arguments passed to the submitted ``run``.
        """
        return inputs

    def prepare_outputs(self, result: Any, *, manifest: ManifestDescription) -> Any:
        """Adapt the raw result of one call; identity by default.

        Args:
            result: What the submitted ``run`` returned.
            manifest: The block's V1-shaped manifest.

        Returns:
            The result handed back to the engine.
        """
        return result


LEGACY_REPRESENTATION = RepresentationPolicy()

_POLICY_METHODS = ("check_compatibility", "prepare_inputs", "prepare_outputs")


def build_dynamic_catalogue(
    definitions: Iterable[Any],
    *,
    catalogue: Catalogue,
    allow_local_code: bool = False,
) -> Catalogue:
    """Turn dynamic block definitions into a catalogue of generated block classes.

    Only parses and validates: no submitted code is executed and no block is
    constructed. Kinds are looked up by name in ``catalogue`` and then among
    the built-in kinds, and the generated classes share those ``Kind`` objects.

    Args:
        definitions: V1-shaped ``DynamicBlockDefinition`` mappings or models,
            one per block type (the compiler applies the duplicate policy).
        catalogue: Catalogue the workflow compiles against; used for kinds
            and to reject block types it already registers.
        allow_local_code: Whether constructing the generated classes may
            execute the submitted code in this process. When ``False``,
            compilation and inspection still work; creating an execution
            session fails with ``LocalCodeNotAllowedError`` as the cause.

    Returns:
        A catalogue containing only the generated classes, in namespace
        ``dynamic_workflows_blocks``, with ``SHARED_STATE_FACTORY`` as the
        provider of their ``shared_state`` resource.

    Raises:
        DynamicBlockError: On a malformed definition, unknown kind, invalid
            manifest combination, syntax error or unusable ``run``/``init``
            signature, or a block type defined twice or already registered.
    """
    known_kinds = {kind.name: kind for kind in BUILTIN_KINDS}
    known_kinds.update(catalogue.kinds)
    raw_definitions = list(definitions)

    classes = []
    first_index: Dict[str, int] = {}
    for index, raw_definition in enumerate(raw_definitions):
        location = f"definition {index}" if len(raw_definitions) > 1 else None
        definition = _parse_definition(raw_definition, location=location)
        block_type = definition.manifest.block_type
        if block_type in first_index:
            raise DynamicBlockError(
                f"is defined by entries {first_index[block_type]} and {index}; "
                "pass one definition per block type",
                block_type=block_type,
            )
        if block_type in catalogue:
            owner = catalogue.entry(block_type).spec.block_class.__qualname__
            raise DynamicBlockError(
                f"is already registered by catalogue block {owner}; choose "
                "another block_type",
                block_type=block_type,
            )

        first_index[block_type] = index
        classes.append(
            _assemble_class(
                definition, kinds=known_kinds, allow_local_code=allow_local_code
            )
        )

    dynamic_catalogue = Catalogue(
        classes,
        namespace=BLOCK_SOURCE,
        providers={"shared_state": SHARED_STATE_FACTORY},
    )

    return dynamic_catalogue


class DynamicBlock(Block):
    """Base class of every generated dynamic block.

    Generated subclasses set ``type``, ``Params``, ``outputs`` and the class
    attributes below. Constructing one checks representation compatibility,
    loads the submitted code into a private module and calls its ``init``
    function once; ``run`` delegates to the submitted ``run(self, **inputs)``,
    which can read ``self._init_results``, keep its own state on ``self``
    across runs of a session and share state through ``globals``.

    Args:
        shared_state: Mapping shared by the dynamic steps that receive it; the
            catalogue provides one per session. ``None`` (direct construction
            outside a session) creates a private mapping.
        representation_policy: Payload boundary around the submitted code.

    Raises:
        LocalCodeNotAllowedError: When local code execution is off.
        TypeError: When ``shared_state`` is not a mutable mapping or the
            policy lacks one of its three methods.
        RepresentationError: When the policy rejects the block.
        DynamicCodeError: When loading the code or ``init`` fails.

    Class attributes:
        dynamic_definition: The validated V1-shaped definition.
        local_code_allowed: Whether construction may execute the code.
        declared_imports: Modules named by import statements in the code,
            found without importing them.
        expands: Whether outputs expand a new child axis
            (``output_dimensionality_offset=1``).
    """

    dynamic_definition: ClassVar[DynamicBlockDefinition]
    local_code_allowed: ClassVar[bool] = False
    declared_imports: ClassVar[Tuple[str, ...]] = ()
    expands: ClassVar[bool] = False

    def __init__(
        self,
        *,
        shared_state: Optional[MutableMapping[str, Any]] = None,
        representation_policy: RepresentationPolicy = LEGACY_REPRESENTATION,
    ):
        block_class = type(self)
        if not block_class.local_code_allowed:
            raise LocalCodeNotAllowedError(
                f"Dynamic block {block_class.type!r} runs submitted Python. Compile "
                "with allow_local_code=True to execute it in this process; the code "
                "is not sandboxed."
            )

        self.shared_state = _checked_shared_state(
            shared_state, block_type=block_class.type
        )
        self.representation_policy = _checked_policy(
            representation_policy, block_type=block_class.type
        )
        self._call_policy(
            "check_compatibility", block_class.dynamic_definition.manifest
        )

        code = block_class.dynamic_definition.code
        module = _load_code(
            code, block_type=block_class.type, shared_state=self.shared_state
        )
        self._run_function = getattr(module, code.run_function_name)
        # As in V1, a top-level function named like the init function is
        # called even when it was written in the run code.
        init_function = getattr(module, code.init_function_name, dict)
        self._init_results = _call_init(init_function, block_type=block_class.type)

    def run(self, **kwargs: Any) -> Any:
        """Call the submitted ``run`` between the representation policy hooks.

        Args:
            **kwargs: One value per manifest input.

        Returns:
            The submitted function's result after ``prepare_outputs``. With
            expanding outputs, V1's list of child dicts becomes one ``Batch``
            per output.

        Raises:
            NameError: With an import hint when the code uses a name V1
                pre-imported, such as ``np``.
            RepresentationError: When a policy hook fails.
            DynamicCodeError: When expanded children do not match the outputs.
        """
        manifest = type(self).dynamic_definition.manifest
        inputs = self._call_policy("prepare_inputs", kwargs, manifest=manifest)
        if not isinstance(inputs, Mapping):
            raise RepresentationError(
                f"Dynamic block {type(self).type!r}: prepare_inputs must return a "
                f"mapping of inputs, got {type(inputs).__name__}"
            )

        try:
            result = self._run_function(self, **inputs)
        except NameError as error:
            hint = _missing_import_hint(error)
            if hint is None:
                raise
            raise NameError(f"{error}. {hint}") from error

        prepared = self._call_policy("prepare_outputs", result, manifest=manifest)
        adapted = self._adapt_result(prepared)

        return adapted

    def get_workflow_context(self) -> Dict[str, Any]:
        """Describe the step and run executing this code, like V1's method.

        Returns:
            ``step_name``, ``step_selector``, ``block_type`` and
            ``workflow_execution_id`` (the run id; ``None`` inside ``init``),
            plus ``session_id``, ``step_path`` and the call's ``indices``.

        Raises:
            NoExecutionContextError: Outside a constructor or call run by the
                engine.
        """
        context = get_execution_context()
        description = {
            "step_name": context.step_path[-1] if context.step_path else None,
            "step_selector": context.step_selector,
            "block_type": context.block_type,
            "workflow_execution_id": context.run_id,
            "session_id": context.session_id,
            "step_path": list(context.step_path),
            "indices": [list(index) for index in context.indices],
        }

        return description

    @classmethod
    def discover_dependent_resources(cls, params: BlockParams) -> Discovery:
        """Declare imported modules; the code may use more, so incomplete."""
        items = [
            DependentResource(resource_type="python_module", identifier=module_name)
            for module_name in cls.declared_imports
        ]
        discovery = incomplete_discovery(items, [_custom_python_problem("resources")])

        return discovery

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> Discovery:
        """Declare custom Python work; what it does inside stays unknown."""
        discovery = incomplete_discovery(
            [WorkOperation.CUSTOM_PYTHON], [_custom_python_problem("operations")]
        )

        return discovery

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> Discovery:
        """Declare local-execution restrictions; the code may add more."""
        restrictions = list(CUSTOM_PYTHON_RESTRICTIONS)
        manifest = cls.dynamic_definition.manifest
        if manifest.tensor_compatibility is TensorCompatibility.TENSOR_NATIVE:
            restrictions.append(TENSOR_NATIVE_RESTRICTION)
        discovery = incomplete_discovery(
            restrictions, [_custom_python_problem("restrictions")]
        )

        return discovery

    def _call_policy(self, hook: str, *args: Any, **kwargs: Any) -> Any:
        try:
            result = getattr(self.representation_policy, hook)(*args, **kwargs)
        except RepresentationError:
            raise
        except Exception as error:
            raise RepresentationError(
                f"Dynamic block {type(self).type!r}: representation policy "
                f"{type(self.representation_policy).__name__}.{hook} failed: "
                f"{type(error).__name__}: {error}"
            ) from error

        return result

    def _adapt_result(self, result: Any) -> Any:
        if not type(self).expands or not isinstance(result, list):
            return result
        # V1: a per-invocation call returns a list of child dicts; a batch
        # call returns one such list per invocation.
        if result and all(isinstance(children, list) for children in result):
            adapted = [self._children_to_batches(children) for children in result]
            return adapted

        adapted = self._children_to_batches(result)

        return adapted

    def _children_to_batches(self, children: List[Any]) -> Dict[str, Batch]:
        names = list(type(self).outputs)
        for position, child in enumerate(children):
            if not isinstance(child, Mapping) or set(child) != set(names):
                found = sorted(child) if isinstance(child, Mapping) else type(child)
                raise DynamicCodeError(
                    f"Dynamic block {type(self).type!r} returned child {position} "
                    f"as {found}; each child must be a dict with exactly the outputs "
                    f"{names}"
                )

        batches = {
            name: Batch.of([child[name] for child in children]) for name in names
        }

        return batches


def _parse_definition(raw: Any, *, location: Optional[str]) -> DynamicBlockDefinition:
    if isinstance(raw, DynamicBlockDefinition):
        return raw

    try:
        definition = DynamicBlockDefinition.model_validate(raw)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in problem['loc']) or '<root>'}: "
            f"{problem['msg']}"
            for problem in error.errors(include_url=False)
        )
        manifest = raw.get("manifest") if isinstance(raw, Mapping) else None
        block_type = (
            manifest.get("block_type") if isinstance(manifest, Mapping) else None
        )
        raise DynamicBlockError(
            f"is malformed: {problems}",
            block_type=block_type if isinstance(block_type, str) else None,
            location=location,
        ) from error

    return definition


def _assemble_class(
    definition: DynamicBlockDefinition,
    *,
    kinds: Mapping[str, Kind],
    allow_local_code: bool,
) -> type:
    manifest = definition.manifest
    block_type = manifest.block_type
    _check_input_names(manifest)
    _check_batch_flags(manifest)

    group_inputs = _group_inputs(manifest)
    fields = {
        name: _params_field(
            name,
            input_definition,
            manifest=manifest,
            group=name in group_inputs,
            kinds=kinds,
        )
        for name, input_definition in manifest.inputs.items()
    }
    params_model = create_model(
        f"DynamicParams[{block_type}]", __base__=BlockParams, **fields
    )
    expands = manifest.output_dimensionality_offset == 1
    outputs = {
        name: Output(
            *_resolve_kinds(
                output.kind,
                kinds=kinds,
                block_type=block_type,
                location=f"outputs.{name}",
            ),
            expand=EXPANDED_AXIS_KEY if expands else None,
        )
        for name, output in manifest.outputs.items()
    }
    declared_imports = _inspect_code(
        definition.code, block_type=block_type, inputs=list(manifest.inputs)
    )

    namespace = {
        "__doc__": manifest.description or "",
        "__module__": __name__,
        "type": block_type,
        "Params": params_model,
        "outputs": outputs,
        "accepts_empty": manifest.accepts_empty_values,
        "engine_compatibility": ENGINE_COMPATIBILITY,
        "metadata": _metadata(
            definition,
            declared_imports=declared_imports,
            allow_local_code=allow_local_code,
        ),
        "dynamic_definition": definition,
        "local_code_allowed": allow_local_code,
        "declared_imports": declared_imports,
        "expands": expands,
    }
    try:
        block_class = type(f"DynamicBlock[{block_type}]", (DynamicBlock,), namespace)
    except ContractError as error:
        raise DynamicBlockError(str(error), block_type=block_type) from error

    return block_class


def _check_input_names(manifest: ManifestDescription) -> None:
    for name in manifest.inputs:
        if not name.isidentifier() or name.startswith("_"):
            raise DynamicBlockError(
                "input names must be Python identifiers not starting with '_'",
                block_type=manifest.block_type,
                location=f"inputs.{name}",
            )
        if name in RESERVED_PARAM_NAMES:
            raise DynamicBlockError(
                f"input name {name!r} is reserved for the step itself",
                block_type=manifest.block_type,
                location=f"inputs.{name}",
            )


def _check_batch_flags(manifest: ManifestDescription) -> None:
    flags = (
        "batch_oriented_parameters",
        "parameters_with_scalars_and_batches",
        "get_parameters_enforcing_auto_batch_casting",
    )
    for flag in flags:
        for name in getattr(manifest, flag):
            input_definition = manifest.inputs.get(name)
            if input_definition is None or not input_definition.selector_types:
                raise DynamicBlockError(
                    f"names {name!r}, which is not an input accepting selectors",
                    block_type=manifest.block_type,
                    location=f"manifest.{flag}",
                )


def _group_inputs(manifest: ManifestDescription) -> FrozenSet[str]:
    """Inputs that receive the group of children of each invocation.

    V1 expresses grouping through dimensionality offsets. With differing input
    offsets, inputs one level deeper than the reference input are groups and
    the output stays at the reference level. With output offset -1, every
    input that can receive batch data is a group and the block runs once per
    parent. V1's own manifest rules are enforced with the same meaning.
    """
    block_type = manifest.block_type
    selector_inputs = {
        name: definition
        for name, definition in manifest.inputs.items()
        if definition.selector_types
    }
    for name, definition in manifest.inputs.items():
        if name in selector_inputs:
            continue
        if definition.dimensionality_offset or definition.is_dimensionality_reference:
            raise DynamicBlockError(
                "declares dimensionality but accepts no selector",
                block_type=block_type,
                location=f"inputs.{name}",
            )

    references = [
        name
        for name, definition in selector_inputs.items()
        if definition.is_dimensionality_reference
    ]
    if len(references) > 1:
        raise DynamicBlockError(
            f"declares several dimensionality references {references}; at most one "
            "input can be the reference",
            block_type=block_type,
        )

    offsets = {
        name: definition.dimensionality_offset
        for name, definition in selector_inputs.items()
    }
    if any(offsets.values()):
        groups = _groups_from_offsets(offsets, references=references, manifest=manifest)
        return groups

    if manifest.output_dimensionality_offset != -1:
        return frozenset()

    groups = frozenset(
        name
        for name, definition in selector_inputs.items()
        if _may_receive_batches(name, definition, manifest=manifest)
    )
    if not groups:
        raise DynamicBlockError(
            "declares output_dimensionality_offset -1, but no input can receive "
            "batch data to collapse",
            block_type=block_type,
        )

    return groups


def _groups_from_offsets(
    offsets: Mapping[str, int],
    *,
    references: List[str],
    manifest: ManifestDescription,
) -> FrozenSet[str]:
    block_type = manifest.block_type
    nonzero = {name: offset for name, offset in offsets.items() if offset}
    if not references:
        raise DynamicBlockError(
            f"declares input dimensionality offsets {nonzero}; mark one input with "
            "is_dimensionality_reference",
            block_type=block_type,
        )
    if manifest.output_dimensionality_offset != 0:
        raise DynamicBlockError(
            f"declares input dimensionality offsets {nonzero} and output offset "
            f"{manifest.output_dimensionality_offset}; with differing inputs the "
            "output offset must be 0",
            block_type=block_type,
        )
    if 0 not in offsets.values():
        raise DynamicBlockError(
            f"declares input dimensionality offsets {nonzero} without an input at "
            "offset 0; shift the offsets so one input is at 0",
            block_type=block_type,
        )

    reference_offset = offsets[references[0]]
    groups = set()
    for name, offset in offsets.items():
        relative = offset - reference_offset
        if relative > 1:
            raise DynamicBlockError(
                f"input {name!r} is {relative} levels below the reference "
                f"{references[0]!r}; only one level of grouping is supported",
                block_type=block_type,
                location=f"inputs.{name}",
            )
        if relative == 1:
            groups.add(name)

    frozen_groups = frozenset(groups)

    return frozen_groups


def _accepts_batches(manifest: ManifestDescription) -> bool:
    accepts = bool(
        manifest.batch_oriented_parameters
        or manifest.parameters_with_scalars_and_batches
        or manifest.accepts_batch_input
    )

    return accepts


def _batch_points(
    name: str, definition: DynamicInputDefinition, *, manifest: ManifestDescription
) -> Set[bool]:
    """V1's ``points_to_batch`` of an input: whether its selectors deliver batches.

    Image and step-output selectors always do, input-parameter selectors never
    do, and generic selectors follow the manifest's batch flags.
    """
    enforced = name in manifest.get_parameters_enforcing_auto_batch_casting
    points = set()
    for selector_type in definition.selector_types:
        if selector_type is SelectorType.INPUT_PARAMETER:
            points.add(False)
        elif selector_type is not SelectorType.GENERIC:
            points.add(True)
        elif name in manifest.parameters_with_scalars_and_batches:
            points.update({True} if enforced else {True, False})
        else:
            points.add(name in manifest.batch_oriented_parameters or enforced)

    return points


def _batch_mode(
    name: str, definition: DynamicInputDefinition, *, manifest: ManifestDescription
) -> str:
    if not _accepts_batches(manifest):
        return "never"

    points = _batch_points(name, definition, manifest=manifest)
    if points == {True}:
        return "always"
    if points == {False}:
        return "never"

    return "if_varying"


def _may_receive_batches(
    name: str, definition: DynamicInputDefinition, *, manifest: ManifestDescription
) -> bool:
    if _accepts_batches(manifest):
        receives = True in _batch_points(name, definition, manifest=manifest)
        return receives

    receives = any(
        selector_type is not SelectorType.INPUT_PARAMETER
        for selector_type in definition.selector_types
    )

    return receives


def _params_field(
    name: str,
    definition: DynamicInputDefinition,
    *,
    manifest: ManifestDescription,
    group: bool,
    kinds: Mapping[str, Kind],
) -> Tuple[Any, Any]:
    block_type = manifest.block_type
    location = f"inputs.{name}"
    members: List[Any] = []
    if definition.selector_types:
        selector_kinds = _selector_kinds(
            definition, kinds=kinds, block_type=block_type, location=location
        )
        marker = Group if group else Ref
        members.append(
            marker(
                *selector_kinds,
                batch=_batch_mode(name, definition, manifest=manifest),
            )
        )
    members.extend(PYTHON_TYPES[value_type] for value_type in definition.value_types)
    declared_kind_names = {
        kind_name
        for kind_names in definition.selector_data_kind.values()
        for kind_name in kind_names
    }
    if declared_kind_names & STRING_LITERAL_KINDS and str not in members:
        members.append(str)
    if not members:
        raise DynamicBlockError(
            "declares neither selector_types nor value_types",
            block_type=block_type,
            location=location,
        )

    annotation = Union[tuple(members)] if len(members) > 1 else members[0]
    if definition.is_optional:
        annotation = Optional[annotation]

    field_info = _field_info(definition)

    return annotation, field_info


def _field_info(definition: DynamicInputDefinition) -> Any:
    if not definition.has_default_value:
        return Field()

    default = definition.default_value
    if isinstance(default, (list, dict, set)):
        field_info = Field(default_factory=lambda: copy.deepcopy(default))
        return field_info

    field_info = Field(default=default)

    return field_info


def _selector_kinds(
    definition: DynamicInputDefinition,
    *,
    kinds: Mapping[str, Kind],
    block_type: str,
    location: str,
) -> Tuple[Kind, ...]:
    kind_names: List[str] = []
    for selector_type in definition.selector_types:
        if selector_type in IMAGE_SELECTOR_TYPES:
            kind_names.append(IMAGE_KIND_NAME)
            continue
        kind_names.extend(definition.selector_data_kind.get(selector_type) or ["*"])

    resolved = _resolve_kinds(
        kind_names, kinds=kinds, block_type=block_type, location=location
    )

    return resolved


def _resolve_kinds(
    kind_names: Iterable[str],
    *,
    kinds: Mapping[str, Kind],
    block_type: str,
    location: str,
) -> Tuple[Kind, ...]:
    """Look up kinds by name; ``()`` means the wildcard."""
    unique_names = list(dict.fromkeys(kind_names))
    if not unique_names or "*" in unique_names:
        return ()

    resolved = []
    for kind_name in unique_names:
        kind = kinds.get(kind_name)
        if kind is None:
            raise DynamicBlockError(
                f"uses unknown kind {kind_name!r}; known kinds: {sorted(kinds)}",
                block_type=block_type,
                location=location,
            )
        resolved.append(kind)

    return tuple(resolved)


def _metadata(
    definition: DynamicBlockDefinition,
    *,
    declared_imports: Tuple[str, ...],
    allow_local_code: bool,
) -> Dict[str, Any]:
    manifest = definition.manifest
    metadata = {
        "ui_manifest": manifest.ui_manifest,
        "dynamic": {
            "source": BLOCK_SOURCE,
            "execution": "local_python",
            "sandboxed": False,
            "local_code_allowed": allow_local_code,
            "declared_imports": list(declared_imports),
            "run_function_name": definition.code.run_function_name,
            "has_init": definition.code.init_function_code is not None,
            "tensor_compatibility": manifest.tensor_compatibility.value,
            "shared_state": "session mapping unless the caller supplies one",
            "manifest": manifest.model_dump(mode="json"),
        },
    }

    return metadata


# Submitted code: static checks at build time, loading at construction -----


def _code_parts(code: PythonCode) -> List[Tuple[str, str]]:
    """Source parts in V1's module order: imports, run code, init code."""
    parts = [
        ("imports", "\n".join(code.imports)),
        ("run_function_code", code.run_function_code),
    ]
    if code.init_function_code is not None:
        parts.append(("init_function_code", code.init_function_code))

    return parts


def _inspect_code(
    code: PythonCode, *, block_type: str, inputs: List[str]
) -> Tuple[str, ...]:
    """Check syntax and signatures without executing the code.

    Args:
        code: Submitted code.
        block_type: Block type, for errors.
        inputs: Manifest input names ``run`` must accept.

    Returns:
        Module names named by import statements, in first-use order.

    Raises:
        DynamicBlockError: On a syntax error, a missing function or a
            signature that cannot receive the manifest inputs.
    """
    trees = []
    for part, source in _code_parts(code):
        try:
            trees.append(ast.parse(source, filename=f"code.{part}"))
        except SyntaxError as error:
            raise DynamicBlockError(
                f"has a syntax error: {error.msg}: {(error.text or '').strip()!r}",
                block_type=block_type,
                location=f"code.{part} line {error.lineno}",
            ) from error

    bindings = _module_bindings(trees)
    run_function = bindings.get(code.run_function_name)
    if run_function is None:
        raise DynamicBlockError(
            f"defines no top-level function {code.run_function_name!r}",
            block_type=block_type,
            location="code.run_function_code",
        )
    if isinstance(run_function, ast.AsyncFunctionDef):
        raise DynamicBlockError(
            f"{code.run_function_name}() must be a plain function, not async",
            block_type=block_type,
            location=f"code.run_function_code line {run_function.lineno}",
        )
    if isinstance(run_function, ast.FunctionDef):
        _check_run_signature(run_function, inputs=inputs, block_type=block_type)

    init_function = bindings.get(code.init_function_name)
    if code.init_function_code is not None and init_function is None:
        raise DynamicBlockError(
            f"defines no top-level function {code.init_function_name!r}",
            block_type=block_type,
            location="code.init_function_code",
        )
    if isinstance(init_function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        _check_init_signature(init_function, block_type=block_type)

    imports = tuple(dict.fromkeys(_imported_modules(trees)))

    return imports


def _module_bindings(trees: List[ast.Module]) -> Dict[str, ast.AST]:
    """Module-level names and the statement binding them; later ones win."""
    bindings: Dict[str, ast.AST] = {}
    pending: List[ast.AST] = [node for tree in trees for node in tree.body]
    while pending:
        node = pending.pop(0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings[node.name] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node
        elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            # Conditional definitions still bind module names.
            nested = [
                child
                for child in ast.iter_child_nodes(node)
                if isinstance(child, ast.stmt)
            ]
            nested.extend(
                statement
                for handler in getattr(node, "handlers", [])
                for statement in handler.body
            )
            pending[:0] = nested

    return bindings


def _check_run_signature(
    function: ast.FunctionDef, *, inputs: List[str], block_type: str
) -> None:
    location = f"code.run_function_code line {function.lineno}"
    arguments = function.args
    positional = arguments.posonlyargs + arguments.args
    if not positional and arguments.vararg is None:
        raise DynamicBlockError(
            f"{function.name}() must take the block instance first: "
            f"def {function.name}(self, ...)",
            block_type=block_type,
            location=location,
        )
    positional_only = [argument.arg for argument in arguments.posonlyargs[1:]]
    if positional_only:
        raise DynamicBlockError(
            f"{function.name}() parameters {positional_only} are positional-only; "
            "inputs are passed by keyword",
            block_type=block_type,
            location=location,
        )

    defaults = [None] * (len(positional) - len(arguments.defaults)) + list(
        arguments.defaults
    )
    parameters = [
        (argument.arg, default is not None)
        for argument, default in zip(positional, defaults)
    ][1:]
    parameters.extend(
        (argument.arg, default is not None)
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
    )
    accepted = {name for name, _ in parameters}

    missing = [name for name in inputs if name not in accepted]
    if missing and arguments.kwarg is None:
        raise DynamicBlockError(
            f"{function.name}() does not accept input(s) {missing}; add them as "
            "parameters or accept **kwargs",
            block_type=block_type,
            location=location,
        )
    unknown_required = [
        name
        for name, has_default in parameters
        if not has_default and name not in inputs
    ]
    if unknown_required:
        raise DynamicBlockError(
            f"{function.name}() requires {unknown_required}, which are not manifest "
            "inputs",
            block_type=block_type,
            location=location,
        )


def _check_init_signature(function: ast.AST, *, block_type: str) -> None:
    arguments = function.args
    positional = arguments.posonlyargs + arguments.args
    required = positional[: len(positional) - len(arguments.defaults)]
    required.extend(
        argument
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults)
        if default is None
    )
    if required or isinstance(function, ast.AsyncFunctionDef):
        raise DynamicBlockError(
            f"{function.name}() is called once without arguments and must be a "
            "plain function with no required parameters",
            block_type=block_type,
            location=f"line {function.lineno}",
        )


def _imported_modules(trees: List[ast.Module]) -> List[str]:
    modules = []
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.append(node.module)

    return modules


def _checked_shared_state(
    shared_state: Optional[MutableMapping[str, Any]], *, block_type: str
) -> MutableMapping[str, Any]:
    if shared_state is None:
        return {}
    if not isinstance(shared_state, MutableMapping):
        raise TypeError(
            f"Dynamic block {block_type!r}: shared_state must be a mutable mapping, "
            f"got {type(shared_state).__name__}"
        )

    return shared_state


def _checked_policy(policy: Any, *, block_type: str) -> Any:
    missing = [
        name for name in _POLICY_METHODS if not callable(getattr(policy, name, None))
    ]
    if missing:
        raise TypeError(
            f"Dynamic block {block_type!r}: representation_policy "
            f"{type(policy).__name__} lacks {missing}; subclass RepresentationPolicy"
        )

    return policy


def _load_code(
    code: PythonCode, *, block_type: str, shared_state: MutableMapping[str, Any]
) -> types.ModuleType:
    """Execute the submitted code into a new private module.

    Each part is compiled under its own file name, registered with
    ``linecache``, so tracebacks show the submitted line numbers and source.
    """
    module = types.ModuleType(f"{BLOCK_SOURCE}.{block_type}")
    module.__dict__.update(PRELOADED_GLOBALS)
    module.__dict__["globals"] = shared_state
    for part, source in _code_parts(code):
        filename = f"<dynamic block {block_type}: {part}>"
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )
        try:
            exec(compile(source, filename, "exec"), module.__dict__)
        except Exception as error:
            hint = _missing_import_hint(error) if isinstance(error, NameError) else None
            suffix = f". {hint}" if hint else ""
            raise DynamicCodeError(
                f"Dynamic block {block_type!r}: loading code.{part} failed: "
                f"{type(error).__name__}: {error}{suffix}"
            ) from error

    return module


def _call_init(init_function: Any, *, block_type: str) -> Any:
    try:
        init_results = init_function()
    except Exception as error:
        raise DynamicCodeError(
            f"Dynamic block {block_type!r}: init() failed: "
            f"{type(error).__name__}: {error}"
        ) from error

    return init_results


def _missing_import_hint(error: NameError) -> Optional[str]:
    import_line = V1_IMPLICIT_IMPORTS.get(getattr(error, "name", None) or "")
    if import_line is None:
        return None

    hint = (
        f"V2 dynamic blocks do not pre-import {error.name!r} as V1 did; add "
        f"{import_line!r} to code.imports"
    )

    return hint


def _custom_python_problem(declaration: str) -> DiscoveryProblem:
    """Custom-Python unknown; introspection adds the step's ``node_id``."""
    problem = DiscoveryProblem(
        code=DiscoveryProblemCode.CUSTOM_PYTHON_INTERNALS_UNKNOWN,
        description=(
            f"The step runs custom Python code, so its internals may add "
            f"{declaration} beyond the declared ones."
        ),
        details={"declaration": declaration},
    )

    return problem
