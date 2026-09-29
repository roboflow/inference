"""Block contracts and the explicit registry of the V2 execution engine.

A block declares, per input, the payload kind and the view it consumes
(``item`` or the one trailing grouping axis as a ``batch``). Per output it
declares the kind and the dimensional transformation relative to a source
input: ``preserve`` keeps the source grouping, ``append`` adds one trailing
nesting axis, ``collapse`` removes the trailing axis.

Supported view/transform combinations in M1:

| Source view | Transform  | Block returns per call                     |
| ----------- | ---------- | ------------------------------------------ |
| item        | preserve   | one payload                                |
| item        | append     | a local-index ``Batch`` of children        |
| batch       | preserve   | a ``Batch`` with the supplied full indices |
| batch       | collapse   | one payload                                |

The registry maps explicit kind names to optional payload validators and
explicit block names to ``BlockRegistration`` records. Nothing is discovered
implicitly; callers register what they select. A registry ``snapshot()`` is
detached and frozen so a compiled plan cannot be changed by later edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from roboflow_workflows.execution_engine.v2.data import (
    AXIS_KIND_DYNAMIC_NESTING,
    AXIS_KIND_STATIC_NESTING,
    Axis,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError

VIEW_ITEM = "item"
VIEW_BATCH = "batch"
INPUT_VIEWS: Tuple[str, ...] = (VIEW_ITEM, VIEW_BATCH)

TRANSFORM_PRESERVE = "preserve"
TRANSFORM_APPEND = "append"
TRANSFORM_COLLAPSE = "collapse"
OUTPUT_TRANSFORMS: Tuple[str, ...] = (
    TRANSFORM_PRESERVE,
    TRANSFORM_APPEND,
    TRANSFORM_COLLAPSE,
)

CONTEXT_POLICY_COMMON_OR_NONE = "common_or_none"
CONTEXT_POLICIES: Tuple[str, ...] = (CONTEXT_POLICY_COMMON_OR_NONE,)

SUPPORTED_VIEW_TRANSFORMS: Tuple[Tuple[str, str], ...] = (
    (VIEW_ITEM, TRANSFORM_PRESERVE),
    (VIEW_ITEM, TRANSFORM_APPEND),
    (VIEW_BATCH, TRANSFORM_PRESERVE),
    (VIEW_BATCH, TRANSFORM_COLLAPSE),
)

KindValidator = Callable[[Any], bool]
BlockFactory = Callable[[Mapping[str, Any]], Any]

_EMPTY_MAPPING: Mapping[Any, Any] = MappingProxyType({})


def _validate_name(value: Any, *, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{what} must be a non-empty string, got {value!r}")

    return value


def _validate_port_name(value: Any, *, what: str) -> str:
    name = _validate_name(value, what=what)
    if not name.isidentifier():
        raise ContractError(
            f"{what} must be a valid Python identifier so it can be passed as a "
            f"named argument to run(), got {name!r}"
        )

    return name


@dataclass(frozen=True)
class InputSpec:
    """Declaration of one block input.

    Args:
        kind: Registered payload kind name.
        view: ``item`` to receive one payload per call, ``batch`` to receive
            the trailing grouping axis as a ``Batch``.

    Raises:
        ContractError: On an empty kind or unknown view.
    """

    kind: str
    view: str = VIEW_ITEM

    def __post_init__(self) -> None:
        _validate_name(self.kind, what="InputSpec kind")
        if self.view not in INPUT_VIEWS:
            raise ContractError(
                f"InputSpec view must be one of {list(INPUT_VIEWS)}, "
                f"got {self.view!r}"
            )


@dataclass(frozen=True)
class OutputSpec:
    """Declaration of one block output.

    Args:
        kind: Registered payload kind name.
        transform: ``preserve``, ``append`` or ``collapse`` relative to the
            source input's grouping.
        source: Name of the input whose grouping the output derives from;
            ``None`` means the contract's reference input.
        axis: Axis key for ``append``. Its identity is derived from the
            producer and this key, so equal keys within one producer promise
            corresponding children.
        stationary: For ``append``, whether the new axis is static nesting
            with stable child identities.
        context_policy: How collapse resolves child contexts; only
            ``common_or_none`` exists in M1.

    Raises:
        ContractError: On unknown transform/policy, an ``append`` without an
            axis key, or ``axis``/``stationary`` used with other transforms.
    """

    kind: str
    transform: str = TRANSFORM_PRESERVE
    source: Optional[str] = None
    axis: Optional[str] = None
    stationary: bool = False
    context_policy: str = CONTEXT_POLICY_COMMON_OR_NONE

    def __post_init__(self) -> None:
        _validate_name(self.kind, what="OutputSpec kind")
        if self.transform not in OUTPUT_TRANSFORMS:
            raise ContractError(
                f"OutputSpec transform must be one of {list(OUTPUT_TRANSFORMS)}, "
                f"got {self.transform!r}"
            )
        if self.source is not None:
            _validate_port_name(self.source, what="OutputSpec source")
        if not isinstance(self.stationary, bool):
            raise ContractError(
                f"OutputSpec stationary must be a bool, got {self.stationary!r}"
            )
        if self.context_policy not in CONTEXT_POLICIES:
            raise ContractError(
                "OutputSpec context_policy must be one of "
                f"{list(CONTEXT_POLICIES)}, got {self.context_policy!r}"
            )
        if self.transform == TRANSFORM_APPEND:
            if self.axis is None:
                raise ContractError(
                    "OutputSpec with transform 'append' requires an axis key"
                )
            _validate_name(self.axis, what="OutputSpec axis")
        else:
            if self.axis is not None:
                raise ContractError(
                    f"OutputSpec axis {self.axis!r} is only allowed with "
                    f"transform 'append', got {self.transform!r}"
                )
            if self.stationary:
                raise ContractError(
                    "OutputSpec stationary=True is only allowed with transform "
                    f"'append', got {self.transform!r}"
                )

    @property
    def appended_axis_kind(self) -> str:
        """Axis kind of the appended axis (``append`` outputs only).

        Raises:
            ContractError: When the transform is not ``append``.
        """
        if self.transform != TRANSFORM_APPEND:
            raise ContractError(
                f"Output with transform {self.transform!r} appends no axis"
            )
        axis_kind = (
            AXIS_KIND_STATIC_NESTING if self.stationary else AXIS_KIND_DYNAMIC_NESTING
        )

        return axis_kind

    def appended_axis(self, *, producer: str) -> Axis:
        """Build the axis an ``append`` output introduces for ``producer``.

        Args:
            producer: Identity of the producing step; combined with the axis
                key so separate producers never share an axis identity.

        Returns:
            Axis with id ``"<producer>/<axis key>"`` and the declared kind.
            The engine derives the same spelling when it compiles a step.

        Raises:
            ContractError: When the transform is not ``append``.
        """
        _validate_name(producer, what="Producer identity")
        axis_kind = self.appended_axis_kind
        axis = Axis(
            id=f"{producer}/{self.axis}",
            kind=axis_kind,
            stationary=self.stationary,
        )

        return axis


def _snapshot_specs(
    value: Any,
    *,
    what: str,
    spec_type: type,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{what} must be a mapping, got {type(value).__name__}")
    if not value:
        raise ContractError(f"{what} must declare at least one port")

    snapshot: Dict[str, Any] = {}
    for name, spec in value.items():
        port_name = _validate_port_name(name, what=f"{what} port name")
        if not isinstance(spec, spec_type):
            raise ContractError(
                f"{what}['{port_name}'] must be an instance of "
                f"{spec_type.__name__}, got {type(spec).__name__}"
            )
        snapshot[port_name] = spec

    frozen_snapshot = MappingProxyType(snapshot)

    return frozen_snapshot


@dataclass(frozen=True)
class BlockContract:
    """Declared inputs, outputs and reference input of one block.

    Args:
        reference: Input whose grouping defines the invocation prefix and the
            default source of every output.
        inputs: Input name to ``InputSpec``; snapshotted read-only.
        outputs: Output name to ``OutputSpec``; snapshotted read-only.
        mutates_inputs: Inputs the block mutates in place. Declared for
            later policy; M1 rejects such blocks at compilation.

    Raises:
        ContractError: On unknown reference/source names, unsupported
            view/transform combinations, or inconsistent stationarity for one
            axis key.
    """

    reference: str
    inputs: Mapping[str, InputSpec]
    outputs: Mapping[str, OutputSpec]
    mutates_inputs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_port_name(self.reference, what="BlockContract reference")
        inputs = _snapshot_specs(
            self.inputs, what="BlockContract inputs", spec_type=InputSpec
        )
        outputs = _snapshot_specs(
            self.outputs, what="BlockContract outputs", spec_type=OutputSpec
        )
        if self.reference not in inputs:
            raise ContractError(
                f"BlockContract reference '{self.reference}' is not a declared "
                f"input; inputs are {list(inputs)}"
            )

        if isinstance(self.mutates_inputs, str):
            raise ContractError(
                "BlockContract mutates_inputs must be a sequence of input names, "
                f"got {self.mutates_inputs!r}"
            )
        mutates_inputs = tuple(self.mutates_inputs)
        for name in mutates_inputs:
            if name not in inputs:
                raise ContractError(
                    f"BlockContract mutates_inputs names unknown input {name!r}; "
                    f"inputs are {list(inputs)}"
                )

        axis_stationarity: Dict[str, bool] = {}
        for output_name, spec in outputs.items():
            source_name = spec.source if spec.source is not None else self.reference
            if source_name not in inputs:
                raise ContractError(
                    f"Output '{output_name}' names unknown source input "
                    f"'{source_name}'; inputs are {list(inputs)}"
                )
            source_view = inputs[source_name].view
            if (source_view, spec.transform) not in SUPPORTED_VIEW_TRANSFORMS:
                raise ContractError(
                    f"Output '{output_name}' declares transform '{spec.transform}' "
                    f"on source input '{source_name}' with view '{source_view}'; "
                    "supported combinations are "
                    f"{[f'{view}/{transform}' for view, transform in SUPPORTED_VIEW_TRANSFORMS]}"
                )
            if spec.transform == TRANSFORM_APPEND:
                previous = axis_stationarity.get(spec.axis)
                if previous is not None and previous != spec.stationary:
                    raise ContractError(
                        f"Outputs sharing axis key '{spec.axis}' must agree on "
                        "stationarity"
                    )
                axis_stationarity[spec.axis] = spec.stationary

        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "mutates_inputs", mutates_inputs)

    @property
    def input_names(self) -> Tuple[str, ...]:
        """Declared input names in declaration order."""
        return tuple(self.inputs)

    @property
    def output_names(self) -> Tuple[str, ...]:
        """Declared output names in declaration order."""
        return tuple(self.outputs)

    @property
    def reference_input(self) -> InputSpec:
        """Specification of the reference input."""
        return self.inputs[self.reference]

    def output_source(self, output_name: str) -> str:
        """Resolve the source input name of an output.

        Args:
            output_name: Declared output name.

        Returns:
            The output's explicit source, or the reference input when the
            output declares none.

        Raises:
            ContractError: On an unknown output name.
        """
        spec = self._output_spec(output_name)
        source_name = spec.source if spec.source is not None else self.reference

        return source_name

    def output_source_view(self, output_name: str) -> str:
        """Return the view of the source input of an output.

        Args:
            output_name: Declared output name.

        Returns:
            ``item`` or ``batch``.

        Raises:
            ContractError: On an unknown output name.
        """
        source_name = self.output_source(output_name)
        source_view = self.inputs[source_name].view

        return source_view

    def _output_spec(self, output_name: str) -> OutputSpec:
        if output_name not in self.outputs:
            raise ContractError(
                f"Unknown output '{output_name}'; outputs are {list(self.outputs)}"
            )

        return self.outputs[output_name]


@dataclass(frozen=True)
class BlockRegistration:
    """A registered block: its contract and per-run instance factory.

    Args:
        contract: Declared inputs/outputs.
        factory: Callable taking the step's static ``config`` mapping and
            returning an object with a ``run(**named_inputs)`` method. The
            engine calls it once per compiled step and keeps per-run instances
            separate.

    Raises:
        ContractError: When ``contract`` is not a ``BlockContract`` or
            ``factory`` is not callable.
    """

    contract: BlockContract
    factory: BlockFactory = field(compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.contract, BlockContract):
            raise ContractError(
                "BlockRegistration contract must be a BlockContract, "
                f"got {type(self.contract).__name__}"
            )
        if not callable(self.factory):
            raise ContractError(
                "BlockRegistration factory must be callable, "
                f"got {type(self.factory).__name__}"
            )


class Registry:
    """Explicit registry of payload kinds and blocks.

    Kinds must be registered before blocks referencing them. Duplicate
    registrations and unknown names raise ``ContractError``. ``snapshot()``
    returns a detached, frozen copy that later registrations on either side
    cannot affect.
    """

    def __init__(self) -> None:
        self._kinds: Dict[str, Optional[KindValidator]] = {}
        self._blocks: Dict[str, BlockRegistration] = {}
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        """Whether this registry rejects further registrations."""
        return self._frozen

    @property
    def kind_names(self) -> Tuple[str, ...]:
        """Registered kind names in registration order."""
        return tuple(self._kinds)

    @property
    def block_names(self) -> Tuple[str, ...]:
        """Registered block names in registration order."""
        return tuple(self._blocks)

    def has_kind(self, name: str) -> bool:
        """Return whether ``name`` is a registered kind.

        Args:
            name: Kind name.

        Returns:
            ``True`` when registered.
        """
        return name in self._kinds

    def has_block(self, name: str) -> bool:
        """Return whether ``name`` is a registered block.

        Args:
            name: Block name.

        Returns:
            ``True`` when registered.
        """
        return name in self._blocks

    def register_kind(
        self,
        name: str,
        validator: Optional[KindValidator] = None,
    ) -> None:
        """Register a payload kind.

        Args:
            name: Kind name.
            validator: Optional ``validator(payload) -> bool``. ``None``
                accepts any payload.

        Raises:
            ContractError: On a frozen registry, empty name, duplicate name or
                non-callable validator.
        """
        self._ensure_mutable(action=f"register kind {name!r}")
        kind_name = _validate_name(name, what="Kind name")
        if kind_name in self._kinds:
            raise ContractError(f"Kind '{kind_name}' is already registered")
        if validator is not None and not callable(validator):
            raise ContractError(
                f"Kind '{kind_name}' validator must be callable, "
                f"got {type(validator).__name__}"
            )

        self._kinds[kind_name] = validator

    def register_block(
        self,
        name: str,
        *,
        contract: BlockContract,
        factory: BlockFactory,
    ) -> None:
        """Register a block.

        Args:
            name: Block type name used by workflow steps.
            contract: Declared inputs/outputs.
            factory: ``factory(config_mapping)`` returning a block instance.

        Raises:
            ContractError: On a frozen registry, empty/duplicate name, or a
                contract referencing an unregistered kind.
        """
        self._ensure_mutable(action=f"register block {name!r}")
        block_name = _validate_name(name, what="Block name")
        if block_name in self._blocks:
            raise ContractError(f"Block '{block_name}' is already registered")

        registration = BlockRegistration(contract=contract, factory=factory)
        for port_name, spec in contract.inputs.items():
            if spec.kind not in self._kinds:
                raise ContractError(
                    f"Block '{block_name}' input '{port_name}' uses unregistered "
                    f"kind '{spec.kind}'; kinds are {list(self._kinds)}"
                )
        for port_name, spec in contract.outputs.items():
            if spec.kind not in self._kinds:
                raise ContractError(
                    f"Block '{block_name}' output '{port_name}' uses unregistered "
                    f"kind '{spec.kind}'; kinds are {list(self._kinds)}"
                )

        self._blocks[block_name] = registration

    def get_block(self, name: str) -> BlockRegistration:
        """Look up a registered block.

        Args:
            name: Block type name.

        Returns:
            The block's registration.

        Raises:
            ContractError: On an unknown name.
        """
        if name not in self._blocks:
            raise ContractError(
                f"Unknown block '{name}'; registered blocks are {list(self._blocks)}"
            )

        return self._blocks[name]

    def validate(self, kind: str, payload: Any) -> None:
        """Validate a payload against a registered kind.

        Args:
            kind: Kind name.
            payload: Payload to check.

        Raises:
            ContractError: On an unknown kind, a validator returning a
                non-bool, a validator raising, or a rejected payload. A
                raising validator is preserved as the cause.
        """
        if kind not in self._kinds:
            raise ContractError(
                f"Unknown kind '{kind}'; registered kinds are {list(self._kinds)}"
            )
        validator = self._kinds[kind]
        if validator is None:
            return

        try:
            accepted = validator(payload)
        except Exception as error:
            raise ContractError(
                f"Validator of kind '{kind}' failed on payload of type "
                f"{type(payload).__name__}: {error}"
            ) from error
        if not isinstance(accepted, bool):
            raise ContractError(
                f"Validator of kind '{kind}' must return a bool, "
                f"got {type(accepted).__name__}"
            )
        if not accepted:
            raise ContractError(
                f"Payload of type {type(payload).__name__} is not a valid '{kind}'"
            )

    def snapshot(self) -> "Registry":
        """Return a detached, frozen copy of the current registrations.

        Returns:
            A new registry that ignores later edits to this one and rejects
            its own registrations.
        """
        snapshot = Registry()
        snapshot._kinds = dict(self._kinds)
        snapshot._blocks = dict(self._blocks)
        snapshot._frozen = True

        return snapshot

    def _ensure_mutable(self, *, action: str) -> None:
        if self._frozen:
            raise ContractError(f"Cannot {action}: this registry is a frozen snapshot")
