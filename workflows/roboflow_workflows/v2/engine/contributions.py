"""Host-supplied implementations of existing blocks.

A host can add implementations to a block it did not write, without changing
the block class, its catalogue or any workflow definition. It describes them in
an ``ImplementationContribution`` and asks the catalogue for a new one::

    contribution = ImplementationContribution(
        block_type="demo/classify@v1",
        contract="v1:sha256:...",            # contract_digest(Classify), pinned
        implementations=(ClassifyWithLut,),
        provider="example-provider",
        version="1.0.0",
    )
    catalogue = public_catalogue.with_contributions([contribution])

The block keeps its one contract: ``Params``, outputs, ``mutates``, batching,
events and kinds. Contributed classes are ordinary ``Implementation``
subclasses, validated exactly like the block's own, and may not restate any of
that contract. The new catalogue's description of the block lists them in
front of the block's own implementations, in the order the host applied them::

    effective = [first contribution] + [second contribution] + [block's own]

Compilation selects from that list as from any other (``targets``): the first
implementation whose ``requires`` the target has wins, quality labels narrow
it, and the plan pins the choice. A contribution may serve only quality labels
the block's own implementations serve, so a definition asking for a label
compiles with or without it. Nothing is imported, discovered or probed; only
the selected implementation's resources resolve, in the block's namespace.
Each implementation declares its own constructor resources; once a block lists
several, its description shows them per implementation, not for the block.
Each contributed ``ImplementationSpec`` records its ``ImplementationOrigin``,
so ``plan.describe()`` shows who supplied a step's code, and a plan comparison
treats a new provider or version as a changed implementation.

A contribution names the block's canonical type and its ``contract_digest``.
The digest covers what an implementation relies on: the parameter schema with
its defaults, constraints, unions and references, field selectors, outputs,
effects, retained inputs and outputs, and events. Documentation (titles, descriptions, examples), the names
of the ``Params`` and nested model classes, aliases, metadata and the module
path do not count. The digest checks declared compatibility; it does not
compare code or behaviour and is not a security boundary. It carries a version
prefix (``v1:``); a digest of another version is rejected as unsupported,
never compared.

Obligations of a contributed implementation, which the engine cannot check:

- **Exactness.** It runs by default wherever its ``requires`` fit, so it must
  meet the block's recorded parity criterion: identical output for
  deterministic work (pixel identity for drawing), or a declared numerical
  tolerance where the block's floating-point contract allows one. It never
  lowers fidelity silently. Approximate implementations are not supported yet.
- **Readiness and lifetime.** The engine waits only for
  ``concurrent.futures.Future`` results and orders no CUDA stream or event.
  A returned value, or a returned future's result, must be safe for any
  consumer on any thread when the engine receives it: complete, or carrying an
  explicit readiness that the public data types and their consumers honour.
  Prefer stream- or event-scoped completion to device-wide synchronisation.
  Keep each allocation alive until all work queued on it has finished.
- **Effects.** Write only inputs the block declares in ``mutates``. Keep an
  input or a returned value after the call only when the block declares it in
  ``retains_inputs`` or ``retains_outputs``. Take inputs in the form the
  contract delivers them; selecting this implementation is no reason to move
  them to another device.
- **Unsupported calls.** Decide before any side effect whether a call is
  supported; delegate unsupported calls to the public reference and count
  them. Never rerun a call as a fallback after it may have written data.
"""

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple, Union

from roboflow_workflows.v2.engine.declaration import BlockSpec, spec_of
from roboflow_workflows.v2.engine.errors import CatalogueError
from roboflow_workflows.v2.engine.implementations import (
    ImplementationOrigin,
    ImplementationSpec,
    read_implementation_specs,
)
from roboflow_workflows.v2.engine.resources import MANAGED_STATE_RESOURCE

__all__ = [
    "CONTRACT_DIGEST_VERSION",
    "ImplementationContribution",
    "apply_contribution",
    "contract_digest",
]

CONTRACT_DIGEST_VERSION = 1
"""Version of the contract canonicalization ``contract_digest`` computes."""

_DIGEST = re.compile(r"v([0-9]+):sha256:([0-9a-f]{64})")
_PROVIDER = re.compile(r"[A-Za-z0-9_\-.]+")
_VERSION = re.compile(r"[A-Za-z0-9_\-.+!]+")

_DEFINITION_PREFIX = "#/$defs/"

# JSON Schema annotation keywords: documentation only.
_SCHEMA_ANNOTATIONS = frozenset({"title", "description", "examples", "$comment"})
# Keywords whose value maps user-chosen names to subschemas. The names stay.
_SCHEMA_MAPS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)
# Keywords whose value is one subschema, or a list of subschemas. Pydantic
# describes a ``Json[Model]`` field's decoded content in ``contentSchema``.
_SCHEMA_NODES = frozenset(
    {
        "contentSchema",
        "items",
        "additionalItems",
        "additionalProperties",
        "unevaluatedItems",
        "unevaluatedProperties",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems",
    }
)


@dataclass(frozen=True)
class ImplementationContribution:
    """Implementations a host adds to one existing block.

    Args:
        block_type: Canonical type of the block, not an alias.
        contract: ``contract_digest`` of the block the implementations were
            built and verified against.
        implementations: ``Implementation`` subclasses in preference order.
        provider: Name of whoever supplies them; letters, digits, ``_``,
            ``-`` and ``.``.
        version: The provider's version; letters, digits and ``_-.+!``.

    Raises:
        CatalogueError: When a value is malformed. Whether the contribution
            fits a block is checked when a catalogue applies it.
    """

    block_type: str
    contract: str
    implementations: Tuple[type, ...]
    provider: str
    version: str

    def __post_init__(self) -> None:
        if not isinstance(self.block_type, str) or not self.block_type:
            raise CatalogueError(
                f"Contribution block_type must be a block type, got {self.block_type!r}"
            )
        if not isinstance(self.contract, str) or not _DIGEST.fullmatch(self.contract):
            raise CatalogueError(
                "Contribution contract must be a contract_digest() value such as "
                f"'v{CONTRACT_DIGEST_VERSION}:sha256:<64 hex digits>', got "
                f"{self.contract!r}"
            )
        if not isinstance(self.provider, str) or not _PROVIDER.fullmatch(self.provider):
            raise CatalogueError(
                "Contribution provider must be letters, digits, _, - or ., got "
                f"{self.provider!r}"
            )
        if not isinstance(self.version, str) or not _VERSION.fullmatch(self.version):
            raise CatalogueError(
                "Contribution version must be letters, digits or _-.+!, got "
                f"{self.version!r}"
            )

        implementations = self.implementations
        if isinstance(implementations, (str, type)) or not isinstance(
            implementations, (tuple, list)
        ):
            raise CatalogueError(
                "Contribution implementations must be a tuple of Implementation "
                f"classes, got {implementations!r}"
            )
        if not implementations:
            raise CatalogueError("Contribution lists no implementations")

        object.__setattr__(self, "implementations", tuple(implementations))

    @property
    def origin(self) -> ImplementationOrigin:
        """The origin recorded on every implementation of this contribution."""
        origin = ImplementationOrigin(
            provider=self.provider, version=self.version, contract=self.contract
        )

        return origin


def contract_digest(block: Union[type, BlockSpec]) -> str:
    """Return the versioned digest of a block's contract.

    Args:
        block: A concrete ``Block`` subclass or its ``BlockSpec``.

    Returns:
        ``"v1:sha256:<hex>"``. Equal for blocks with the same type, parameter
        schema and selectors, outputs, effects, retained inputs and outputs,
        and events, whatever their documentation, aliases, metadata, module
        or implementations. Empty retention declarations leave the digest a
        block had before they existed.

    Raises:
        DeclarationError: When ``block`` is not a concrete block class.
    """
    spec = block if isinstance(block, BlockSpec) else spec_of(block)
    canonical = json.dumps(_contract_of(spec), sort_keys=True, separators=(",", ":"))
    hexdigest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    digest = f"v{CONTRACT_DIGEST_VERSION}:sha256:{hexdigest}"

    return digest


def apply_contribution(
    spec: BlockSpec, contribution: ImplementationContribution
) -> BlockSpec:
    """Return the description of a block with one contribution added.

    ``Catalogue.with_contributions`` calls this; ``spec`` is not changed.

    Args:
        spec: The block's current description, possibly with earlier
            contributions.
        contribution: Implementations to add.

    Returns:
        A copy of ``spec`` listing earlier contributions, then this one's
        implementations with their origin, then the block's own.

    Raises:
        CatalogueError: When the contribution names an alias or a different
            block, carries another contract or digest version, or an
            implementation is invalid for the block, requests managed state,
            repeats a listed name or serves a quality label the block's own
            implementations do not.
    """

    def fail(message: str) -> CatalogueError:
        return CatalogueError(
            f"Contribution from {contribution.provider} {contribution.version} to "
            f"{spec.type}: {message}"
        )

    if contribution.block_type != spec.type:
        raise fail(
            f"names {contribution.block_type!r}; a contribution must name the "
            f"canonical type {spec.type!r}"
        )
    _check_contract(spec, contract=contribution.contract, fail=fail)

    declared = read_implementation_specs(
        contribution.implementations, fields=spec.fields, fail=fail
    )
    added = tuple(
        dataclasses.replace(implementation, origin=contribution.origin)
        for implementation in declared
    )
    _check_added(added, listed=spec.implementations, fail=fail)

    contributed = tuple(
        implementation
        for implementation in spec.implementations
        if implementation.origin is not None
    )
    own = tuple(
        implementation
        for implementation in spec.implementations
        if implementation.origin is None
    )
    extended = dataclasses.replace(spec, implementations=contributed + added + own)

    return extended


def _check_contract(
    spec: BlockSpec, *, contract: str, fail: Callable[[str], CatalogueError]
) -> None:
    version = int(_DIGEST.fullmatch(contract).group(1))
    if version != CONTRACT_DIGEST_VERSION:
        raise fail(
            f"contract digest version v{version} is not supported; this engine "
            f"computes v{CONTRACT_DIGEST_VERSION}"
        )

    actual = contract_digest(spec)
    if contract != actual:
        raise fail(
            f"was built for contract {contract}, but the block's contract is now "
            f"{actual}; use a provider release built for the current contract, or "
            "leave this contribution out so the block runs its own implementations"
        )


def _check_added(
    added: Tuple[ImplementationSpec, ...],
    *,
    listed: Tuple[ImplementationSpec, ...],
    fail: Callable[[str], CatalogueError],
) -> None:
    stateful = [
        implementation.name
        for implementation in added
        if any(
            resource.name == MANAGED_STATE_RESOURCE
            for resource in implementation.resources
        )
    ]
    if stateful:
        raise fail(
            f"implementation(s) {stateful} request {MANAGED_STATE_RESOURCE!r}, "
            "state other steps and handlers read; a contribution cannot add it"
        )

    taken = {implementation.name for implementation in listed}
    repeated = sorted(
        implementation.name for implementation in added if implementation.name in taken
    )
    if repeated:
        raise fail(f"implementation name(s) {repeated} are already listed")

    # A definition asking for a label must compile without the contribution.
    public_labels = {
        label
        for implementation in listed
        if implementation.origin is None
        for label in implementation.quality
    }
    new_labels = sorted(
        {label for implementation in added for label in implementation.quality}
        - public_labels
    )
    if new_labels:
        raise fail(
            f"quality label(s) {new_labels} are not served by the block's own "
            f"implementations (they serve {sorted(public_labels)}); a contribution "
            "may serve only labels a definition can also get without it"
        )


def _contract_of(spec: BlockSpec) -> Dict[str, Any]:
    """The parts of a block's description an implementation relies on."""
    contract = {
        "digest_version": CONTRACT_DIGEST_VERSION,
        "type": spec.type,
        "params_schema": _canonical_schema(spec.params_schema()),
        "fields": {
            name: _undocumented(field.describe()) for name, field in spec.fields.items()
        },
        "outputs": {
            name: _undocumented(output.describe())
            for name, output in spec.outputs.items()
        },
        "configured_outputs": spec.configured_outputs,
        "output_fields": list(spec.output_fields),
        "is_control": spec.is_control,
        "accepts_batches": spec.accepts_batches,
        "accepts_empty": spec.accepts_empty,
        "mutates": list(spec.mutates),
        "prunable": spec.prunable,
        "events": {
            name: _undocumented(event.describe()) for name, event in spec.events.items()
        },
    }
    # Added after v1 digests were published: present only when declared, so
    # a block that keeps nothing keeps its v1 digest.
    if spec.retains_inputs:
        contract["retains_inputs"] = list(spec.retains_inputs)
    if spec.retains_outputs:
        contract["retains_outputs"] = list(spec.retains_outputs)

    return contract


def _undocumented(description: Dict[str, Any]) -> Dict[str, Any]:
    undocumented = {
        key: value for key, value in description.items() if key != "description"
    }

    return undocumented


def _canonical_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """The parameter schema without documentation or model class names.

    Annotations are dropped (``_without_annotations``). Definitions keep their
    content but are renamed ``d0``, ``d1``, ... in the order the schema first
    refers to them, walking keys and property names sorted, so renaming a
    ``Params`` or nested model class changes nothing while every reference,
    union, default and constraint stays.
    """
    stripped = _without_annotations(schema)
    definitions = stripped.pop("$defs", {})

    # Breadth-first from the root: each definition gets the next alias when
    # it is first referred to, then its own references are visited.
    aliases: Dict[str, str] = {}
    pending = [stripped]
    while pending:
        for reference in _references(pending.pop(0)):
            name = _definition_name(reference)
            if name in definitions and name not in aliases:
                aliases[name] = f"d{len(aliases)}"
                pending.append(definitions[name])

    canonical = _with_references(stripped, aliases=aliases)
    if aliases:
        canonical["$defs"] = {
            alias: _with_references(definitions[name], aliases=aliases)
            for name, alias in aliases.items()
        }

    return canonical


def _definition_name(reference: str) -> Optional[str]:
    if not reference.startswith(_DEFINITION_PREFIX):
        return None

    name = reference[len(_DEFINITION_PREFIX) :]

    return name


def _renamed_reference(reference: Any, *, aliases: Mapping[str, str]) -> Any:
    name = _definition_name(reference) if isinstance(reference, str) else None
    if name not in aliases:
        return reference

    renamed = f"{_DEFINITION_PREFIX}{aliases[name]}"

    return renamed


def _references(node: Any) -> Iterator[str]:
    """Yield the ``$ref`` targets of a schema node in a name-independent order."""
    if isinstance(node, list):
        for item in node:
            yield from _references(item)
        return
    if not isinstance(node, dict):
        return

    for key in sorted(node):
        value = node[key]
        if key == "$ref" and isinstance(value, str):
            yield value
        elif key == "discriminator" and isinstance(value, dict):
            mapping = value.get("mapping", {})
            for tag in sorted(mapping):
                yield mapping[tag]
        elif key in _SCHEMA_MAPS and isinstance(value, dict):
            for name in sorted(value):
                yield from _references(value[name])
        elif key in _SCHEMA_NODES:
            yield from _references(value)


def _with_references(node: Any, *, aliases: Mapping[str, str]) -> Any:
    """Copy a schema node with its definition references renamed by ``aliases``."""
    if isinstance(node, list):
        return [_with_references(item, aliases=aliases) for item in node]
    if not isinstance(node, dict):
        return node

    renamed = {}
    for key, value in node.items():
        if key == "$ref":
            value = _renamed_reference(value, aliases=aliases)
        elif key == "discriminator" and isinstance(value, dict):
            mapping = value.get("mapping")
            if isinstance(mapping, dict):
                mapping = {
                    tag: _renamed_reference(reference, aliases=aliases)
                    for tag, reference in mapping.items()
                }
                value = {**value, "mapping": mapping}
        elif key in _SCHEMA_MAPS and isinstance(value, dict):
            value = {
                name: _with_references(item, aliases=aliases)
                for name, item in value.items()
            }
        elif key in _SCHEMA_NODES:
            value = _with_references(value, aliases=aliases)
        renamed[key] = value

    return renamed


def _without_annotations(node: Any) -> Any:
    """Drop annotation keywords from every schema node; keep names and data.

    Only schema nodes are visited: property names, defaults, enums, constants
    and selector metadata are data and stay exactly as declared, even when
    they contain a key such as ``"description"``.
    """
    if isinstance(node, list):
        return [_without_annotations(item) for item in node]
    if not isinstance(node, dict):
        return node

    canonical = {}
    for key, value in node.items():
        if key in _SCHEMA_ANNOTATIONS:
            continue
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            value = {name: _without_annotations(item) for name, item in value.items()}
        elif key in _SCHEMA_NODES:
            value = _without_annotations(value)
        canonical[key] = value

    return canonical
