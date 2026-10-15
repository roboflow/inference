"""Explicit, immutable collection of V2 block, source and operator classes and kinds.

A catalogue only collects classes; every contract detail comes from the class
itself (see ``declaration``, ``sources`` and ``operators``)::

    catalogue = Catalogue(
        [Scale, Crop], sources=[CsvTemperature], operators=[Align], namespace="demo"
    )

Blocks, sources and operators are separate registries: a ``steps`` entry names
a block type, a ``sources`` entry a source type and an ``operators`` entry an
operator type; the registries may spell the same identity without conflict.
There is no hidden process-wide operator registry: the built-in operators are
collected by the V2 built-in catalogue like any third-party class.

Catalogues are immutable. ``Catalogue.merge``, ``with_blocks`` and
``with_contributions`` return new catalogues, so a compiled plan can keep the
catalogue it was compiled with.

``with_contributions`` adds host-supplied implementations to existing blocks
(``contributions``). The new catalogue holds an extended description of each
such block; the block class and the original catalogue are unchanged. Merging
keeps one block class once when its implementation lists match (the same names
and origins in the same order, each with the same class, requirements, quality,
phases and resources) and rejects it otherwise, so apply contributions once,
after merging.

Plugins are imported only when the caller asks for them:
``Catalogue.from_modules(["my_package.blocks"])`` imports each module and reads
its ``WORKFLOWS_V2_CATALOGUE`` attribute (a ``Catalogue`` or a zero-argument
callable returning one). The V1 loader, its ``load_blocks()`` convention and the
``WORKFLOWS_PLUGINS`` environment variable are neither read nor changed.

Kinds referenced by registered blocks are collected automatically. Two
different kind objects with one name, two blocks claiming one identity, and a
block whose ``engine_compatibility`` excludes this engine are rejected. The
built-in wildcard is a neutral placeholder: an explicit wildcard policy replaces
it and survives later registration of the placeholder.

Recording codecs (``PayloadCodec``) are registered like kinds. Two different
codecs with one name, two codecs for one payload type, and a codec replacing a
generic built-in codec are rejected.
"""

import importlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Tuple, Union

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from roboflow_workflows.v2.data.kinds import WILDCARD_KIND, Kind
from roboflow_workflows.v2.engine.contributions import (
    ImplementationContribution,
    apply_contribution,
)
from roboflow_workflows.v2.engine.declaration import BlockSpec, spec_of
from roboflow_workflows.v2.engine.errors import CatalogueError, ContractError
from roboflow_workflows.v2.engine.implementations import ImplementationSpec
from roboflow_workflows.v2.engine.recording.codecs import (
    PayloadCodec,
    codec_conflict,
)
from roboflow_workflows.v2.engine.resources import ResourceSpec
from roboflow_workflows.v2.engine.sources import SourceSpec, spec_of_source

if TYPE_CHECKING:
    from roboflow_workflows.v2.engine.operators.contract import OperatorSpec

V2_ENGINE_VERSION = "2.0.0"
CATALOGUE_ATTRIBUTE = "WORKFLOWS_V2_CATALOGUE"


@dataclass(frozen=True)
class CatalogueEntry:
    """A registered block class and the namespace that registered it.

    Args:
        spec: The class's validated declaration.
        namespace: Catalogue namespace; scopes resource providers and keys.
    """

    spec: BlockSpec
    namespace: str

    def describe(self) -> Dict[str, Any]:
        """Return the block description including its namespace."""
        description = {"namespace": self.namespace, **self.spec.describe()}

        return description


@dataclass(frozen=True)
class SourceEntry:
    """A registered source class and the namespace that registered it.

    Args:
        spec: The class's validated declaration.
        namespace: Catalogue namespace; scopes resource providers and keys.
    """

    spec: SourceSpec
    namespace: str

    def describe(self) -> Dict[str, Any]:
        """Return the source description including its namespace."""
        description = {"namespace": self.namespace, **self.spec.describe()}

        return description


@dataclass(frozen=True)
class OperatorEntry:
    """A registered operator class and the namespace that registered it.

    Args:
        spec: The class's validated ``OperatorSpec``.
        namespace: Catalogue namespace of the registration.
    """

    spec: "OperatorSpec"
    namespace: str

    def describe(self) -> Dict[str, Any]:
        """Return the operator description including its namespace."""
        description = {"namespace": self.namespace, **self.spec.describe()}

        return description


class Catalogue:
    """Immutable set of block, source and operator classes, kinds and providers.

    Args:
        blocks: Concrete ``Block`` subclasses.
        sources: Concrete ``Source`` subclasses.
        operators: Concrete ``Operator`` subclasses.
        kinds: Additional kinds, e.g. for workflow inputs no block references.
        namespace: Namespace of these classes, used for resource keys.
        providers: Resource providers for blocks and sources of this
            namespace, by constructor parameter name. Wrap lazily created
            values in ``Factory``.
        codecs: Recording codecs of payload types these classes produce.

    Raises:
        CatalogueError: On a class that is not a concrete class of its
            registry, duplicate identities, conflicting kinds or codecs, or
            an incompatible declaration.
    """

    def __init__(
        self,
        blocks: Iterable[type] = (),
        *,
        sources: Iterable[type] = (),
        operators: Iterable[type] = (),
        kinds: Iterable[Kind] = (),
        namespace: str = "",
        providers: Optional[Mapping[str, Any]] = None,
        codecs: Iterable[PayloadCodec] = (),
    ):
        self._entries: Dict[str, CatalogueEntry] = {}
        self._identities: Dict[str, str] = {}
        self._source_entries: Dict[str, SourceEntry] = {}
        self._source_identities: Dict[str, str] = {}
        self._operator_entries: Dict[str, OperatorEntry] = {}
        self._operator_identities: Dict[str, str] = {}
        self._kinds: Dict[str, Kind] = {}
        self._providers: Dict[str, Dict[str, Any]] = {}
        self._codecs: Dict[str, PayloadCodec] = {}
        self._codec_types: Dict[str, PayloadCodec] = {}

        if not isinstance(namespace, str):
            raise CatalogueError(
                f"Catalogue namespace must be a string, got {namespace!r}"
            )

        self._add_kind(WILDCARD_KIND)
        for kind in kinds:
            self._add_kind(kind)
        for block_class in blocks:
            try:
                spec = spec_of(block_class)
            except ContractError as error:
                raise CatalogueError(
                    f"Cannot register {block_class!r}: {error}"
                ) from error
            self._add_entry(CatalogueEntry(spec=spec, namespace=namespace))
        for source_class in sources:
            try:
                source_spec = spec_of_source(source_class)
            except ContractError as error:
                raise CatalogueError(
                    f"Cannot register {source_class!r}: {error}"
                ) from error
            self._add_source(SourceEntry(spec=source_spec, namespace=namespace))
        for operator_class in operators:
            operator_spec = _spec_of_operator(operator_class)
            self._add_operator(OperatorEntry(spec=operator_spec, namespace=namespace))
        for name, value in (providers or {}).items():
            self._add_provider(namespace, name=name, value=value)
        for codec in codecs:
            self._add_codec(codec)

    @classmethod
    def merge(cls, *catalogues: "Catalogue") -> "Catalogue":
        """Combine catalogues into a new one.

        The same class registered in several inputs under the same namespace
        is kept once.

        Args:
            *catalogues: Catalogues to combine.

        Returns:
            A new catalogue containing every block, source, operator, kind,
            provider and recording codec.

        Raises:
            CatalogueError: On conflicting identities, kinds, providers or
                codecs, or one block class listing different implementations.
        """
        merged = cls()
        for catalogue in catalogues:
            if not isinstance(catalogue, Catalogue):
                raise CatalogueError(
                    f"Can only merge Catalogue objects, got {catalogue!r}"
                )
            merged._add_catalogue(catalogue)

        return merged

    @classmethod
    def from_modules(
        cls,
        module_names: Iterable[str],
        *,
        attribute: str = CATALOGUE_ATTRIBUTE,
    ) -> "Catalogue":
        """Import plugin modules and merge their catalogues.

        Args:
            module_names: Importable module names.
            attribute: Module attribute holding a ``Catalogue`` or a
                zero-argument callable returning one.

        Returns:
            The merged catalogue of all modules.

        Raises:
            CatalogueError: When a module cannot be imported or does not expose
                a catalogue.
        """
        catalogues = []
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError as error:
                raise CatalogueError(
                    f"Cannot import V2 plugin module {module_name!r}: {error}"
                ) from error

            exposed = getattr(module, attribute, None)
            if callable(exposed) and not isinstance(exposed, Catalogue):
                exposed = exposed()
            if not isinstance(exposed, Catalogue):
                raise CatalogueError(
                    f"V2 plugin module {module_name!r} must expose `{attribute}` as a "
                    f"Catalogue or a callable returning one, got {exposed!r}"
                )
            catalogues.append(exposed)

        merged = cls.merge(*catalogues)

        return merged

    def with_blocks(
        self, blocks: Iterable[type], *, namespace: str = ""
    ) -> "Catalogue":
        """Return a new catalogue with additional blocks.

        Args:
            blocks: Block classes to add, e.g. assembled dynamic blocks.
            namespace: Namespace of the added blocks.

        Returns:
            A new catalogue; this one is unchanged.
        """
        extended = Catalogue.merge(self, Catalogue(blocks, namespace=namespace))

        return extended

    def with_contributions(
        self, contributions: Iterable[ImplementationContribution]
    ) -> "Catalogue":
        """Return a new catalogue whose blocks also list contributed implementations.

        Each block a contribution names gets an extended description: earlier
        contributions, then this one's implementations, then the block's own.
        Block classes and this catalogue are unchanged.

        Args:
            contributions: Contributions in the host's preference order.

        Returns:
            A new catalogue; this one is unchanged.

        Raises:
            CatalogueError: When a contribution names an unknown type or an
                alias, carries another contract, or lists an implementation
                the block cannot accept (``contributions.apply_contribution``).
        """
        extended_specs: Dict[str, BlockSpec] = {}
        for contribution in contributions:
            entry = self._contribution_target(contribution)
            current = extended_specs.get(entry.spec.type, entry.spec)
            extended_specs[entry.spec.type] = apply_contribution(current, contribution)

        extended = Catalogue()
        extended._add_catalogue(self, block_specs=extended_specs)

        return extended

    def contribution_problem(
        self, contribution: ImplementationContribution
    ) -> Optional[str]:
        """Tell why ``with_contributions`` would reject one contribution.

        Lets a host skip an incompatible contribution explicitly and record
        why, instead of handling an exception.

        Args:
            contribution: Contribution to check against this catalogue alone.

        Returns:
            ``None`` when it applies, otherwise the reason it does not.
        """
        try:
            self.with_contributions([contribution])
        except CatalogueError as error:
            return str(error)

        return None

    @property
    def block_types(self) -> Tuple[str, ...]:
        """Canonical block types in registration order."""
        return tuple(self._entries)

    @property
    def source_types(self) -> Tuple[str, ...]:
        """Canonical source types in registration order."""
        return tuple(self._source_entries)

    @property
    def operator_types(self) -> Tuple[str, ...]:
        """Canonical operator types in registration order."""
        return tuple(self._operator_entries)

    @property
    def kinds(self) -> Mapping[str, Kind]:
        """Kinds by name, including the wildcard."""
        return MappingProxyType(self._kinds)

    @property
    def providers(self) -> Mapping[str, Mapping[str, Any]]:
        """Resource providers by namespace and parameter name."""
        return MappingProxyType(
            {
                namespace: MappingProxyType(values)
                for namespace, values in self._providers.items()
            }
        )

    @property
    def codecs(self) -> Mapping[str, PayloadCodec]:
        """Recording codecs by codec name, in registration order."""
        return MappingProxyType(self._codecs)

    def find(self, identity: str) -> Optional[CatalogueEntry]:
        """Look up a block by canonical type or alias.

        Args:
            identity: Type or alias used by a workflow step.

        Returns:
            The entry, or ``None`` when unknown.
        """
        canonical = self._identities.get(identity)
        if canonical is None:
            return None

        return self._entries[canonical]

    def entry(self, identity: str) -> CatalogueEntry:
        """Look up a block by canonical type or alias.

        Args:
            identity: Type or alias used by a workflow step.

        Returns:
            The entry.

        Raises:
            CatalogueError: When the identity is unknown.
        """
        found = self.find(identity)
        if found is None:
            raise CatalogueError(
                f"Unknown block type {identity!r}; known types: {sorted(self._identities)}"
            )

        return found

    def find_source(self, identity: str) -> Optional[SourceEntry]:
        """Look up a source by canonical type or alias.

        Args:
            identity: Type or alias used by a ``sources`` declaration.

        Returns:
            The entry, or ``None`` when unknown.
        """
        canonical = self._source_identities.get(identity)
        if canonical is None:
            return None

        return self._source_entries[canonical]

    def source_entry(self, identity: str) -> SourceEntry:
        """Look up a source by canonical type or alias.

        Args:
            identity: Type or alias used by a ``sources`` declaration.

        Returns:
            The entry.

        Raises:
            CatalogueError: When the identity is unknown.
        """
        found = self.find_source(identity)
        if found is None:
            raise CatalogueError(
                f"Unknown source type {identity!r}; known source types: "
                f"{sorted(self._source_identities)}"
            )

        return found

    def find_operator(self, identity: str) -> Optional[OperatorEntry]:
        """Look up an operator by canonical type or alias.

        Args:
            identity: Type or alias used by an ``operators`` declaration.

        Returns:
            The entry, or ``None`` when unknown.
        """
        canonical = self._operator_identities.get(identity)
        if canonical is None:
            return None

        return self._operator_entries[canonical]

    def resolve_operator(self, identity: str) -> OperatorEntry:
        """Look up an operator by canonical type or alias.

        Args:
            identity: Type or alias used by an ``operators`` declaration.

        Returns:
            The entry; ``entry.spec.operator_class`` implements it.

        Raises:
            CatalogueError: When the identity is unknown.
        """
        found = self.find_operator(identity)
        if found is None:
            raise CatalogueError(
                f"Unknown operator type {identity!r}; known operator types: "
                f"{sorted(self._operator_identities)}"
            )

        return found

    def kind(self, name: str) -> Kind:
        """Look up a kind by name.

        Args:
            name: Kind name.

        Returns:
            The kind.

        Raises:
            CatalogueError: When the name is unknown.
        """
        if name not in self._kinds:
            raise CatalogueError(
                f"Unknown kind {name!r}; known kinds: {sorted(self._kinds)}"
            )

        return self._kinds[name]

    def describe(self) -> Dict[str, Any]:
        """Describe blocks and kinds without constructing any block.

        Returns:
            JSON-friendly catalogue description.
        """
        description = {
            "engine_version": V2_ENGINE_VERSION,
            "blocks": [entry.describe() for entry in self._entries.values()],
            "sources": [entry.describe() for entry in self._source_entries.values()],
            "operators": [
                entry.describe() for entry in self._operator_entries.values()
            ],
            "kinds": [_describe_kind(kind) for kind in self._kinds.values()],
        }

        return description

    def __contains__(self, identity: object) -> bool:
        return identity in self._identities

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"Catalogue(blocks={list(self._entries)}, "
            f"sources={list(self._source_entries)}, "
            f"operators={list(self._operator_entries)})"
        )

    def _add_catalogue(
        self,
        catalogue: "Catalogue",
        *,
        block_specs: Optional[Mapping[str, BlockSpec]] = None,
    ) -> None:
        # block_specs replaces the description of blocks by canonical type;
        # with_contributions passes the extended ones.
        replacements = block_specs or {}
        for kind in catalogue._kinds.values():
            self._add_kind(kind)
        for entry in catalogue._entries.values():
            spec = replacements.get(entry.spec.type, entry.spec)
            self._add_entry(CatalogueEntry(spec=spec, namespace=entry.namespace))
        for source_entry in catalogue._source_entries.values():
            self._add_source(source_entry)
        for operator_entry in catalogue._operator_entries.values():
            self._add_operator(operator_entry)
        for namespace, values in catalogue._providers.items():
            for name, value in values.items():
                self._add_provider(namespace, name=name, value=value)
        for codec in catalogue._codecs.values():
            self._add_codec(codec)

    def _contribution_target(
        self, contribution: ImplementationContribution
    ) -> CatalogueEntry:
        if not isinstance(contribution, ImplementationContribution):
            raise CatalogueError(
                f"Expected an ImplementationContribution, got {contribution!r}"
            )

        entry = self.find(contribution.block_type)
        if entry is None:
            raise CatalogueError(
                f"Contribution from {contribution.provider} {contribution.version} "
                f"names unknown block type {contribution.block_type!r}"
            )

        return entry

    def _add_entry(self, entry: CatalogueEntry) -> None:
        spec = entry.spec
        existing = self._entries.get(spec.type)
        if (
            existing is not None
            and existing.spec.block_class is spec.block_class
            and existing.namespace == entry.namespace
        ):
            problem = _implementation_conflict(existing.spec, spec)
            if problem is not None:
                raise CatalogueError(problem)
            return

        _require_compatible(spec)
        for identity in spec.identities:
            owner = self._identities.get(identity)
            if owner is not None:
                other_class = self._entries[owner].spec.block_class
                raise CatalogueError(
                    f"Identity {identity!r} of {spec.block_class.__qualname__} is "
                    f"already registered by {other_class.__qualname__}"
                )
        for kind in spec.kinds:
            self._add_kind(kind)

        self._entries[spec.type] = entry
        for identity in spec.identities:
            self._identities[identity] = spec.type

    def _add_source(self, entry: SourceEntry) -> None:
        spec = entry.spec
        existing = self._source_entries.get(spec.type)
        if (
            existing is not None
            and existing.spec.source_class is spec.source_class
            and existing.namespace == entry.namespace
        ):
            return

        _require_compatible(spec)
        for identity in spec.identities:
            owner = self._source_identities.get(identity)
            if owner is not None:
                other_class = self._source_entries[owner].spec.source_class
                raise CatalogueError(
                    f"Source identity {identity!r} of {spec.source_class.__qualname__} "
                    f"is already registered by {other_class.__qualname__}"
                )
        for kind in spec.kinds:
            self._add_kind(kind)

        self._source_entries[spec.type] = entry
        for identity in spec.identities:
            self._source_identities[identity] = spec.type

    def _add_operator(self, entry: OperatorEntry) -> None:
        spec = entry.spec
        existing = self._operator_entries.get(spec.type)
        if (
            existing is not None
            and existing.spec.operator_class is spec.operator_class
            and existing.namespace == entry.namespace
        ):
            return

        _require_compatible(spec)
        for identity in spec.identities:
            owner = self._operator_identities.get(identity)
            if owner is not None:
                other_class = self._operator_entries[owner].spec.operator_class
                raise CatalogueError(
                    f"Operator identity {identity!r} of "
                    f"{spec.operator_class.__qualname__} is already registered by "
                    f"{other_class.__qualname__}"
                )
        for kind in spec.kinds:
            self._add_kind(kind)

        self._operator_entries[spec.type] = entry
        for identity in spec.identities:
            self._operator_identities[identity] = spec.type

    def _add_kind(self, kind: Kind) -> None:
        if not isinstance(kind, Kind):
            raise CatalogueError(f"Catalogue kinds must be Kind objects, got {kind!r}")

        known = self._kinds.get(kind.name)
        if known is WILDCARD_KIND:
            self._kinds[kind.name] = kind
            return
        if kind is WILDCARD_KIND and known is not None:
            return

        known = self._kinds.setdefault(kind.name, kind)
        if known != kind:
            raise CatalogueError(
                f"Two different kinds are named {kind.name!r}; blocks must share "
                "one Kind object per name"
            )

    def _add_codec(self, codec: PayloadCodec) -> None:
        problem = codec_conflict(codec, by_name=self._codecs, by_type=self._codec_types)
        if problem is not None:
            raise CatalogueError(problem)

        self._codecs[codec.name] = codec
        self._codec_types[codec.type_name] = codec

    def _add_provider(self, namespace: str, *, name: str, value: Any) -> None:
        values = self._providers.setdefault(namespace, {})
        if name in values and values[name] is not value:
            raise CatalogueError(
                f"Namespace {namespace!r} has two different providers for resource {name!r}"
            )

        values[name] = value


def _spec_of_operator(operator_class: Any) -> "OperatorSpec":
    # Imported here: the operator contract imports declaration helpers, and
    # the plan imports this module, so a module-level import could cycle.
    from roboflow_workflows.v2.engine.operators.contract import (
        spec_of_operator,
    )

    try:
        spec = spec_of_operator(operator_class)
    except ContractError as error:
        raise CatalogueError(f"Cannot register {operator_class!r}: {error}") from error

    return spec


def _implementation_conflict(first: BlockSpec, second: BlockSpec) -> Optional[str]:
    # Why one block class cannot be kept once: its two descriptions list or
    # declare implementations differently. None when they match.
    listed = [(spec.name, spec.origin) for spec in first.implementations]
    if listed != [(spec.name, spec.origin) for spec in second.implementations]:
        problem = (
            f"{first.type} lists implementations {_describe_implementations(first)} "
            f"in one catalogue and {_describe_implementations(second)} in another; "
            "apply contributions once, after merging"
        )

        return problem

    changed = [
        a.name
        for a, b in zip(first.implementations, second.implementations)
        if not _same_implementation(a, b)
    ]
    if changed:
        problem = (
            f"{first.type} declares implementation(s) {changed} differently in two "
            "catalogues (class, requires, quality, phases or resources); apply "
            "contributions once, after merging"
        )

        return problem

    return None


def _same_implementation(a: ImplementationSpec, b: ImplementationSpec) -> bool:
    # Each spec is a snapshot of its class when a catalogue read it, and the
    # class may have changed since, so every field counts. Resource defaults
    # and annotations are user objects: only the same object matches, as in
    # an update's plan comparison; their own == is never called.
    same = (
        a.name == b.name
        and a.implementation_class is b.implementation_class
        and a.origin == b.origin
        and a.requires == b.requires
        and a.quality == b.quality
        and a.phases == b.phases
        and a.phase_overlap == b.phase_overlap
        and a.run_many == b.run_many
        and len(a.resources) == len(b.resources)
        and all(map(_same_resource, a.resources, b.resources))
    )

    return same


def _same_resource(a: ResourceSpec, b: ResourceSpec) -> bool:
    same = (
        a.name == b.name
        and a.required == b.required
        and a.default is b.default
        and a.annotation is b.annotation
    )

    return same


def _describe_implementations(spec: BlockSpec) -> str:
    described = [
        (
            implementation.name
            if implementation.origin is None
            else f"{implementation.name} ({implementation.origin.provider} "
            f"{implementation.origin.version})"
        )
        for implementation in spec.implementations
    ]

    return repr(described)


def _require_compatible(spec: Union[BlockSpec, SourceSpec, "OperatorSpec"]) -> None:
    if spec.engine_compatibility is None:
        return

    if Version(V2_ENGINE_VERSION) not in SpecifierSet(spec.engine_compatibility):
        raise CatalogueError(
            f"{spec.type} requires engine {spec.engine_compatibility}, but this "
            f"engine is {V2_ENGINE_VERSION}"
        )


def _describe_kind(kind: Kind) -> Dict[str, Any]:
    description = {
        "name": kind.name,
        "description": kind.description,
        "validates": kind.validate is not None,
        "deserializes": kind.deserialize is not None,
        "serializes": kind.serialize is not None,
        "converts_output": kind.convert_output is not None,
    }

    return description
