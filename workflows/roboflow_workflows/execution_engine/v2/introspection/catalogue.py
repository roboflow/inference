"""Catalogue description: what blocks, sources and operators exist and how they wire.

Everything comes from class declarations (``spec_of``, ``spec_of_source``,
``spec_of_operator``); no block, implementation, source, operator, provider or
submitted dynamic code is touched, so no model is loaded.
"""

from typing import Any, Dict, List, Sequence, Tuple

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.declaration import (
    BATCH_MODES,
    DATA_SELECTOR_PATTERN,
    SELECTOR_SEGMENT,
    STEP_SELECTOR_PATTERN,
    BlockSpec,
)
from roboflow_workflows.execution_engine.v2.kinds import kinds_compatible
from roboflow_workflows.execution_engine.v2.resources import Factory
from roboflow_workflows.execution_engine.v2.sources import SourceSpec

Producer = Tuple[str, Tuple[str, ...]]
"""A ``type.output`` producer id and the kinds it produces."""

SELECTOR_GRAMMAR: Dict[str, Any] = {
    "data_selector": DATA_SELECTOR_PATTERN,
    "step_selector": STEP_SELECTOR_PATTERN,
    "name_segment": SELECTOR_SEGMENT,
    "roles": {
        "item": "one element per invocation",
        "group": "the group of children of each invocation, as a Batch",
        "step": "a control target ($steps.<name>)",
    },
    "batch_modes": list(BATCH_MODES),
    "positions": "the whole field, one list element or one dict value",
    "operator_input_maps": {
        "inputs": "values an alignment operator relates",
        "collect": "values a window collects over time",
        "hold": "parent values a window keeps from its last arrival",
    },
}


def describe_catalogue(catalogue: Catalogue) -> Dict[str, Any]:
    """Describe the blocks, kinds and possible connections of a catalogue.

    Extends ``Catalogue.describe()`` with the selector grammar, each field's
    default and, per consumer field, the producer outputs whose kinds it
    accepts (the V2 counterpart of V1's connection discovery).

    Args:
        catalogue: Catalogue to describe.

    Returns:
        JSON-friendly mapping with ``engine_version``, ``selector_grammar``,
        ``kinds``, ``blocks``, ``sources``, ``operators``, ``providers`` (constructor
        resources the catalogue supplies per namespace; factories are
        described, never called), ``connections``, ``source_connections``
        and ``connection_inputs``. A field default is ``{"value": v}``,
        ``{"factory": True}`` (computed per step) or ``None`` when the field
        is required, so a ``None`` default stays distinguishable from no
        default. ``connections`` and each block's ``fields`` are keyed by
        Python field name, which may differ from the keys a step writes.
        ``connections`` lists, per consumer block type and field, the
        kind-compatible ``type.output`` block producers; ``source_connections``
        lists the kind-compatible ``type.port`` source producers under the
        same keys, kept apart because a source is declared in ``sources``,
        not in ``steps``. ``connection_inputs`` has the same keys as
        ``connections`` and gives, per field, where a step writes the
        selector (``input_path``, nested object keys), every accepted
        location (``input_paths``) and the ``params_schema`` property
        describing the value (``schema_property``). Each source lists its
        identities, ports with source-local axes, constructor resources and
        the static parameters it accepts (with defaults). Each operator lists
        its identities, accepted input roles and literal parameters (with
        defaults); its ports depend on the bound inputs, so it is no static
        producer. Each block lists its ``implementations`` in selection
        order: name, class, required target capabilities, constructor
        resources and phase graph (``None`` when it only has ``run``). An
        ordinary block has the single implementation ``default``.
    """
    described = catalogue.describe()
    specs = [catalogue.entry(block_type).spec for block_type in catalogue.block_types]
    for block, spec in zip(described["blocks"], specs):
        block["identities"] = list(spec.identities)
        for name, field in block["fields"].items():
            field["default"] = _describe_default(spec, field_name=name)
    source_specs = [
        catalogue.source_entry(source_type).spec
        for source_type in catalogue.source_types
    ]
    for source, spec in zip(described["sources"], source_specs):
        source["identities"] = list(spec.identities)
        for name, field in source["fields"].items():
            field["default"] = _describe_default(spec, field_name=name)

    operator_specs = [
        catalogue.resolve_operator(operator_type).spec
        for operator_type in catalogue.operator_types
    ]
    for operator, spec in zip(described["operators"], operator_specs):
        operator["identities"] = list(spec.identities)
        operator["parameters"] = {
            name: {"default": _describe_default(spec, field_name=name)}
            for name in spec.params_model.model_fields
        }

    connections = _compatible_connections(specs, producers=_block_producers(specs))
    description = {
        "engine_version": described["engine_version"],
        "selector_grammar": SELECTOR_GRAMMAR,
        "kinds": described["kinds"],
        "blocks": described["blocks"],
        "sources": described["sources"],
        "operators": described["operators"],
        "providers": {
            namespace: {
                name: _describe_provider(value) for name, value in values.items()
            }
            for namespace, values in catalogue.providers.items()
        },
        "connections": connections,
        "source_connections": _compatible_connections(
            specs, producers=_source_producers(source_specs)
        ),
        "connection_inputs": _connection_inputs(specs, connections=connections),
    }

    return description


def _describe_default(spec: Any, *, field_name: str) -> Any:
    info = spec.params_model.model_fields[field_name]
    if info.is_required():
        return None
    if info.default_factory is not None:
        return {"factory": True}

    default = {"value": info.default}

    return default


def _describe_provider(value: Any) -> Dict[str, Any]:
    if isinstance(value, Factory):
        created_by = getattr(value.create, "__qualname__", repr(value.create))
        description = {"factory": created_by, "scope": value.scope}
        return description

    description = {"value_type": type(value).__qualname__}

    return description


def _block_producers(specs: List[BlockSpec]) -> List[Producer]:
    """Static block outputs as ``type.output`` producers.

    Blocks whose outputs depend on their configuration are marked
    ``configured_outputs`` in their description and list none here.
    """
    producers = [
        (f"{spec.type}.{name}", output.kind_names)
        for spec in specs
        for name, output in spec.outputs.items()
    ]

    return producers


def _source_producers(specs: List[SourceSpec]) -> List[Producer]:
    """Every source port as a ``type.port`` producer."""
    producers = [
        (f"{spec.type}.{name}", output.kind_names)
        for spec in specs
        for name, output in spec.outputs.items()
    ]

    return producers


def _compatible_connections(
    specs: List[BlockSpec], *, producers: Sequence[Producer]
) -> Dict[str, Dict[str, List[str]]]:
    """Per consumer type and data field: the kind-compatible producers."""
    connections: Dict[str, Dict[str, List[str]]] = {}
    for spec in specs:
        fields = {}
        for name, field in spec.fields.items():
            accepted = [
                kind_name
                for marker in field.markers
                if marker.role != "step"
                for kind_name in marker.kind_names
            ]
            if not accepted:
                continue
            fields[name] = [
                producer
                for producer, produced in producers
                if kinds_compatible(produced, accepted)
            ]
        connections[spec.type] = fields

    return connections


def _connection_inputs(
    specs: List[BlockSpec], *, connections: Dict[str, Dict[str, List[str]]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Per consumer type and connectable field: the step keys to write."""
    inputs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in specs:
        described = {}
        for name in connections[spec.type]:
            field = spec.fields[name].describe()
            described[name] = {
                key: field[key]
                for key in ("input_path", "input_paths", "schema_property")
            }
        inputs[spec.type] = described

    return inputs
