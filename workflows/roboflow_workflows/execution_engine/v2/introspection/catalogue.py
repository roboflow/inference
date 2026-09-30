"""Catalogue description: what blocks exist and how they can be wired.

Everything comes from class declarations (``spec_of``); no block, provider or
submitted dynamic code is touched.
"""

from typing import Any, Dict, List

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
        ``kinds``, ``blocks``, ``providers`` (constructor resources the
        catalogue supplies per namespace; factories are described, never
        called), ``connections`` and ``connection_inputs``. A field default
        is ``{"value": v}``, ``{"factory": True}`` (computed per step) or
        ``None`` when the field is required, so a ``None`` default stays
        distinguishable from no default. ``connections`` and each block's
        ``fields`` are keyed by Python field name, which may differ from the
        keys a step writes. ``connection_inputs`` has the same keys as
        ``connections`` and gives, per field, where a step writes the
        selector (``input_path``, nested object keys), every accepted
        location (``input_paths``) and the ``params_schema`` property
        describing the value (``schema_property``).
    """
    described = catalogue.describe()
    specs = [catalogue.entry(block_type).spec for block_type in catalogue.block_types]
    for block, spec in zip(described["blocks"], specs):
        block["identities"] = list(spec.identities)
        for name, field in block["fields"].items():
            field["default"] = _describe_default(spec, field_name=name)

    connections = _compatible_connections(specs)
    description = {
        "engine_version": described["engine_version"],
        "selector_grammar": SELECTOR_GRAMMAR,
        "kinds": described["kinds"],
        "blocks": described["blocks"],
        "providers": {
            namespace: {
                name: _describe_provider(value) for name, value in values.items()
            }
            for namespace, values in catalogue.providers.items()
        },
        "connections": connections,
        "connection_inputs": _connection_inputs(specs, connections=connections),
    }

    return description


def _describe_default(spec: BlockSpec, *, field_name: str) -> Any:
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


def _compatible_connections(specs: List[BlockSpec]) -> Dict[str, Dict[str, List[str]]]:
    """Per consumer type and data field: kind-compatible ``type.output`` producers.

    Only static outputs are listed; blocks whose outputs depend on their
    configuration are marked ``configured_outputs`` in their description.
    """
    producers = [
        (f"{spec.type}.{name}", output.kind_names)
        for spec in specs
        for name, output in spec.outputs.items()
    ]
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
