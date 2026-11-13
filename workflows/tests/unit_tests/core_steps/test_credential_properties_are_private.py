"""Census of the credential properties across the whole block library.

Roboflow's workflow sharing reads ``private`` from ``/workflows/blocks/describe``
to decide what it strips from a shared workflow, so a credential property without
the flag keeps the owner's value in shared copies and guest runs.

The census reads ``model_json_schema()`` instead of the ``Field`` calls. Pydantic
keeps an unknown ``Field`` keyword only while ``json_schema_extra`` is unset, so
``Field(private=True, json_schema_extra={...})`` loses the flag and only the schema
shows it.
"""

import re
from typing import Any, Dict, Iterator, List, Set, Tuple

import pytest
from roboflow_workflows.execution_engine.introspection import blocks_loader
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    get_manifest_type_identifiers,
    load_workflow_blocks,
)

ENTERPRISE_PLUGIN = "roboflow_workflows.enterprise_blocks.loader"
SECRET_KIND = "secret"
CREDENTIAL_PROPERTY_NAME = re.compile(
    r"password$|passwd$|passphrase$|secret|credential|api_?key|access_?key"
    r"|private_?key|(^|_)token$|account_sid|^headers$|^query_parameters$",
    flags=re.IGNORECASE,
)
# holds the names of environment variables, not their values
NOT_CREDENTIAL_VALUES = {
    ("roboflow_core/environment_secrets_store@v1", "variables_storing_secrets"),
}
# objects without a whole-field selector: they take selectors per value only
SELECTORS_PER_VALUE_ONLY = {
    ("roboflow_core/webhook_sink@v1", "headers"),
    ("roboflow_core/webhook_sink@v1", "query_parameters"),
}


@pytest.fixture(scope="module")
def block_schemas() -> Iterator[Dict[str, dict]]:
    """Schemas of the real registry: core blocks plus the enterprise plugin."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("WORKFLOWS_PLUGINS", ENTERPRISE_PLUGIN)
        blocks_loader.clear_caches()
        try:
            schemas = {}
            for block in load_workflow_blocks():
                schema = block.manifest_class.model_json_schema()
                identifiers = get_manifest_type_identifiers(
                    block_schema=schema,
                    block_source=block.block_source,
                    block_identifier=block.identifier,
                )
                schemas[identifiers[0]] = schema
            yield schemas
        finally:
            blocks_loader.clear_caches()


def _accepted_kinds(schema: Any) -> Set[str]:
    if isinstance(schema, list):
        children = schema
    elif isinstance(schema, dict):
        children = list(schema.values())
    else:
        return set()

    kinds = set()
    if isinstance(schema, dict) and schema.get("reference") is True:
        kinds.update(kind["name"] for kind in schema.get("kind", []))
    for child in children:
        kinds.update(_accepted_kinds(child))

    return kinds


def _accepts_selector(schema: dict) -> bool:
    variants = [schema, *schema.get("anyOf", [])]

    return any(variant.get("reference") is True for variant in variants)


def _iterate_properties(
    block_schemas: Dict[str, dict],
) -> List[Tuple[str, str, dict]]:
    return [
        (block_type, name, schema)
        for block_type, block_schema in sorted(block_schemas.items())
        for name, schema in block_schema.get("properties", {}).items()
    ]


def test_census_covers_core_and_enterprise_blocks(
    block_schemas: Dict[str, dict],
) -> None:
    assert "roboflow_core/s3_sink@v1" in block_schemas
    assert "roboflow_enterprise/kafka_producer_sink@v1" in block_schemas


def test_credential_properties_are_private(block_schemas: Dict[str, dict]) -> None:
    unmarked = [
        f"{block_type}.{name}"
        for block_type, name, schema in _iterate_properties(block_schemas)
        if (block_type, name) not in NOT_CREDENTIAL_VALUES
        and (
            CREDENTIAL_PROPERTY_NAME.search(name)
            or SECRET_KIND in _accepted_kinds(schema)
        )
        and schema.get("private") is not True
    ]

    assert unmarked == []


def test_private_properties_accept_a_whole_field_selector(
    block_schemas: Dict[str, dict],
) -> None:
    # sharing replaces a private literal with `$inputs.<step>__<key>`, and a property
    # without a selector variant would stop the shared workflow from compiling
    without_selector = [
        f"{block_type}.{name}"
        for block_type, name, schema in _iterate_properties(block_schemas)
        if schema.get("private") is True
        and (block_type, name) not in SELECTORS_PER_VALUE_ONLY
        and not _accepts_selector(schema)
    ]

    assert without_selector == []
