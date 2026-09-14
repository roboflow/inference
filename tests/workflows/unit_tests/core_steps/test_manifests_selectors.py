from inference.core.workflows.core_steps.loader import load_blocks
from inference.core.workflows.execution_engine.introspection.schema_parser import (
    parse_block_manifest,
)
from inference.enterprise.workflows.enterprise_blocks.loader import (
    load_enterprise_blocks,
)


def test_every_block_manifest_exposes_selectors_under_schema_property_names() -> None:
    # Execution Engine reads selector values from manifest instances using JSON schema
    # property names. For `validation_alias=AliasChoices(...)` the schema property is
    # named after the first choice - which must be the field name.
    violations = []
    for block in load_blocks() + load_enterprise_blocks():
        manifest_class = block.get_manifest()
        not_exposed = set(parse_block_manifest(manifest_class).selectors).difference(
            manifest_class.model_fields
        )
        if not_exposed:
            violations.append(
                f"{block.__module__}.{block.__name__}: {sorted(not_exposed)}"
            )

    assert not violations, (
        "Block manifests declare selectors under JSON schema property names that are "
        f"not manifest fields (check `AliasChoices` order): {violations}"
    )
