from dataclasses import dataclass
from typing import Dict, FrozenSet, List

from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    get_manifest_type_identifiers,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
)

# These sink versions predate the `disable_sink` convention. Keep this compatibility
# list until workflows using them can move to versions with native no-op support.
LEGACY_SINK_TYPES = {
    "roboflow_core/local_file_sink@v1",
    "roboflow_core/onvif_sink@v1",
    "roboflow_core/roboflow_custom_metadata@v1",
    "RoboflowCustomMetadata",
    "roboflow_core/model_monitoring_inference_aggregator@v1",
    "roboflow_core/s3_sink@v1",
    "roboflow_core/modbus_tcp@v1",
    "roboflow_core/sinks@v1",
    "roboflow_core/microsoft_sql_server_sink@v1",
    "mqtt_writer_sink@v1",
}


@dataclass(frozen=True)
class SinkDisablingResult:
    workflow_definition: dict
    steps_without_native_noop: FrozenSet[str]


def disable_workflow_sinks(
    workflow_definition: dict,
    available_blocks: List[BlockSpecification],
) -> SinkDisablingResult:
    block_types = _index_block_types(available_blocks=available_blocks)
    steps_without_native_noop = set()
    updated_steps = []
    for step in workflow_definition.get("steps", []):
        step_type = step.get("type")
        block = block_types.get(step_type)
        if block is not None and "disable_sink" in block.manifest_class.model_fields:
            updated_steps.append({**step, "disable_sink": True})
        else:
            updated_steps.append(step)
            if step_type in LEGACY_SINK_TYPES:
                steps_without_native_noop.add(step["name"])
    return SinkDisablingResult(
        workflow_definition={**workflow_definition, "steps": updated_steps},
        steps_without_native_noop=frozenset(steps_without_native_noop),
    )


def _index_block_types(
    available_blocks: List[BlockSpecification],
) -> Dict[str, BlockSpecification]:
    result = {}
    for block in available_blocks:
        identifiers = get_manifest_type_identifiers(
            block_schema=block.manifest_class.model_json_schema(),
            block_source=block.block_source,
            block_identifier=block.identifier,
        )
        result.update({identifier: block for identifier in identifiers})
    return result
