# TODO - for everyone: start migrating other handlers to bring relief to http_api.py
import copy
from typing import Any, List, Optional

from roboflow_workflows.http_contract.describe import (  # noqa: F401
    describe_workflow_interface as handle_describe_workflows_interface,
)
from roboflow_workflows.http_contract.describe import (  # noqa: F401
    describe_workflows_blocks,
    filter_out_unwanted_workflow_outputs,
    get_unique_kinds,
)

from inference.core.cache.air_gapped import has_cached_model_variant
from inference.core.entities.responses.workflows import WorkflowsBlocksDescription
from inference.core.env import ENABLE_BUILDER
from inference.core.interfaces.roboflow_platform_client import SERVER_WORKSPACE_RESOLVER
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    DynamicBlockDefinition,
)
from inference.core.workflows.prototypes.block import BlockAirGappedInfo


def handle_describe_workflows_blocks_request(
    dynamic_blocks_definitions: Optional[List[DynamicBlockDefinition]] = None,
    requested_execution_engine_version: Optional[str] = None,
    api_key: Optional[str] = None,
    air_gapped: bool = False,
) -> WorkflowsBlocksDescription:
    result = describe_workflows_blocks(
        dynamic_blocks_definitions=dynamic_blocks_definitions,
        requested_execution_engine_version=requested_execution_engine_version,
        api_key=api_key,
        workspace_resolver=SERVER_WORKSPACE_RESOLVER,
    )
    if air_gapped and ENABLE_BUILDER:
        result = enrich_with_air_gapped_info(result)
    return result


def enrich_with_air_gapped_info(
    result: WorkflowsBlocksDescription,
) -> WorkflowsBlocksDescription:
    """Post-process block descriptions to include air-gapped availability info.

    Deep-copies block schemas before mutating so the LRU-cached objects are
    not modified.
    """
    enriched_blocks = []
    for block in result.blocks:
        manifest_cls = block.manifest_class
        air_gapped_info = _get_air_gapped_info_for_block(manifest_cls)
        enriched_schema = copy.deepcopy(block.block_schema)
        if "json_schema_extra" not in enriched_schema:
            enriched_schema["json_schema_extra"] = {}
        enriched_schema["json_schema_extra"][
            "air_gapped_info"
        ] = air_gapped_info.to_dict()
        enriched_blocks.append(
            block.model_copy(update={"block_schema": enriched_schema})
        )
    return result.model_copy(update={"blocks": enriched_blocks})


def _get_air_gapped_info_for_block(
    manifest_cls: Any,
) -> BlockAirGappedInfo:
    """Determine air-gapped availability for a single block manifest class.

    Resolution order:

    1. **Cloud-only blocks** — ``get_air_gapped_availability()`` returns
       ``available=False`` (e.g. blocks wrapping OpenAI / Anthropic APIs).
    2. **Foundation-model blocks** — ``get_supported_model_variants()``
       returns a non-None list; availability depends on whether any
       variant has cached weights.
    3. **Default** — the block is available (pure logic, local-network, etc.).

    Compatible task types from ``get_compatible_task_types()`` are always
    attached when present.
    """
    task_types = manifest_cls.get_compatible_task_types()

    # 1. Explicit cloud/internet declaration
    availability = manifest_cls.get_air_gapped_availability()
    if not availability.available:
        return BlockAirGappedInfo(
            available=False,
            reason=availability.reason,
            compatible_task_types=task_types,
        )

    # 2. Foundation models with locally-cacheable weights
    model_variants = manifest_cls.get_supported_model_variants()
    if model_variants is not None:
        cached = has_cached_model_variant(model_variants)
        # Use the first variant as a representative identifier for the UI.
        # The list is ordered by convention (default/primary variant first),
        # and individual variant selection happens at workflow-execution time.
        representative_id = model_variants[0] if model_variants else None
        return BlockAirGappedInfo(
            available=cached,
            reason=None if cached else "missing_cache_artifacts",
            model_id=representative_id,
            compatible_task_types=task_types,
        )

    # 3. Default: block is available (pure logic blocks, local-network blocks, etc.)
    return BlockAirGappedInfo(
        available=True,
        compatible_task_types=task_types,
    )
