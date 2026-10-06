"""Workflows plugin holding the Roboflow-platform blocks.

These blocks talk to the Roboflow API (dataset upload, custom metadata, model
monitoring, vision events, asset-library attributes, visual search). They used
to live in `inference/core/workflows/core_steps/{sinks,integrations}/roboflow`
and were the last reason that package imported `inference.core.roboflow_api`
and `inference.core.active_learning`.

`BLOCKS_SOURCE` is deliberately the core source name: the server supplies these
blocks' init parameters as `workflows_core.{cache,api_key,background_tasks,
thread_pool_executor,update_attributes_offloader,disable_sinks}`, and
`retrieve_init_parameter_values` resolves `{block_source}.{param}` first.
Re-tagging it breaks every one of them - see the regression test in
`tests/inference/unit_tests/core/test_roboflow_plugin_blocks.py`.

The block-disable policy is applied HERE. `core_steps/loader.load_blocks()`
filters its own list through `_should_filter_block` (WORKFLOW_DISABLED_BLOCK_TYPES
/ WORKFLOW_DISABLED_BLOCK_PATTERNS against block type, class name, display name
and module path), but `blocks_loader.load_workflow_blocks` applies only
engine-version compatibility to plugin blocks. Moving the blocks out of the core
list must not silently re-enable them, so `load_blocks()` runs the same predicate
- and, because the predicate matches the CURRENT module path, also matches every
pattern against the blocks' pre-move path (`…core_steps.sinks.roboflow.…`,
`…core_steps.integrations.roboflow.…`) so an operator's existing pattern keeps
disabling the same blocks.
"""

from typing import List, Type

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps import loader as core_loader
from inference.core.workflows.prototypes.block import WorkflowBlock
from inference.roboflow_workflows_plugin.integrations.visual_search.v1 import (
    RoboflowVisualSearchBlockV1,
)
from inference.roboflow_workflows_plugin.sinks.asset_library_attributes.v1 import (
    RoboflowAssetLibraryAttributesBlockV1,
)

if not ENABLE_TENSOR_DATA_REPRESENTATION:
    from inference.roboflow_workflows_plugin.integrations.visual_search_classifier.v1 import (
        RoboflowVisualSearchClassifierBlockV1,
    )
    from inference.roboflow_workflows_plugin.sinks.custom_metadata.v1 import (
        RoboflowCustomMetadataBlockV1,
    )
    from inference.roboflow_workflows_plugin.sinks.dataset_upload.v1 import (
        RoboflowDatasetUploadBlockV1,
    )
    from inference.roboflow_workflows_plugin.sinks.dataset_upload.v2 import (
        RoboflowDatasetUploadBlockV2,
    )
    from inference.roboflow_workflows_plugin.sinks.model_monitoring_inference_aggregator.v1 import (
        ModelMonitoringInferenceAggregatorBlockV1,
    )
    from inference.roboflow_workflows_plugin.sinks.vision_events.v1 import (
        RoboflowVisionEventsBlockV1,
    )
    from inference.roboflow_workflows_plugin.sinks.vision_events_bundle.v1 import (
        VisionEventBundleSinkBlockV1,
    )
else:
    from inference.roboflow_workflows_plugin.integrations.visual_search_classifier.v1_tensor import (
        RoboflowVisualSearchClassifierBlockV1,
    )
    from inference.roboflow_workflows_plugin.sinks.custom_metadata.v1_tensor import (
        RoboflowCustomMetadataBlockV1,
    )
    from inference.roboflow_workflows_plugin.sinks.dataset_upload.v1_tensor import (
        RoboflowDatasetUploadBlockV1,
    )
    from inference.roboflow_workflows_plugin.sinks.dataset_upload.v2_tensor import (
        RoboflowDatasetUploadBlockV2,
    )
    from inference.roboflow_workflows_plugin.sinks.model_monitoring_inference_aggregator.v1_tensor import (
        ModelMonitoringInferenceAggregatorBlockV1,
    )
    from inference.roboflow_workflows_plugin.sinks.vision_events.v1_tensor import (
        RoboflowVisionEventsBlockV1,
    )
    from inference.roboflow_workflows_plugin.sinks.vision_events_bundle.v1_tensor import (
        VisionEventBundleSinkBlockV1,
    )

BLOCKS_SOURCE = "workflows_core"

# Where these modules lived before Phase 9. WORKFLOW_DISABLED_BLOCK_PATTERNS
# written against the old path must keep matching.
_LEGACY_MODULE_PREFIXES = (
    (
        "inference.roboflow_workflows_plugin.sinks.",
        "inference.core.workflows.core_steps.sinks.roboflow.",
    ),
    (
        "inference.roboflow_workflows_plugin.integrations.",
        "inference.core.workflows.core_steps.integrations.roboflow.",
    ),
)


def _legacy_module_name(block_class: Type[WorkflowBlock]) -> str:
    module = block_class.__module__
    for new_prefix, old_prefix in _LEGACY_MODULE_PREFIXES:
        if module.startswith(new_prefix):
            return old_prefix + module[len(new_prefix) :]
    return module


def _is_disabled(block_class: Type[WorkflowBlock]) -> bool:
    # The core policy - block type, class name, display name and the CURRENT
    # module path - through the very predicate core_steps/loader.py uses, read
    # through the module so a test's mock.patch.object on the core loader's
    # WORKFLOW_DISABLED_BLOCK_* names applies here too.
    if core_loader._should_filter_block(block_class):
        return True
    legacy_module = _legacy_module_name(block_class).lower()
    return any(
        pattern.lower() in legacy_module
        for pattern in core_loader.WORKFLOW_DISABLED_BLOCK_PATTERNS
    )


def load_blocks() -> List[Type[WorkflowBlock]]:
    blocks = [
        RoboflowVisualSearchBlockV1,
        RoboflowVisualSearchClassifierBlockV1,
        RoboflowDatasetUploadBlockV1,
        RoboflowAssetLibraryAttributesBlockV1,
        RoboflowCustomMetadataBlockV1,
        ModelMonitoringInferenceAggregatorBlockV1,
        RoboflowDatasetUploadBlockV2,
        RoboflowVisionEventsBlockV1,
        VisionEventBundleSinkBlockV1,
    ]
    return [block for block in blocks if not _is_disabled(block)]
