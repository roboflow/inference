"""The `inference` pipeline host of the stream manager.

Built inside each pipeline process from
`inference.core.interfaces.streams_configuration.LEGACY_PIPELINE_HOST_DESCRIPTOR`.
It resolves workflows exactly as `InferencePipeline.init_with_workflow` does -
through the same `prepare_workflow_for_pipeline` - with the manager's
historical arguments: definition cache on, no caller init parameters and the
default model manager stack.
"""

from typing import Any, Dict, Optional, Tuple

# Through the module, not the name, so a patch made through the historical
# `inference.core.interfaces.stream.inference_pipeline` name applies here too.
from inference.core.interfaces.legacy_stream import inference_pipeline


class LegacyPipelineHost:
    def prepare_workflow(
        self,
        *,
        workflow_specification: Optional[dict],
        workspace_name: Optional[str],
        workflow_id: Optional[str],
        workflow_version_id: Optional[str],
        api_key: Optional[str],
        profiler: Any,
    ) -> Tuple[dict, Dict[str, Any], Any]:
        prepared_workflow = inference_pipeline.prepare_workflow_for_pipeline(
            workflow_specification=workflow_specification,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
            api_key=api_key,
            use_workflow_definition_cache=True,
            workflow_init_parameters=None,
            model_manager=None,
            profiler=profiler,
        )

        return (
            prepared_workflow.workflow_specification,
            prepared_workflow.workflow_init_parameters,
            prepared_workflow.step_error_handler,
        )

    def close(self) -> None:
        # The host keeps nothing between requests: the model manager built for
        # a pipeline is owned by that pipeline's workflow.
        return None
