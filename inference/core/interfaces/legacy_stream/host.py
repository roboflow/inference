"""The `inference` pipeline host of the stream manager.

Built inside each pipeline process from
`inference.core.interfaces.streams_configuration.LEGACY_PIPELINE_HOST_DESCRIPTOR`.
It resolves workflows exactly as `InferencePipeline.init_with_workflow` does -
through the same `prepare_workflow_for_pipeline` - with the manager's
historical arguments: definition cache on, no caller init parameters and the
default model manager stack.
It also serves the WebRTC worker: each worker method below forwards to the
function the worker called directly before it moved out of `inference`.
"""

from typing import Any, Dict, Optional, Tuple

# Via the module, so a patch made through the historical name applies here too.
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
        # Nothing kept between requests: the model manager is owned by the workflow.
        return None

    def init_workflow_pipeline(self, **kwargs: Any) -> Any:
        """Build the session pipeline with the `inference` pipeline factory.

        Args:
            **kwargs: Arguments of `InferencePipeline.init_with_workflow`,
                passed on unchanged.

        Returns:
            The pipeline the factory built.
        """
        pipeline = inference_pipeline.InferencePipeline.init_with_workflow(**kwargs)

        return pipeline

    def get_workflow_specification(
        self,
        *,
        api_key: Optional[str],
        workspace_id: Optional[str],
        workflow_id: Optional[str],
        workflow_version_id: Optional[str],
    ) -> dict:
        """Fetch a registered workflow definition from the Roboflow API.

        Args:
            api_key: API key sent with the request, if any.
            workspace_id: Workspace of the workflow.
            workflow_id: Identifier of the workflow.
            workflow_version_id: Version of the workflow.

        Returns:
            The workflow definition.
        """
        # Imported on use: stream manager pipeline processes never need these.
        from inference.core import roboflow_api

        specification = roboflow_api.get_workflow_specification(
            api_key=api_key,
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
        )

        return specification

    async def async_push_usage_payloads(self) -> None:
        """Flush the usage collector from the running event loop."""
        from inference.usage_tracking.collector import usage_collector

        await usage_collector.async_push_usage_payloads()

    def open_mjpeg_player(self, url: str, *, allow_non_global_addresses: bool) -> Any:
        """Open an MJPEG source through the address-validating HTTP opener.

        Args:
            url: MJPEG source address.
            allow_non_global_addresses: Whether private addresses are allowed.

        Returns:
            The media player of the source.
        """
        from inference.core.utils import mjpeg

        player = mjpeg.open_mjpeg_player(
            url, allow_non_global_addresses=allow_non_global_addresses
        )

        return player
