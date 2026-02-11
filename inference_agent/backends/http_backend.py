"""HTTP Backend: wraps InferenceHTTPClient for remote inference server communication."""

from __future__ import annotations

import asyncio
import logging
import time
from functools import partial
from typing import Optional

from inference_agent.core.protocols import InferenceResult, PipelineHandle

logger = logging.getLogger(__name__)


class HTTPBackend:
    """InferenceBackend implementation using the inference_sdk HTTP client.

    Supports:
    - Single-shot inference (model_id or workflow)
    - Hosted pipeline management (start/stop/pause/resume/consume)
    - Workflow execution by spec or by workspace/workflow ID
    """

    def __init__(self, api_url: str, api_key: Optional[str] = None):
        from inference_sdk import InferenceHTTPClient

        self._client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self._api_url = api_url
        self._pipelines: dict[str, PipelineHandle] = {}

    async def run_single(
        self,
        image: str | bytes,
        model_id: Optional[str] = None,
        workflow_spec: Optional[dict] = None,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        parameters: Optional[dict] = None,
    ) -> InferenceResult:
        """Run inference on a single image via the HTTP client."""
        loop = asyncio.get_event_loop()

        if model_id:
            result = await loop.run_in_executor(
                None,
                partial(self._client.infer, image, model_id=model_id),
            )
        elif workspace_name and workflow_id:
            kwargs = {
                "workspace_name": workspace_name,
                "workflow_id": workflow_id,
                "images": {"image": image},
            }
            if parameters:
                kwargs["parameters"] = parameters
            result = await loop.run_in_executor(
                None,
                partial(self._client.run_workflow, **kwargs),
            )
        elif workflow_spec:
            kwargs = {
                "specification": workflow_spec,
                "images": {"image": image},
            }
            if parameters:
                kwargs["parameters"] = parameters
            result = await loop.run_in_executor(
                None,
                partial(self._client.run_workflow, **kwargs),
            )
        else:
            raise ValueError(
                "Must provide model_id, workflow_spec, or workspace_name+workflow_id"
            )

        predictions = result if isinstance(result, dict) else {"outputs": result}
        return InferenceResult(predictions=predictions, timestamp=time.time())

    async def start_pipeline(
        self,
        video_reference: str,
        workflow_spec: Optional[dict] = None,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        max_fps: Optional[float] = None,
        parameters: Optional[dict] = None,
    ) -> PipelineHandle:
        """Start a hosted inference pipeline on the server."""
        loop = asyncio.get_event_loop()

        kwargs: dict = {"video_reference": video_reference}
        if workflow_spec:
            kwargs["workflow_specification"] = workflow_spec
        if workspace_name:
            kwargs["workspace_name"] = workspace_name
        if workflow_id:
            kwargs["workflow_id"] = workflow_id
        if max_fps is not None:
            kwargs["max_fps"] = max_fps
        if parameters:
            kwargs["workflows_parameters"] = parameters

        response = await loop.run_in_executor(
            None,
            partial(
                self._client.start_inference_pipeline_with_workflow, **kwargs
            ),
        )

        pipeline_id = response.get("pipeline_id", str(response))
        description = f"workflow={workflow_id or 'custom'} on {video_reference}"
        handle = PipelineHandle(
            pipeline_id=pipeline_id,
            backend="http",
            video_reference=video_reference,
            workflow_description=description,
            started_at=time.time(),
            status="running",
        )
        self._pipelines[pipeline_id] = handle
        return handle

    async def consume_results(
        self,
        pipeline_id: str,
        max_results: int = 5,
    ) -> list[InferenceResult]:
        """Consume latest results from a hosted pipeline."""
        loop = asyncio.get_event_loop()
        results: list[InferenceResult] = []

        for _ in range(max_results):
            try:
                result = await loop.run_in_executor(
                    None,
                    partial(
                        self._client.consume_inference_pipeline_result,
                        pipeline_id,
                    ),
                )
                if result:
                    predictions = (
                        result if isinstance(result, dict) else {"outputs": result}
                    )
                    results.append(
                        InferenceResult(
                            predictions=predictions, timestamp=time.time()
                        )
                    )
            except Exception as e:
                logger.debug("No more pipeline results: %s", e)
                break

        return results

    async def stop_pipeline(self, pipeline_id: str) -> None:
        """Terminate a hosted pipeline."""
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                partial(
                    self._client.terminate_inference_pipeline, pipeline_id
                ),
            )
        except Exception as e:
            logger.warning("Error stopping pipeline %s: %s", pipeline_id, e)
        if pipeline_id in self._pipelines:
            self._pipelines[pipeline_id].status = "stopped"

    async def pause_pipeline(self, pipeline_id: str) -> None:
        """Pause a hosted pipeline."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(self._client.pause_inference_pipeline, pipeline_id),
        )
        if pipeline_id in self._pipelines:
            self._pipelines[pipeline_id].status = "paused"

    async def resume_pipeline(self, pipeline_id: str) -> None:
        """Resume a paused hosted pipeline."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            partial(self._client.resume_inference_pipeline, pipeline_id),
        )
        if pipeline_id in self._pipelines:
            self._pipelines[pipeline_id].status = "running"

    async def list_pipelines(self) -> list[PipelineHandle]:
        """List all tracked pipelines."""
        # Also try to get live status from the server
        loop = asyncio.get_event_loop()
        try:
            status_response = await loop.run_in_executor(
                None,
                self._client.list_inference_pipelines,
            )
            if isinstance(status_response, list):
                # Update local tracking with server state
                server_ids = set()
                for entry in status_response:
                    pid = entry.get("id", entry.get("pipeline_id", ""))
                    server_ids.add(pid)
                    if pid in self._pipelines:
                        self._pipelines[pid].status = entry.get(
                            "status", self._pipelines[pid].status
                        )
        except Exception as e:
            logger.debug("Could not fetch live pipeline status: %s", e)

        return list(self._pipelines.values())

    async def get_pipeline_status(self, pipeline_id: str) -> PipelineHandle:
        """Get status of a specific pipeline."""
        if pipeline_id in self._pipelines:
            return self._pipelines[pipeline_id]
        raise ValueError(f"Unknown pipeline: {pipeline_id}")
