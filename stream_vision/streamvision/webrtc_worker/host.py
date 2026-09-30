"""The host contract of the WebRTC worker.

The worker is host-neutral: it does not know how a workflow pipeline is
built, how usage is reported or how an MJPEG source is opened safely. The
process host supplies all of that. It is the host named by the pipeline host
descriptor installed for this process, built once on first use.

This module must stay import-light: no aiortc, av or pipeline imports.
"""

import threading
from typing import Any, Dict, Optional, Protocol

from streamvision.stream_manager.manager_app.host import (
    PipelineHost,
    create_pipeline_host,
    resolve_host_descriptor,
)


class WebRTCWorkerHost(PipelineHost, Protocol):
    """What the WebRTC worker needs from its host, on top of `PipelineHost`."""

    def init_workflow_pipeline(
        self,
        *,
        video_reference: Any,
        workflow_specification: Optional[dict],
        workspace_name: Optional[str],
        workflow_id: Optional[str],
        api_key: Optional[str],
        image_input_name: str,
        workflows_parameters: Optional[Dict[str, Any]],
        disable_sinks: bool,
        workflows_thread_pool_workers: int,
        execution_engine_thread_pool_workers: int,
        cancel_thread_pool_tasks_on_exit: bool,
        video_metadata_input_name: str,
        model_manager: Optional[Any],
        _is_preview: bool,
        workflow_version_id: Optional[str],
    ) -> Any:
        """Build the pipeline that runs the session's workflow.

        Args:
            video_reference: Frame producer type the pipeline reads from.
            workflow_specification: Inline workflow definition, if any.
            workspace_name: Workspace of a registered workflow.
            workflow_id: Identifier of a registered workflow.
            api_key: API key sent with the request, if any.
            image_input_name: Workflow input receiving the video frames.
            workflows_parameters: Additional workflow input values.
            disable_sinks: Whether sink writes and notifications are disabled.
            workflows_thread_pool_workers: Workers of the blocks thread pool.
            execution_engine_thread_pool_workers: Workers of the engine pool.
            cancel_thread_pool_tasks_on_exit: Whether pending tasks are
                cancelled when the pipeline stops.
            video_metadata_input_name: Workflow input receiving frame metadata.
            model_manager: Host-owned models object, passed through untouched.
            _is_preview: Whether the session is a preview.
            workflow_version_id: Version of a registered workflow.

        Returns:
            A pipeline exposing `_on_video_frame`.
        """

    def get_workflow_specification(
        self,
        *,
        api_key: Optional[str],
        workspace_id: Optional[str],
        workflow_id: Optional[str],
        workflow_version_id: Optional[str],
    ) -> dict:
        """Fetch the definition of a registered workflow.

        Args:
            api_key: API key sent with the request, if any.
            workspace_id: Workspace of the workflow.
            workflow_id: Identifier of the workflow.
            workflow_version_id: Version of the workflow.

        Returns:
            The workflow definition.
        """

    async def async_push_usage_payloads(self) -> None:
        """Flush the usage recorded during the session."""

    def open_mjpeg_player(self, url: str, *, allow_non_global_addresses: bool) -> Any:
        """Open an MJPEG source over a connection the host has validated.

        Args:
            url: MJPEG source address.
            allow_non_global_addresses: Whether private addresses are allowed.

        Returns:
            A media player exposing a `video` track.
        """


_HOST: Optional[WebRTCWorkerHost] = None
_HOST_LOCK = threading.Lock()


def get_webrtc_worker_host() -> WebRTCWorkerHost:
    """Return this process's WebRTC worker host, building it on first use.

    Returns:
        The host built from the installed pipeline host descriptor.

    Raises:
        PipelineHostNotConfiguredError: No host descriptor is installed.
    """
    global _HOST
    with _HOST_LOCK:
        if _HOST is None:
            _HOST = create_pipeline_host(resolve_host_descriptor(None))

        return _HOST
