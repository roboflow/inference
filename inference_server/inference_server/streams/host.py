"""The server's pipeline host of the stream manager.

Built inside each pipeline process from `SERVER_PIPELINE_HOST_DESCRIPTOR`. It
resolves workflows and their Execution Engine bindings through the server's
gateway stack: a gateway resolved in the pipeline process and driven on a
private event loop, the legacy model bridge over it and the gateway-backed
models provider the Workflow routes use. The loop thread, the gateway, the
usage collector of the process and the image codec binding start on first use
and live until `close()`. Every workflow run of a pipeline records one usage
row through the collector, the way a request does on the HTTP server.
"""

import asyncio
import logging
import os
import threading
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

from roboflow_workflows.prototypes.observer import NULL_EXECUTION_OBSERVER
from streamvision.stream.exceptions import MissingApiKeyError
from streamvision.stream_manager.manager_app.host import PipelineHostDescriptor

from inference_server import configuration, gateway_resolver
from inference_server.legacy.bridge import (
    LegacyModelBridge,
    LoopBridge,
    SyncLegacyBridge,
)
from inference_server.usage import collector as usage_collector_module
from inference_server.usage.observer import StreamUsageExecutionObserver
from inference_server.workflows import execution
from inference_server.workflows import host as workflows_host
from inference_server.workflows.models_provider import GatewayModelsProvider

logger = logging.getLogger(__name__)

CLOSE_TIMEOUT_S = 30.0
MISSING_WORKFLOW_MESSAGE = (
    "Either (`workspace_name`, `workflow_id`) or `workflow_specification` must be "
    "provided."
)
MISSING_API_KEY_MESSAGE = (
    "Roboflow API key needs to be provided either as parameter or via env variable "
    "ROBOFLOW_API_KEY. If you do not know how to get API key - visit "
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to "
    "learn how to retrieve one."
)


class ServerPipelineHost:
    """Pipeline host resolving workflows through the server's gateway stack.

    Args:
        gateway_kind: Gateway name used when the process environment does not
            name one through `INFERENCE_GATEWAY`.
    """

    def __init__(self, *, gateway_kind: str) -> None:
        self._gateway_kind = gateway_kind
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._gateway: Any = None
        self._bridge: Optional[LegacyModelBridge] = None
        self._loop_bridge: Optional[LoopBridge] = None
        self._collector: Any = None
        self._closed = False

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
        """Resolve a workflow and the Execution Engine bindings it runs with.

        Args:
            workflow_specification: Inline workflow definition, if any.
            workspace_name: Workspace of a registered workflow.
            workflow_id: Identifier of a registered workflow.
            workflow_version_id: Version of a registered workflow.
            api_key: API key sent with the request; the server's default key
                applies when `None`.
            profiler: Workflows profiler recording the definition fetch.

        Returns:
            The workflow specification, the Execution Engine init parameters
            and the step error handler.

        Raises:
            ValueError: Neither an inline specification nor a workspace name
                and workflow id are given.
            MissingApiKeyError: A registered workflow must be fetched without
                an API key.
            RuntimeError: The host was already closed.
        """
        if api_key is None:
            api_key = configuration.DEFAULT_API_KEY
        named_workflow_specified = (workspace_name is not None) and (
            workflow_id is not None
        )
        if not named_workflow_specified and not workflow_specification:
            raise ValueError(MISSING_WORKFLOW_MESSAGE)

        named_from_registry = workflow_specification is None
        if workflow_specification is None:
            if api_key is None:
                raise MissingApiKeyError(MISSING_API_KEY_MESSAGE)
            with profiler.profile_execution_phase(
                name="workflow_definition_fetching",
                categories=["inference_package_operation"],
            ):
                workflow_specification = workflows_host.get_workflow_specification(
                    api_key=api_key,
                    workspace_id=workspace_name,
                    workflow_id=workflow_id,
                    workflow_version_id=workflow_version_id,
                    use_cache=True,
                )

        bridge, loop_bridge, collector = self._ensure_started()
        observer, holders_scope = NULL_EXECUTION_OBSERVER, nullcontext()
        if collector is not None:
            observer = StreamUsageExecutionObserver(
                collector,
                workflow_id=workflow_id if named_from_registry else None,
                specification=workflow_specification,
            )
            holders_scope = observer.holders_scope()
        with holders_scope:
            sync_bridge = SyncLegacyBridge(bridge, loop_bridge)
        provider = GatewayModelsProvider(sync_bridge, api_key)
        init_parameters = execution.build_init_parameters(
            provider=provider,
            api_key=api_key,
            background_tasks=None,
            disable_sinks=False,
            inner_workflow_dispatch_depth=0,
            execution_observer=observer,
        )

        return (
            workflow_specification,
            init_parameters,
            workflows_host.step_error_handler,
        )

    def close(self) -> None:
        """Flush and stop the collector, shut the gateway down, stop the loop thread.

        Idempotent, never raises. The whole call is bounded by
        `CLOSE_TIMEOUT_S`; the collector goes first so the rows of the last
        runs are sent while the gateway is still up. When a start holds the
        host, the host is only marked closed and the starter tears its own
        gateway down once its start returns.
        """
        deadline = time.monotonic() + CLOSE_TIMEOUT_S
        self._closed = True
        if not self._lock.acquire(timeout=CLOSE_TIMEOUT_S):
            logger.warning(
                "The pipeline host is still starting; its starter will tear it down."
            )
            return None
        try:
            loop, thread, gateway = self._loop, self._thread, self._gateway
            collector = self._collector
            self._loop, self._thread, self._gateway = None, None, None
            self._bridge, self._loop_bridge, self._collector = None, None, None
        finally:
            self._lock.release()
        if collector is not None:
            _stop_collector(collector, deadline, flush=True)
        if loop is None:
            return None

        _stop_loop(loop, thread, gateway, deadline)

        return None

    def _ensure_started(self) -> Tuple[LegacyModelBridge, LoopBridge, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("The pipeline host is closed.")
            if self._loop is None:
                self._start()

            return self._bridge, self._loop_bridge, self._collector

    def _start(self) -> None:
        os.environ.setdefault(configuration.INFERENCE_GATEWAY_ENV, self._gateway_kind)
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, name="pipeline-host-loop", daemon=True
        )
        thread.start()
        gateway = None
        collector = None
        try:
            gateway = gateway_resolver.resolve_gateway()
            asyncio.run_coroutine_threadsafe(gateway.start(), loop).result()
            collector = _build_usage_collector()
            if collector is not None:
                collector.start()
            loop_bridge = LoopBridge(loop)
            workflows_host.GUARDED_IMAGE_CODEC.bind_loop(loop_bridge)
            bridge = LegacyModelBridge(gateway)
            if self._closed:
                raise RuntimeError("The pipeline host is closed.")
        except BaseException:
            deadline = time.monotonic() + CLOSE_TIMEOUT_S
            if collector is not None:
                _stop_collector(collector, deadline, flush=False)
            _stop_loop(loop, thread, gateway, deadline)
            raise

        self._loop, self._thread, self._gateway = loop, thread, gateway
        self._bridge, self._loop_bridge = bridge, loop_bridge
        self._collector = collector


def _build_usage_collector() -> Any:
    usage_collector = usage_collector_module.UsageCollector()

    return usage_collector


def _stop_collector(collector: Any, deadline: float, *, flush: bool) -> None:
    def _flush_and_stop() -> None:
        try:
            if flush:
                collector.flush()
            collector.stop(max(deadline - time.monotonic(), 0.0))
        except BaseException as error:
            logger.warning(
                f"Could not stop the pipeline usage collector. Error: {error!r}"
            )

    worker = threading.Thread(
        target=_flush_and_stop, name="pipeline-host-usage-stop", daemon=True
    )
    worker.start()
    worker.join(max(deadline - time.monotonic(), 0.0))
    if worker.is_alive():
        logger.warning("The pipeline usage collector did not stop in time.")

    return None


def _stop_loop(
    loop: asyncio.AbstractEventLoop,
    thread: threading.Thread,
    gateway: Any,
    deadline: float,
) -> None:
    if gateway is not None:
        try:
            asyncio.run_coroutine_threadsafe(gateway.shutdown(), loop).result(
                max(deadline - time.monotonic(), 0.0)
            )
        except BaseException as error:
            logger.warning(
                f"Could not shut the pipeline gateway down. Error: {error!r}"
            )
    loop.call_soon_threadsafe(loop.stop)
    thread.join(max(deadline - time.monotonic(), 0.0))
    if thread.is_alive():
        logger.warning("The pipeline host loop thread did not stop in time.")
        return None

    loop.close()

    return None


SERVER_PIPELINE_HOST_DESCRIPTOR = PipelineHostDescriptor(
    factory="inference_server.streams.host:ServerPipelineHost",
    settings={
        "gateway_kind": os.environ.get(
            configuration.INFERENCE_GATEWAY_ENV, configuration.INFERENCE_GATEWAY_DEFAULT
        )
    },
)
