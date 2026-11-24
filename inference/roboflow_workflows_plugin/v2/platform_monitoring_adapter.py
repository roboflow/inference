"""Inference-owned Model Monitoring transport and metadata.

Uses the existing synchronous host API, workspace cache, headers, timeout and
OFFLINE_MODE behavior. No new scheduler, cache or logger is introduced. The
audited ``inference`` and urllib3 emitters are protected inside each actual
host call, including metadata gathering, even when invoked on another thread.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List

from roboflow_workflows.v2.data.secrets import Secret, reveal_secrets

import inference.core.roboflow_api as roboflow_api
from inference.core.env import DEVICE_ID
from inference.core.managers.metrics import get_system_info
from inference.core.version import __version__
from inference.roboflow_workflows_plugin.v2.credential_boundary import (
    HostOperation,
    call_host,
)


class RoboflowModelMonitoringAdapter:
    """Host adapter for the standalone ModelMonitoringPort.

    Args:
        clock: Report timestamp clock, independent of the block's interval clock.
        system_info: Host metadata collector; V1's collector is the default.

    Stateless except for injected dependencies, which the caller owns and must
    make safe for concurrent use. Requires the explicit MONITORING operation in
    the shared credential boundary; older boundaries fail before any revelation.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = datetime.now,
        system_info: Callable[[], dict] = get_system_info,
    ) -> None:
        self._clock = clock
        self._system_info = system_info

    def get_workspace(self, *, api_key: Secret) -> str:
        """Resolve the key's workspace through the existing cached host API.

        Args:
            api_key: Roboflow credential, revealed only for host dispatch.

        Returns:
            Workspace URL slug.

        Raises:
            PlatformRequestError: When workspace lookup fails.
            SecretUnavailableError: When the key has no available value.
        """
        workspace = call_host(
            HostOperation.WORKSPACE,
            roboflow_api.get_roboflow_workspace,
            platform=True,
            api_key=reveal_secrets(api_key).value,
        )

        return workspace

    def send_report(
        self,
        *,
        api_key: Secret,
        workspace_id: str,
        predictions: List[Dict[str, Any]],
    ) -> None:
        """Build V1 report metadata and send through the real host API.

        Args:
            api_key: Credential revealed inside this synchronous operation.
            workspace_id: Workspace URL slug.
            predictions: Scalar representative prediction rows.

        Raises:
            PlatformRequestError: When metadata collection or upload fails.
            SecretUnavailableError: When the key has no available value.
            AttributeError: When the host lacks the required boundary extension.
        """
        operation = HostOperation.MONITORING
        call_host(
            operation,
            self._send_report,
            platform=True,
            api_key=reveal_secrets(api_key).value,
            workspace_id=workspace_id,
            predictions=predictions,
        )

    def _send_report(
        self,
        *,
        api_key: str,
        workspace_id: str,
        predictions: List[Dict[str, Any]],
    ) -> None:
        # The boundary surrounds the actual callback, rather than relying on
        # ContextVar inheritance from whichever thread scheduled this method.
        data = {
            "timestamp": self._clock().isoformat(),
            "source": "workflow",
            "source_info": "ModelMonitoringInferenceAggregatorBlockV1",
            "inference_results": [],
            "device_id": DEVICE_ID,
            "inference_server_version": __version__,
        }
        data.update(self._system_info() or {})
        data["inference_results"] = [dict(row) for row in predictions]
        roboflow_api.send_inference_results_to_model_monitoring(
            api_key=api_key,
            workspace_id=workspace_id,
            inference_data=data,
        )
