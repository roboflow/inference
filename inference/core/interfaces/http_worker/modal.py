from typing import Dict, Optional

from inference.core.env import (
    WEBRTC_MODAL_APP_NAME,
    WEBRTC_MODAL_FUNCTION_MAX_TIME_LIMIT,
    WEBRTC_MODAL_TOKEN_ID,
    WEBRTC_MODAL_TOKEN_SECRET,
)
from inference.core.exceptions import RoboflowAPIUnsuccessfulRequestError
from inference.core.interfaces.http_worker.entities import WorkerPayload
from inference.core.logger import logger
from inference.usage_tracking.collector import usage_collector


def spawn_http_worker_modal(payload) -> Optional[str]:
    import modal
    from inference.core.interfaces.webrtc_worker.modal import (
        RTCPeerConnectionModalCPU,
        RTCPeerConnectionModalGPU,
        app,
        docker_tag,
    )
    from inference.usage_tracking.plan_details import WebRTCPlan

    if not isinstance(payload, WorkerPayload):
        payload = WorkerPayload.model_validate(payload)

    requested_gpu: Optional[str] = None
    requested_ram_mb: Optional[int] = None
    requested_cpu_cores: Optional[int] = None
    webrtc_plans: Optional[Dict[str, WebRTCPlan]] = (
        usage_collector._plan_details.get_webrtc_plans(api_key=payload.api_key)
        if payload.api_key
        else None
    )
    if webrtc_plans and payload.requested_plan:
        if payload.requested_plan not in webrtc_plans:
            raise RoboflowAPIUnsuccessfulRequestError(
                f"Unknown requested plan {payload.requested_plan}, available plans: {', '.join(webrtc_plans.keys())}"
            )
        requested_gpu = webrtc_plans[payload.requested_plan].gpu
        requested_ram_mb = webrtc_plans[payload.requested_plan].ram_mb
        requested_cpu_cores = webrtc_plans[payload.requested_plan].cpu_cores

    client = modal.Client.from_credentials(
        token_id=WEBRTC_MODAL_TOKEN_ID,
        token_secret=WEBRTC_MODAL_TOKEN_SECRET,
    )
    try:
        modal.App.lookup(
            name=WEBRTC_MODAL_APP_NAME, client=client, create_if_missing=False
        )
    except modal.exception.NotFoundError:
        logger.info("Deploying webrtc modal app %s", WEBRTC_MODAL_APP_NAME)
        app.deploy(name=WEBRTC_MODAL_APP_NAME, client=client, tag=docker_tag)

    modal_cls = (
        RTCPeerConnectionModalGPU if requested_gpu else RTCPeerConnectionModalCPU
    )
    deployed_cls = modal.Cls.from_name(
        app_name=app.name,
        name=modal_cls.__name__,
    )
    deployed_cls.hydrate(client=client)
    timeout = payload.processing_timeout or WEBRTC_MODAL_FUNCTION_MAX_TIME_LIMIT
    cls_with_options = deployed_cls.with_options(timeout=timeout)
    if requested_gpu is not None:
        cls_with_options = cls_with_options.with_options(gpu=requested_gpu)
    if requested_ram_mb is not None:
        cls_with_options = cls_with_options.with_options(memory=requested_ram_mb)
    if requested_cpu_cores is not None:
        cls_with_options = cls_with_options.with_options(cpu=requested_cpu_cores)

    modal_obj = cls_with_options(preload_hf_ids="", preload_models="")
    function_call: modal.FunctionCall = modal_obj.http_worker_modal.spawn(
        payload=payload.model_dump(mode="json"),
    )
    call_id = getattr(function_call, "object_id", None) or getattr(
        function_call, "call_id", None
    )
    logger.info("Spawned HTTP worker Modal function")
    return str(call_id) if call_id else None
