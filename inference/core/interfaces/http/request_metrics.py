import json
import time
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware

from inference.core.constants import (
    MODEL_COLD_START_COUNT_HEADER,
    MODEL_COLD_START_HEADER,
    MODEL_ID_HEADER,
    MODEL_LOAD_DETAILS_HEADER,
    MODEL_LOAD_TIME_HEADER,
    PROCESSING_TIME_HEADER,
)
from inference.core.env import (
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
    WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING,
)

try:
    from inference_sdk.config import (
        EXECUTION_ID_HEADER,
        INTERNAL_REMOTE_EXEC_REQ_HEADER,
        INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER,
        RemoteProcessingTimeCollector,
        apply_duration_minimum,
        execution_id,
        remote_processing_times,
    )
except ImportError:
    execution_id = None
    remote_processing_times = None
    RemoteProcessingTimeCollector = None
    EXECUTION_ID_HEADER = None
    INTERNAL_REMOTE_EXEC_REQ_HEADER = None
    INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER = None
    apply_duration_minimum = None


REMOTE_PROCESSING_TIME_HEADER = "X-Remote-Processing-Time"
REMOTE_PROCESSING_TIMES_HEADER = "X-Remote-Processing-Times"


def summarize_model_load_entries(
    entries: List[Tuple[str, float]], max_detail_bytes: int = 4096
) -> Tuple[float, Optional[str]]:
    total = sum(load_time for _, load_time in entries)
    detail = json.dumps(
        [{"m": model_id, "t": load_time} for model_id, load_time in entries]
    )
    if len(detail) > max_detail_bytes:
        detail = None
    return total, detail


def build_model_response_headers(
    local_model_ids: set,
    local_cold_start_entries: List[Tuple[str, float]],
    remote_model_ids: set,
    remote_cold_start_entries: List[Tuple[str, float]],
    remote_cold_start_count: int,
    remote_cold_start_total_load_time: float,
) -> Dict[str, str]:
    response_headers = {
        MODEL_COLD_START_HEADER: "false",
        MODEL_COLD_START_COUNT_HEADER: "0",
    }
    model_ids = sorted(local_model_ids | remote_model_ids)
    if model_ids:
        response_headers[MODEL_ID_HEADER] = ",".join(model_ids)
    local_cold_start_count = len(local_cold_start_entries)
    cold_start_count = local_cold_start_count + remote_cold_start_count
    response_headers[MODEL_COLD_START_COUNT_HEADER] = str(cold_start_count)
    if cold_start_count == 0:
        return response_headers
    response_headers[MODEL_COLD_START_HEADER] = "true"
    local_load_time = sum(load_time for _, load_time in local_cold_start_entries)
    response_headers[MODEL_LOAD_TIME_HEADER] = str(
        local_load_time + remote_cold_start_total_load_time
    )
    detailed_entries = local_cold_start_entries + remote_cold_start_entries
    if len(detailed_entries) != cold_start_count:
        return response_headers
    _, detail = summarize_model_load_entries(entries=detailed_entries)
    if detail is not None:
        response_headers[MODEL_LOAD_DETAILS_HEADER] = detail
    return response_headers


class GCPServerlessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if execution_id is not None:
            execution_id_value = request.headers.get(EXECUTION_ID_HEADER)
            if not execution_id_value:
                execution_id_value = f"{time.time_ns()}_{uuid4().hex[:4]}"
            execution_id.set(execution_id_value)
        is_verified_internal = False
        if apply_duration_minimum is not None:
            is_verified_internal = bool(
                ROBOFLOW_INTERNAL_SERVICE_SECRET
                and INTERNAL_REMOTE_EXEC_REQ_HEADER
                and request.headers.get(INTERNAL_REMOTE_EXEC_REQ_HEADER)
                == ROBOFLOW_INTERNAL_SERVICE_SECRET
            )
            apply_duration_minimum.set(not is_verified_internal)
        collector = None
        if (
            remote_processing_times is not None
            and RemoteProcessingTimeCollector is not None
        ):
            collector = RemoteProcessingTimeCollector()
            request.state.remote_processing_time_collector = collector
            remote_processing_times.set(collector)
        t1 = time.time()
        response = await call_next(request)
        t2 = time.time()
        response.headers[PROCESSING_TIME_HEADER] = str(t2 - t1)
        if (
            WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING
            and collector is not None
            and collector.has_data()
        ):
            total, detail = collector.snapshot_summary()
            response.headers[REMOTE_PROCESSING_TIME_HEADER] = str(total)
            if detail is not None:
                response.headers[REMOTE_PROCESSING_TIMES_HEADER] = detail
        if execution_id is not None:
            response.headers[EXECUTION_ID_HEADER] = execution_id_value
        if INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER is not None:
            response.headers[INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER] = str(
                is_verified_internal
            ).lower()
        return response
