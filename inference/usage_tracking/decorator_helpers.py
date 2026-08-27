from contextvars import ContextVar, Token
from typing import Any, Callable, Dict, List, Optional, Tuple

from inference.core.env import SAM3_EXEC_MODE
from inference.core.logger import logger
from inference.core.roboflow_api import service_secret_is_valid
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    CompiledWorkflow,
)
from inference.usage_tracking.megapixel_buckets import (
    build_megapixel_buckets,
    clear_measured_model_input,
    consume_measured_model_input,
    count_inference_images,
    record_measured_model_hw,
    resolve_model_input_hw,
)
from inference.usage_tracking.model_types import get_recorded_model_type
from inference.usage_tracking.utils import (
    coerce_optional_bool,
    collect_func_params,
    get_signature,
)

# Whether usage rows recorded in the current execution context must be marked
# non-billable. Bound by the `usage_collector` wrappers from the arguments the
# decorated call was made with, and inherited by everything that runs inside
# them: nested decorators, and Execution Engine steps, whose thread pool
# re-enters the caller's context in every worker.
usage_billing_suppressed: ContextVar[bool] = ContextVar(
    "usage_billing_suppressed", default=False
)

EXTERNAL_SOURCE_SENTINEL = "external"

# Source tags (`source` / `source_info`) the current request-level call
# carried, resolved once by the `usage_collector` wrappers and inherited the
# same way as billing suppression - so a nested model decorator, which is
# handed only the typed request, can attribute its row to the caller. The
# default is never mutated, only replaced by `set()`.
usage_source_tags: ContextVar[Dict[str, str]] = ContextVar(
    "usage_source_tags", default={}
)


def _meaningful_source(value: Any) -> Optional[str]:
    """A source tag worth recording, or None.

    ``"external"`` is the placeholder the HTTP layer fills in when the caller
    said nothing, so it identifies no one and is dropped rather than recorded as
    a bucket of its own.
    """
    if not isinstance(value, str) or not value:
        return None
    if value == EXTERNAL_SOURCE_SENTINEL:
        return None

    return value


def _lookup_in_func_kwargs(func_kwargs: Dict[str, Any], key: str) -> Any:
    """Read a named parameter that may have landed in a catch-all ``**kwargs``."""
    if func_kwargs.get(key) is not None:
        return func_kwargs[key]
    nested_kwargs = func_kwargs.get("kwargs")
    if isinstance(nested_kwargs, dict):
        return nested_kwargs.get(key)

    return None


def _source_tag_bound_to_handler(
    func_kwargs: Dict[str, Any],
    key: str,
) -> Optional[str]:
    """Resolve a source tag from a handler's arguments, however it was declared.

    Handlers spell these three ways. The legacy route declares them plainly. Two
    SAM3 routes declare them under ``request_``-prefixed names, deliberately, so
    that the raw names stay out of ``func_kwargs`` where ``source_info`` would
    displace ``roboflow_service_name``. Every other route declares nothing at
    all, leaving the value reachable only through the request's query string.
    """
    for candidate in (func_kwargs.get(f"request_{key}"), func_kwargs.get(key)):
        tag = _meaningful_source(candidate)
        if tag is not None:
            return tag
    query_params = getattr(func_kwargs.get("request"), "query_params", None)
    if query_params is None:
        return None

    return _meaningful_source(query_params.get(key))


def _source_tag_on_bound_requests(
    func_kwargs: Dict[str, Any],
    key: str,
) -> Optional[str]:
    """Read a source tag persisted on a bound request payload."""
    for request_key in ("inference_request", "request", "workflow_request"):
        tag = _meaningful_source(getattr(func_kwargs.get(request_key), key, None))
        if tag is not None:
            return tag

    return None


def read_source_tags_bound_to_call(
    func: Callable[..., Any],
    args: Any,
    kwargs: Dict[str, Any],
) -> Dict[str, str]:
    """Source tags the decorated call carries, however the handler declares them.

    Gated the same way as the billing intent: only request-level handlers
    declare ``countinference``, so the model hot path never binds its call.
    A handler's explicit declaration wins over what the request payload already
    carries, matching how the tags used to be stamped onto the payload.

    Usage tracking must never break inference, so binding failures are swallowed
    the same way the surrounding recording calls swallow theirs.
    """
    try:
        if "countinference" not in get_signature(func).parameters:
            return {}
        func_kwargs = collect_func_params(func, args, kwargs)
        tags = {}
        for key in ("source", "source_info"):
            tag = _source_tag_bound_to_handler(
                func_kwargs, key
            ) or _source_tag_on_bound_requests(func_kwargs, key)
            if tag is not None:
                tags[key] = tag
        return tags
    except Exception as exc:
        logger.debug("Failed to read source tags from call - %s", exc)
        return {}


def non_billable_intent_is_authenticated(
    countinference: Any,
    service_secret: Any,
) -> bool:
    """Whether a caller both asked to skip billing and proved it may.

    ``countinference=false`` is an internal-services affordance, not a public
    one. It is honoured only alongside a valid service secret, matching the gate
    applied in ``roboflow_api`` before contacting the platform and in the
    serverless authorization middleware. An unauthenticated request to skip
    billing is ignored rather than rejected, so this stays a telemetry decision
    and never turns an otherwise valid inference into an error.

    Args:
        countinference: Raw per-request flag, as a bool or a string.
        service_secret: Shared secret supplied alongside the flag.

    Returns:
        True when billing should be suppressed for this call.
    """
    if coerce_optional_bool(countinference) is not False:
        return False
    if not service_secret_is_valid(service_secret):
        logger.debug("Ignoring countinference=false - service secret is not valid")
        return False

    return True


def call_carries_authenticated_non_billable_intent(
    func: Callable[..., Any],
    args: Any,
    kwargs: Dict[str, Any],
) -> bool:
    """Whether the call bound to a usage decorator opted out of billing, provably.

    The HTTP handler owns the query string; everything nested beneath it - the
    workflow, the models it runs - is handed nothing about billing at all. So
    the intent is read once, here, from the arguments the decorated call was
    actually made with, and published as context for the rest of the call.

    Handlers that cannot carry the intent are answered from the cached signature
    rather than by binding the call: the model hot path goes through the same
    decorator.

    Usage tracking must never break inference, so binding failures are swallowed
    the same way the surrounding recording calls swallow theirs.

    Args:
        func: The decorated function.
        args: Positional arguments it was called with.
        kwargs: Keyword arguments it was called with.

    Returns:
        True when the call carries an authenticated ``countinference=false``.
    """
    try:
        if "countinference" not in get_signature(func).parameters:
            return False
        func_kwargs = collect_func_params(func, args, kwargs)
        return non_billable_intent_is_authenticated(
            func_kwargs.get("countinference"),
            func_kwargs.get("service_secret"),
        )
    except Exception as exc:
        logger.debug("Failed to read billing intent from call - %s", exc)
        return False


def bind_billing_suppression(
    authenticated_opt_out: bool,
    usage_billable: bool,
) -> Optional[Token[bool]]:
    """Suppress billing for the current call, unless it already is suppressed.

    Suppression is downgrade-only: an inherited one is left alone, so nothing
    nested can restore billing for a caller who opted out - and nothing has to
    be reset that this call did not set.

    Args:
        authenticated_opt_out: Whether the call proved an intent to skip billing.
        usage_billable: The decorator argument the call was made with.

    Returns:
        The token to reset once the call is recorded, or None when this call
        bound nothing.
    """
    if usage_billing_suppressed.get():
        return None
    if usage_billable and not authenticated_opt_out:
        return None

    return usage_billing_suppressed.set(True)


def _non_empty_model_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    model_id = str(value).strip()
    if not model_id:
        return None
    return model_id


def get_model_id_from_kwargs(func_kwargs: Dict[str, Any]) -> Optional[str]:
    """Resolve the usage ``resource_id``.

    Caller-supplied ids (request / kwargs) beat ``self.model_id``. Some classes
    store a de-aliased or rewritten id on the instance (vLLM Qwen, TrOCR);
    promoting that field would split usage history at upgrade.
    """
    if "self" in func_kwargs:
        _self = func_kwargs["self"]
        dataset_id = getattr(_self, "dataset_id", None)
        if dataset_id:
            model_id = str(dataset_id)
            version_id = getattr(_self, "version_id", None)
            if version_id:
                model_id += f"/{version_id}"
            return model_id
    model_id = _non_empty_model_id(func_kwargs.get("model_id"))
    if model_id:
        return model_id
    nested_kwargs = func_kwargs.get("kwargs")
    if isinstance(nested_kwargs, dict):
        model_id = _non_empty_model_id(nested_kwargs.get("model_id"))
        if model_id:
            return model_id
    for request_key in ("inference_request", "request", "workflow_request"):
        request = func_kwargs.get(request_key)
        model_id = _non_empty_model_id(getattr(request, "model_id", None))
        if model_id:
            return model_id
    if "self" in func_kwargs:
        return _non_empty_model_id(getattr(func_kwargs["self"], "model_id", None))
    return None


def get_model_api_key_from_kwargs(func_kwargs: Dict[str, Any]) -> Optional[str]:
    api_key = func_kwargs.get("api_key")
    if api_key:
        return api_key
    nested_kwargs = func_kwargs.get("kwargs")
    if isinstance(nested_kwargs, dict) and nested_kwargs.get("api_key"):
        return nested_kwargs["api_key"]
    for request_key in ("inference_request", "request", "workflow_request"):
        request = func_kwargs.get(request_key)
        api_key = getattr(request, "api_key", None)
        if api_key:
            return api_key
    return None


def get_model_type_from_kwargs(func_kwargs: Dict[str, Any]) -> Optional[str]:
    """Resolve Roboflow model type (variant when known, else architecture).

    Prefer ``self.model_type`` (bound at load to the platform variant when
    known, otherwise the architecture). Fall back to the process-local map
    keyed by model id. Asking the model registry would be a network call on
    the inference hot path. A model whose type was never recorded is reported
    without one.
    """
    model = func_kwargs.get("self")
    if model is not None:
        model_type = getattr(model, "model_type", None)
        if model_type:
            return str(model_type)
    return get_recorded_model_type(get_model_id_from_kwargs(func_kwargs))


def get_model_resource_details_from_kwargs(
    func_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    resource_details = {}
    # A model decorator nested under an HTTP handler never sees the query
    # string, so a tag that arrived there reaches it only as request context.
    source = (
        _meaningful_source(_lookup_in_func_kwargs(func_kwargs, "source"))
        or _source_tag_on_bound_requests(func_kwargs, "source")
        or usage_source_tags.get().get("source")
    )
    if source is not None:
        resource_details["source"] = source
    if "self" in func_kwargs:
        _self = func_kwargs["self"]
        if hasattr(_self, "task_type"):
            resource_details["task_type"] = _self.task_type
    model_type = get_model_type_from_kwargs(func_kwargs)
    if model_type:
        resource_details["model_type"] = model_type
    return resource_details


def get_model_image_from_kwargs(func_kwargs: Dict[str, Any]) -> Any:
    if "image" in func_kwargs:
        return func_kwargs["image"]
    nested_kwargs = func_kwargs.get("kwargs")
    if isinstance(nested_kwargs, dict) and "image" in nested_kwargs:
        return nested_kwargs["image"]
    for request_key in ("inference_request", "request"):
        request = func_kwargs.get(request_key)
        image = getattr(request, "image", None)
        if image is not None:
            return image
    return None


def get_model_frames_and_input_hw(
    func_kwargs: Dict[str, Any],
) -> Tuple[int, Optional[Tuple[int, int]]]:
    """Frames recorded for one model call, and the input resolution to attribute
    them to.

    The frame count comes from the request rather than from the preprocessed
    tensor: preprocessing pads the batch dimension up to a fixed model batch size
    when ``FIX_BATCH_SIZE`` is set, and that padding is not part of what the
    caller asked for. The tensor batch size is only a fallback for calls whose
    images are not introspectable.
    """
    model = func_kwargs.get("self")
    measured_hw, measured_frames = consume_measured_model_input()

    frames = count_inference_images(get_model_image_from_kwargs(func_kwargs))
    if frames <= 0 and measured_frames:
        frames = measured_frames
    if frames <= 0:
        frames = 1

    return frames, resolve_model_input_hw(model, measured_hw=measured_hw)


def get_model_megapixel_buckets(
    *,
    frames: int,
    input_hw: Optional[Tuple[int, int]],
    execution_duration: float,
    inference_test_run: bool = False,
) -> Dict[str, Dict[str, Any]]:
    if inference_test_run:
        return {}
    height, width = input_hw if input_hw else (None, None)
    return build_megapixel_buckets(
        height=height,
        width=width,
        frames=frames,
        execution_duration=execution_duration,
    )


def record_fixed_model_input_for_request(model: Any, request: Any = None) -> None:
    """Publish a model's fixed input size for usage telemetry on a request call.

    Use this for entrypoints that decorate ``infer_from_request`` (rather than
    ``BaseInference.infer``), so they never hit the preprocess hook that normally
    records measured tensor size. The model's configured/fixed size is preferred
    over native upload resolution.
    """
    clear_measured_model_input()
    input_hw = resolve_model_input_hw(model)
    if input_hw is None:
        return
    record_measured_model_hw(
        height=input_hw[0],
        width=input_hw[1],
        frames=count_inference_images(getattr(request, "image", None)),
    )


def get_source_info_from_kwargs(func_kwargs: Dict[str, Any]) -> Optional[str]:
    # source_info can arrive as a direct kwarg (request-category HTTP handlers),
    # nested under the catch-all kwargs of a model's infer(self, image, **kwargs),
    # or as an attribute of the request object passed to infer_from_request.
    source_info = None
    if "source_info" in func_kwargs:
        source_info = func_kwargs["source_info"]
    elif "kwargs" in func_kwargs and isinstance(func_kwargs["kwargs"], dict):
        source_info = func_kwargs["kwargs"].get("source_info")
    if not source_info:
        for request_key in ("inference_request", "request", "workflow_request"):
            request = func_kwargs.get(request_key)
            if request is not None and hasattr(request, "source_info"):
                source_info = request.source_info
                if source_info:
                    break
    return _meaningful_source(source_info) or usage_source_tags.get().get("source_info")


def get_resource_details_from_workflow_json(
    workflow_json: Dict[str, Any],
) -> List[str]:
    return [
        f"{step.get('type', 'unknown')}:{step.get('name', 'unknown')}"
        for step in workflow_json.get("steps", [])
        if isinstance(step, dict)
    ]


def get_workflow_resource_details_from_kwargs(
    func_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if "workflow" not in func_kwargs:
        return {}

    workflow: CompiledWorkflow = func_kwargs["workflow"]
    if not hasattr(workflow, "workflow_json"):
        return {}
    if not isinstance(workflow.workflow_json, dict):
        logger.debug("Got non-dict workflow JSON, '%s'", workflow.workflow_json)
        return {}

    return {
        "steps": get_resource_details_from_workflow_json(
            workflow_json=workflow.workflow_json,
        )
    }


def get_workflow_api_key_from_kwargs(func_kwargs: Dict[str, Any]) -> Optional[str]:
    if "workflow" not in func_kwargs:
        return None

    workflow: CompiledWorkflow = func_kwargs["workflow"]
    if not hasattr(workflow, "init_parameters"):
        return None
    if not isinstance(workflow.init_parameters, dict):
        logger.debug(
            "Got non-dict workflow init parameters, '%s'", workflow.init_parameters
        )
        return None

    return workflow.init_parameters.get("workflows_core.api_key")


def get_request_api_key_from_kwargs(func_kwargs: Dict[str, Any]) -> Optional[str]:
    if "inference_request" in func_kwargs:
        inference_request = func_kwargs["inference_request"]
        if hasattr(inference_request, "api_key"):
            return inference_request.api_key
    if "api_key" in func_kwargs:
        return func_kwargs["api_key"]
    if "workflow_request" in func_kwargs:
        workflow_request = func_kwargs["workflow_request"]
        if hasattr(workflow_request, "api_key"):
            return workflow_request.api_key
    return None


def get_request_resource_id_from_kwargs(func_kwargs: Dict[str, Any]) -> Optional[str]:
    if "inference_request" in func_kwargs:
        inference_request = func_kwargs["inference_request"]
        if hasattr(inference_request, "dataset_id") and hasattr(
            inference_request, "version_id"
        ):
            dataset_id = inference_request.dataset_id
            version_id = inference_request.version_id
            if version_id:
                return f"{dataset_id}/{version_id}"
            return str(dataset_id)
        if hasattr(inference_request, "model_id"):
            return str(inference_request.model_id)
    if "request" in func_kwargs:
        request = func_kwargs["request"]
        if hasattr(request, "model_id"):
            return str(request.model_id)
    if "dataset_id" in func_kwargs and "version_id" in func_kwargs:
        dataset_id = func_kwargs["dataset_id"]
        version_id = func_kwargs["version_id"]
        if version_id:
            return f"{dataset_id}/{version_id}"
        return str(dataset_id)
    if "workflow_id" in func_kwargs and func_kwargs["workflow_id"]:
        return str(func_kwargs["workflow_id"])
    if "workflow_request" in func_kwargs:
        workflow_request = func_kwargs["workflow_request"]
        if hasattr(workflow_request, "workflow_id"):
            return str(workflow_request.workflow_id)
    if "self" in func_kwargs:
        _self = func_kwargs["self"]
        if hasattr(_self, "dataset_id") and hasattr(_self, "version_id"):
            dataset_id = _self.dataset_id
            version_id = _self.version_id
            if version_id:
                return f"{dataset_id}/{version_id}"
            return str(dataset_id)
        if hasattr(_self, "model_id"):
            return str(_self.model_id)
        if hasattr(_self, "endpoint"):
            return str(_self.endpoint)
    return None


def get_request_resource_details_from_kwargs(
    func_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    resource_details = {}
    if "workflow_request" in func_kwargs:
        workflow_request = func_kwargs["workflow_request"]
        if hasattr(workflow_request, "specification") and isinstance(
            workflow_request.specification, dict
        ):
            resource_details["steps"] = get_resource_details_from_workflow_json(
                workflow_json=workflow_request.specification,
            )
    source = _source_tag_bound_to_handler(
        func_kwargs, "source"
    ) or _source_tag_on_bound_requests(func_kwargs, "source")
    if source is not None:
        resource_details["source"] = source
    model_id = getattr(func_kwargs.get("inference_request"), "model_id", None)
    if isinstance(model_id, str) and model_id.startswith("sam3/"):
        resource_details["execution_mode"] = SAM3_EXEC_MODE
    return resource_details
