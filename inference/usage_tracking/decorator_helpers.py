from typing import Any, Dict, List, Optional, Tuple

from inference.core.env import SAM3_EXEC_MODE
from inference.core.logger import logger
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    CompiledWorkflow,
)
from inference.usage_tracking.megapixel_buckets import (
    build_megapixel_buckets,
    consume_measured_model_input,
    count_inference_images,
    resolve_model_input_hw,
)
from inference.usage_tracking.model_types import get_recorded_model_type


def get_model_id_from_kwargs(func_kwargs: Dict[str, Any]) -> Optional[str]:
    if "self" in func_kwargs:
        _self = func_kwargs["self"]
        if hasattr(_self, "dataset_id") and hasattr(_self, "version_id"):
            model_id = str(_self.dataset_id)
            if _self.version_id:
                model_id += f"/{_self.version_id}"
            return model_id
    if "model_id" in func_kwargs:
        return func_kwargs["model_id"]
    if "kwargs" in func_kwargs and "model_id" in func_kwargs["kwargs"]:
        return func_kwargs["kwargs"]["model_id"]
    for request_key in ("inference_request", "request", "workflow_request"):
        request = func_kwargs.get(request_key)
        model_id = getattr(request, "model_id", None)
        if model_id:
            return str(model_id)
    if "self" in func_kwargs:
        model_id = getattr(func_kwargs["self"], "model_id", None)
        if model_id:
            return str(model_id)
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
    """Resolve Roboflow ``modelType`` (architecture / size), not the resource id.

    Only values already known in-process are consulted. Asking the model registry
    would be a network call on the inference hot path, so model types are instead
    recorded when the registry resolves them during model loading. A model whose
    type was never recorded is reported without one.
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
    if "source" in func_kwargs:
        resource_details["source"] = func_kwargs["source"]
    elif "kwargs" in func_kwargs and "source" in func_kwargs["kwargs"]:
        resource_details["source"] = func_kwargs["kwargs"]["source"]
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
    if source_info and source_info != "external":
        return source_info
    return None


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
    if func_kwargs.get("countinference") is not None:
        resource_details["billable"] = func_kwargs["countinference"]
    model_id = getattr(func_kwargs.get("inference_request"), "model_id", None)
    if isinstance(model_id, str) and model_id.startswith("sam3/"):
        resource_details["execution_mode"] = SAM3_EXEC_MODE
    return resource_details
