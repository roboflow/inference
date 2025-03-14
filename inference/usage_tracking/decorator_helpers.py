from typing import Any, Dict, List, Optional

from inference.core.logger import logger
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    CompiledWorkflow,
)


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
    return None


def get_model_resource_details_from_kwargs(
    func_kwargs: Dict[str, Any]
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
    return resource_details


def get_resource_details_from_workflow_json(
    workflow_json: Dict[str, Any],
) -> List[str]:
    return [
        f"{step.get('type', 'unknown')}:{step.get('name', 'unknown')}"
        for step in workflow_json.get("steps", [])
        if isinstance(step, dict)
    ]


def get_workflow_resource_details_from_kwargs(
    func_kwargs: Dict[str, Any]
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
    return None


def get_request_resource_details_from_kwargs(
    func_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    if "workflow_request" in func_kwargs:
        workflow_request = func_kwargs["workflow_request"]
        if hasattr(workflow_request, "specification") and isinstance(
            workflow_request.specification, dict
        ):
            return {
                "steps": get_resource_details_from_workflow_json(
                    workflow_json=workflow_request.specification,
                )
            }
    return {}
