from functools import partial
from typing import Any, Dict, List, Optional

from openai import OpenAI

from inference.core.env import WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
from inference.core.roboflow_api import post_to_roboflow_api
from inference.core.workflows.core_steps.common.utils import run_in_parallel

ROBOFLOW_MANAGED_OPENROUTER_KEY_ALIASES = {"rf_key:account", "rf_key", "account"}


def is_roboflow_managed_openrouter_key(openrouter_api_key: str) -> bool:
    return openrouter_api_key.strip() in ROBOFLOW_MANAGED_OPENROUTER_KEY_ALIASES


def validate_openrouter_api_key(openrouter_api_key: str) -> str:
    key = openrouter_api_key.strip()
    if key.startswith("rf_key:user:"):
        raise ValueError(
            "User-stored OpenRouter API keys are not supported by the Roboflow "
            "OpenRouter proxy yet. Use 'rf_key:account' to bill Roboflow credits "
            "through the proxy, or pass a direct OpenRouter API key."
        )
    if key.startswith("rf_key:") and not is_roboflow_managed_openrouter_key(key):
        raise ValueError(
            "Unsupported Roboflow OpenRouter key reference. Use 'rf_key:account' "
            "to bill Roboflow credits through the proxy, or pass a direct "
            "OpenRouter API key."
        )
    return key


def execute_openrouter_requests(
    roboflow_api_key: Optional[str],
    openrouter_api_key: str,
    prompts: List[List[dict]],
    model_version_id: str,
    max_tokens: int,
    temperature: Optional[float],
    max_concurrent_requests: Optional[int],
    privacy_level: Optional[str] = None,
) -> List[str]:
    openrouter_api_key = validate_openrouter_api_key(
        openrouter_api_key=openrouter_api_key
    )
    if is_roboflow_managed_openrouter_key(openrouter_api_key=openrouter_api_key):
        request_executor = partial(
            execute_proxied_openrouter_request,
            roboflow_api_key=roboflow_api_key,
            openrouter_api_key=openrouter_api_key,
            model_version_id=model_version_id,
            privacy_level=privacy_level,
        )
    else:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
        request_executor = partial(
            execute_direct_openrouter_request,
            client=client,
            model_version_id=model_version_id,
            privacy_level=privacy_level,
        )
    tasks = [
        partial(
            request_executor,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for prompt in prompts
    ]
    max_workers = (
        max_concurrent_requests
        or WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS
    )
    return run_in_parallel(tasks=tasks, max_workers=max_workers)


def execute_proxied_openrouter_request(
    roboflow_api_key: Optional[str],
    openrouter_api_key: str,
    prompt: List[dict],
    model_version_id: str,
    max_tokens: int,
    temperature: Optional[float],
    privacy_level: Optional[str] = None,
) -> str:
    payload = {
        "openrouter_api_key": openrouter_api_key,
        "model": model_version_id,
        "messages": prompt,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if privacy_level is not None:
        payload["privacy_level"] = privacy_level
    response_data = post_to_roboflow_api(
        endpoint="apiproxy/openrouter",
        api_key=roboflow_api_key,
        payload=payload,
    )
    return extract_openrouter_response_content(
        response_data=response_data,
        provider_name="OpenRouter proxy",
    )


def execute_direct_openrouter_request(
    client: OpenAI,
    prompt: List[dict],
    model_version_id: str,
    max_tokens: int,
    temperature: Optional[float],
    privacy_level: Optional[str] = None,
) -> str:
    kwargs = {
        "model": model_version_id,
        "messages": prompt,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    provider = _build_provider_object(privacy_level=privacy_level)
    if provider is not None:
        kwargs["extra_body"] = {"provider": provider}
    response = client.chat.completions.create(**kwargs)
    if response.choices is None:
        error_detail = _get_openrouter_error_detail(response=response)
        raise RuntimeError(
            "OpenRouter provider failed in delivering response. This issue happens from time "
            "to time - raise issue to OpenRouter if that's problematic for you. "
            f"Details: {error_detail}"
        )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenRouter response missing message.content.")
    return content


def _build_provider_object(privacy_level: Optional[str]) -> Optional[Dict[str, Any]]:
    if privacy_level is None or privacy_level == "allow":
        return None
    provider = {"data_collection": "deny"}
    if privacy_level == "zdr":
        provider["zdr"] = True
    return provider


def extract_openrouter_response_content(
    response_data: Dict[str, Any],
    provider_name: str,
) -> str:
    choices = response_data.get("choices") or []
    if not choices:
        error_detail = _get_openrouter_response_error_detail(
            response_data=response_data
        )
        raise RuntimeError(
            f"{provider_name} returned no completion. Details: {error_detail}"
        )
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise RuntimeError(f"{provider_name} response missing message.content.")
    return content


def _get_openrouter_error_detail(response: Any) -> str:
    error_detail = getattr(response, "error", {}) or {}
    if isinstance(error_detail, dict):
        return error_detail.get("message", "N/A")
    return str(error_detail)


def _get_openrouter_response_error_detail(response_data: Dict[str, Any]) -> str:
    error_detail = response_data.get("error", {}) or {}
    if isinstance(error_detail, dict):
        return error_detail.get("message", "N/A")
    return str(error_detail)
