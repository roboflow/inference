"""Tests for the VLLM_PROXY_ENABLED registration switch in inference.models.utils.

`inference.models.utils` resolves the switch at import time, so these tests
(re)import the module with the env var set. The module is removed from
sys.modules afterwards so later imports re-evaluate against the then-current
environment (importing with the proxy DISABLED requires the full HF qwen3_5
stack, which CPU test environments may not ship - that path is therefore not
exercised here).
"""

import importlib
import os
import sys

import pytest

from inference.models.vllm_proxy.qwen3_5_vllm import Qwen35VLLMProxy
from inference.models.vllm_proxy.qwen3vl_vllm import Qwen3VLVLLMProxy

UTILS_MODULE = "inference.models.utils"


def _reload_vllm_proxy_switch() -> None:
    """Re-reads VLLM_PROXY_ENABLED (resolved at import time) from the env."""
    import inference.models.vllm_proxy as vllm_proxy_package
    import inference.models.vllm_proxy.config as config_module

    importlib.reload(config_module)
    importlib.reload(vllm_proxy_package)


@pytest.fixture
def model_types_with_proxy_enabled():
    previous_value = os.environ.get("VLLM_PROXY_ENABLED")
    os.environ["VLLM_PROXY_ENABLED"] = "true"
    previous_utils = sys.modules.pop(UTILS_MODULE, None)
    try:
        _reload_vllm_proxy_switch()
        utils_module = importlib.import_module(UTILS_MODULE)
        yield utils_module.ROBOFLOW_MODEL_TYPES
    finally:
        if previous_value is None:
            os.environ.pop("VLLM_PROXY_ENABLED", None)
        else:
            os.environ["VLLM_PROXY_ENABLED"] = previous_value
        _reload_vllm_proxy_switch()
        # Drop the proxy-enabled registry; the next importer re-evaluates the
        # module against the restored environment.
        sys.modules.pop(UTILS_MODULE, None)
        if previous_utils is not None:
            sys.modules[UTILS_MODULE] = previous_utils


class TestQwen3VLRegistrationSwitch:
    @pytest.mark.parametrize(
        "task_and_variant",
        [
            ("text-image-pairs", "qwen3vl-2b-instruct"),
            ("text-image-pairs", "qwen3vl-2b-instruct-peft"),
        ],
    )
    def test_explicit_qwen3vl_ids_resolve_to_proxy(
        self, model_types_with_proxy_enabled, task_and_variant
    ) -> None:
        assert model_types_with_proxy_enabled[task_and_variant] is Qwen3VLVLLMProxy

    def test_vlm_qwen3vl_alias_resolves_to_proxy(
        self, model_types_with_proxy_enabled
    ) -> None:
        from inference.core.env import USE_INFERENCE_MODELS

        if not USE_INFERENCE_MODELS:
            pytest.skip("variant-startswith branch requires USE_INFERENCE_MODELS")
        assert model_types_with_proxy_enabled[("vlm", "qwen3vl")] is Qwen3VLVLLMProxy


class TestQwen35RegistrationSwitch:
    @pytest.mark.parametrize(
        "task_and_variant",
        [
            ("lmm", "qwen3_5-0.8b"),
            ("lmm", "qwen3_5-0.8b-peft"),
            ("text-image-pairs", "qwen3_5-0.8b"),
            ("vlm", "qwen_3_5"),
        ],
    )
    def test_qwen3_5_ids_resolve_to_proxy(
        self, model_types_with_proxy_enabled, task_and_variant
    ) -> None:
        assert model_types_with_proxy_enabled[task_and_variant] is Qwen35VLLMProxy
