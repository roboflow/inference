import importlib
import sys

import pytest

PACKAGE = "inference_models.models.vllm_proxy"
MODULES = [
    PACKAGE,
    f"{PACKAGE}.qwen3vl_vllm",
    f"{PACKAGE}.qwen3_5_vllm",
    f"{PACKAGE}.qwen3_8_vllm",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_vllm_proxy_imports_without_transformers(monkeypatch, module_name):
    for name in [
        loaded
        for loaded in sys.modules
        if loaded == PACKAGE or loaded.startswith(f"{PACKAGE}.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "transformers", None)

    module = importlib.import_module(module_name)

    assert module.__name__ == module_name
