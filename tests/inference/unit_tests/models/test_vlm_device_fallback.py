import importlib
import sys
from types import ModuleType

import pytest
import torch

import inference.core.env

VLM_MODULES = [
    "inference.models.paligemma.paligemma",
    "inference.models.florence2.florence2",
    "inference.models.qwen25vl.qwen25vl",
    "inference.models.qwen3vl.qwen3vl",
]


@pytest.fixture
def reload_with_unset_device(monkeypatch):
    # Import first, with the real CUDA probe: a first import of the model stack
    # while torch.cuda.is_available() lies would load CUDA-only code paths.
    for module_name in VLM_MODULES:
        importlib.import_module(module_name)
    monkeypatch.setattr(inference.core.env, "DEVICE", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    reloaded = []

    def _reload(module_name: str) -> ModuleType:
        module = importlib.reload(importlib.import_module(module_name))
        reloaded.append(module)
        return module

    yield _reload
    monkeypatch.undo()
    for module in reloaded:
        importlib.reload(module)


@pytest.mark.parametrize("module_name", VLM_MODULES)
def test_vlm_module_falls_back_to_the_gpu_when_device_is_unset(
    reload_with_unset_device, module_name
) -> None:
    # when
    module = reload_with_unset_device(module_name)

    # then
    assert module.DEVICE == "cuda:0"


def test_paligemma_ampere_check_gets_a_torch_device(
    reload_with_unset_device, monkeypatch
) -> None:
    # given
    paligemma = reload_with_unset_device("inference.models.paligemma.paligemma")
    monkeypatch.setattr(paligemma, "is_flash_attn_2_available", lambda: True)
    monkeypatch.setitem(sys.modules, "flash_attn", ModuleType("flash_attn"))
    seen = []

    def _capability(device):
        seen.append(device)
        return (8, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", _capability)

    # when
    implementation = paligemma._get_paligemma_attn_implementation()

    # then
    assert implementation == "flash_attention_2"
    assert seen == [torch.device("cuda:0")]
