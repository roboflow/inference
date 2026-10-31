"""Shared fixture builders for vLLM proxy unit tests.

Builds tiny synthetic adapter / base-model packages programmatically (no
network, no real model weights).
"""

import json
import os
from typing import Dict, List, Optional

import torch
from safetensors.torch import save_file

OUT_FEATURES = 16
IN_FEATURES = 12
LORA_RANK = 4
LORA_ALPHA = 8

LANGUAGE_MODULE_PATH = "base_model.model.model.layers.0.self_attn.q_proj"
VISION_MODULE_PATH = "base_model.model.model.visual.blocks.0.attn.qkv"
BASE_WEIGHT_KEY = "model.language_model.layers.0.self_attn.q_proj.weight"


def build_adapter_config(
    r: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    use_dora: bool = False,
    use_rslora: bool = False,
    target_modules: Optional[List[str]] = None,
    modules_to_save: Optional[List[str]] = None,
    base_model_name_or_path: Optional[str] = None,
) -> dict:
    config = {
        "peft_type": "LORA",
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "target_modules": (
            target_modules if target_modules is not None else ["q_proj", "v_proj"]
        ),
        "modules_to_save": modules_to_save,
        "use_dora": use_dora,
        "use_rslora": use_rslora,
        "bias": "none",
        "eva_config": None,
        "lora_bias": False,
        "exclude_modules": None,
    }
    if base_model_name_or_path is not None:
        config["base_model_name_or_path"] = base_model_name_or_path
    return config


def build_adapter_tensors(
    r: int = LORA_RANK,
    use_dora: bool = False,
    include_vision: bool = False,
    vision_lora_b_nonzero: bool = False,
    seed: int = 7,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        f"{LANGUAGE_MODULE_PATH}.lora_A.weight": torch.randn(
            (r, IN_FEATURES), generator=generator
        ),
        f"{LANGUAGE_MODULE_PATH}.lora_B.weight": torch.randn(
            (OUT_FEATURES, r), generator=generator
        ),
    }
    if use_dora:
        tensors[f"{LANGUAGE_MODULE_PATH}.lora_magnitude_vector.weight"] = (
            torch.rand((OUT_FEATURES,), generator=generator) + 0.5
        )
    if include_vision:
        tensors[f"{VISION_MODULE_PATH}.lora_A.weight"] = torch.randn(
            (r, IN_FEATURES), generator=generator
        )
        if vision_lora_b_nonzero:
            tensors[f"{VISION_MODULE_PATH}.lora_B.weight"] = torch.randn(
                (OUT_FEATURES, r), generator=generator
            )
        else:
            tensors[f"{VISION_MODULE_PATH}.lora_B.weight"] = torch.zeros(
                (OUT_FEATURES, r)
            )
    return tensors


def write_adapter_package(
    target_dir: str,
    config: Optional[dict] = None,
    tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> str:
    os.makedirs(target_dir, exist_ok=True)
    if config is None:
        config = build_adapter_config()
    if tensors is None:
        tensors = build_adapter_tensors(use_dora=bool(config.get("use_dora")))
    with open(os.path.join(target_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f)
    save_file(tensors, os.path.join(target_dir, "adapter_model.safetensors"))
    return target_dir


def write_base_package(
    target_dir: str, base_weight: Optional[torch.Tensor] = None, seed: int = 11
) -> torch.Tensor:
    os.makedirs(target_dir, exist_ok=True)
    if base_weight is None:
        generator = torch.Generator().manual_seed(seed)
        base_weight = torch.randn((OUT_FEATURES, IN_FEATURES), generator=generator)
    save_file(
        {BASE_WEIGHT_KEY: base_weight},
        os.path.join(target_dir, "model.safetensors"),
    )
    return base_weight
