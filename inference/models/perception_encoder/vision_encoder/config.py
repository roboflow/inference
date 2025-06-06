# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Include all available vision encoder configurations.
"""

from dataclasses import dataclass, replace

from functools import partial
from typing import Callable, Optional, Sequence, Tuple, List

from huggingface_hub import hf_hub_download



def fetch_pe_checkpoint(name: str, path: Optional[str] = None):
    path = path or f"hf://facebook/{name}:{name}.pt"

    if path.startswith("hf://"):
        # Load from huggingface
        path = path[len("hf://"):]
        repo, file = path.split(":")

        # To count the download, config.yaml is empty
        hf_hub_download(repo_id=repo, filename="config.yaml")
        return hf_hub_download(repo_id=repo, filename=file)
    else:
        return path




@dataclass
class PEConfig:
    """ Vision Tower Config. """
    patch_size: int
    width: int
    layers: int
    heads: int
    mlp_ratio: float
    output_dim: Optional[int]

    ls_init_value: float = None
    drop_path: float = 0.0

    image_size: int = 224,
    use_abs_posemb: bool = True
    use_cls_token: bool = False
    use_rope2d: bool = True

    pool_type: str = "attn"
    attn_pooler_heads: int = 8

    use_ln_pre: bool = True
    use_ln_post: bool = True


@dataclass
class PETextConfig:
    """ Text Tower Config. """
    context_length: int
    width: int
    heads: int
    layers: int

    output_dim: int

    mlp_ratio: float = 4.0
    vocab_size: int = 49408




PE_VISION_CONFIG = {}
PE_TEXT_CONFIG = {}



#########################################
#                PE CORE                #
#########################################

PE_VISION_CONFIG["PE-Core-G14-448"] = PEConfig(
    image_size=448,
    patch_size=14,
    width=1536,
    layers=50,
    heads=16,
    mlp_ratio=8960 / 1536,
    pool_type="attn",
    output_dim=1280,
    use_cls_token=False,
)
PE_TEXT_CONFIG["PE-Core-G14-448"] = PETextConfig(
    context_length=72,
    width=1280,
    heads=20,
    layers=24,
    output_dim=1280
)


PE_VISION_CONFIG["PE-Core-L14-336"] = PEConfig(
    image_size=336,
    patch_size=14,
    width=1024,
    layers=24,
    heads=16,
    mlp_ratio=4.0,
    pool_type="attn",
    output_dim=1024,
    use_cls_token=True,
)
PE_TEXT_CONFIG["PE-Core-L14-336"] = PETextConfig(
    context_length=32,
    width=1024,
    heads=16,
    layers=24,
    output_dim=1024
)


PE_VISION_CONFIG["PE-Core-B16-224"] = PEConfig(
    image_size=224,
    patch_size=16,
    width=768,
    layers=12,
    heads=12,
    mlp_ratio=4.0,
    pool_type="attn",
    output_dim=1024,
    use_cls_token=True,
)
PE_TEXT_CONFIG["PE-Core-B16-224"] = PE_TEXT_CONFIG["PE-Core-L14-336"]








#########################################
#                PE Lang                #
#########################################

PE_VISION_CONFIG["PE-Lang-G14-448"] = replace(
    PE_VISION_CONFIG["PE-Core-G14-448"],
    image_size=448,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
    ls_init_value=0.1,
    layers=47,
)

PE_VISION_CONFIG["PE-Lang-L14-448"] = replace(
    PE_VISION_CONFIG["PE-Core-L14-336"],
    image_size=448,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
    ls_init_value=0.1,
    layers=23
)



#########################################
#               PE Spatial              #
#########################################

PE_VISION_CONFIG["PE-Spatial-G14-448"] = replace(
    PE_VISION_CONFIG["PE-Core-G14-448"],
    image_size=448,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
    ls_init_value=0.1,
)
