import torch
from transformers import AutoModelForCausalLM

from inference.models.transformers import LoRATransformerModel, TransformerModel


class Florence2(TransformerModel):
    hf_args = {"trust_remote_code": True}
    transformers_class = AutoModelForCausalLM
    default_dtype = torch.float32


class LoRAFlorence2(LoRATransformerModel):
    hf_args = {"trust_remote_code": True}
    transformers_class = AutoModelForCausalLM
    default_dtype = torch.float32
