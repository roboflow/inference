import os

import torch
from peft import LoraConfig
from peft.peft_model import PeftModel
from transformers import PaliGemmaForConditionalGeneration
from transformers.utils import is_flash_attn_2_available

from inference.core.env import DEVICE, MODEL_CACHE_DIR
from inference.models.transformers import LoRATransformerModel, TransformerModel


def _get_paligemma_attn_implementation():
    """Use flash_attention_2 if available, otherwise eager.

    SDPA has dtype mismatch issues with token_type_ids in transformers 4.57+.
    """
    if is_flash_attn_2_available() and DEVICE and "cuda" in DEVICE:
        # Verify flash_attn can actually be imported (not just installed)
        try:
            import flash_attn  # noqa: F401

            if _is_model_running_against_ampere_plus_aarch(device=DEVICE):
                return "flash_attention_2"
            return "eager"
        except ImportError:
            pass
    return "eager"


def _is_model_running_against_ampere_plus_aarch(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device=device)
    return major >= 8


class PaliGemma(TransformerModel):
    """By using you agree to the terms listed at https://ai.google.dev/gemma/terms"""

    generation_includes_input = True
    transformers_class = PaliGemmaForConditionalGeneration

    def initialize_model(self, **kwargs):
        if not self.load_base_from_roboflow:
            model_id = self.dataset_id
        else:
            model_id = self.cache_dir

        self.model = (
            self.transformers_class.from_pretrained(
                model_id,
                cache_dir=self.cache_dir,
                device_map=DEVICE,
                token=self.huggingface_token,
                torch_dtype=self.default_dtype,
                attn_implementation=_get_paligemma_attn_implementation(),
            )
            .eval()
            .to(self.dtype)
        )

        self.processor = self.processor_class.from_pretrained(
            model_id, cache_dir=self.cache_dir, token=self.huggingface_token
        )


class LoRAPaliGemma(LoRATransformerModel):
    """By using you agree to the terms listed at https://ai.google.dev/gemma/terms"""

    generation_includes_input = True
    transformers_class = PaliGemmaForConditionalGeneration
    load_base_from_roboflow = True

    def initialize_model(self, **kwargs):
        import torch

        lora_config = LoraConfig.from_pretrained(self.cache_dir, device_map=DEVICE)
        model_id = lora_config.base_model_name_or_path
        revision = lora_config.revision
        if revision is not None:
            try:
                self.dtype = getattr(torch, revision)
            except AttributeError:
                pass
        if not self.load_base_from_roboflow:
            model_load_id = model_id
            cache_dir = os.path.join(MODEL_CACHE_DIR, "huggingface")
            revision = revision
            token = self.huggingface_token
        else:
            model_load_id = self.get_lora_base_from_roboflow(model_id, revision)
            cache_dir = model_load_id
            revision = None
            token = None
        self.base_model = self.transformers_class.from_pretrained(
            model_load_id,
            revision=revision,
            device_map=DEVICE,
            cache_dir=cache_dir,
            token=token,
            attn_implementation=_get_paligemma_attn_implementation(),
        ).to(self.dtype)
        self.model = (
            PeftModel.from_pretrained(self.base_model, self.cache_dir)
            .eval()
            .to(self.dtype)
        )

        self.model.merge_and_unload()

        self.processor = self.processor_class.from_pretrained(
            model_load_id, revision=revision, cache_dir=cache_dir, token=token
        )
