import logging
import os
import re
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
)

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_QWEN3_5_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_QWEN3_5_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.entities import ColorFormat
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_inference_config,
)
from inference_models.models.qwen3vl.qwen3vl_hf import _get_qwen3vl_attn_implementation

logger = logging.getLogger(__name__)


def _fixed_torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Fixed copy of transformers' torch_chunk_gated_delta_rule.

    The upstream pure-PyTorch fallback allocates a state tensor on CPU then
    moves it to GPU via ``.to(value)``, which triggers "illegal memory access"
    on some GPU/driver combinations (observed on L40S + CUDA 12.8).

    This version allocates directly on the correct device/dtype.
    """
    initial_dtype = query.dtype

    if use_qk_l2norm_in_kernel:
        inv_norm = torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        query = query * inv_norm
        inv_norm = torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * inv_norm

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = torch.nn.functional.pad(query, (0, 0, 0, pad_size))
    key = torch.nn.functional.pad(key, (0, 0, 0, pad_size))
    value = torch.nn.functional.pad(value, (0, 0, 0, pad_size))
    beta = torch.nn.functional.pad(beta, (0, pad_size))
    g = torch.nn.functional.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    # FIX: allocate directly on device instead of CPU .to(value)
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim,
            dtype=value.dtype, device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _fixed_torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    """Fixed copy of transformers' torch_recurrent_gated_delta_rule.

    Same CPU allocation bug as torch_chunk_gated_delta_rule — two
    ``torch.zeros(...).to(value)`` calls that must allocate on device directly.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        inv_norm = torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        query = query * inv_norm
        inv_norm = torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * inv_norm

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    # FIX: allocate directly on device instead of CPU .to(value)
    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim,
        dtype=value.dtype, device=value.device,
    )
    last_recurrent_state = (
        torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim,
            dtype=value.dtype, device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _patch_model_linear_attn_layers(model: Qwen3_5ForConditionalGeneration):
    """Replace gated delta rule fallbacks on all linear attention layers if fla is missing."""
    try:
        from transformers.utils.import_utils import is_flash_linear_attention_available

        if is_flash_linear_attention_available():
            return
    except ImportError:
        pass

    logger.warning(
        "flash-linear-attention (fla) is not installed. Using fixed "
        "torch fallbacks to avoid CUDA illegal memory access. "
        "Install fla for optimal Qwen3.5 performance: "
        "pip install 'flash-linear-attention>=0.4.0,<0.5.0'"
    )
    patched = 0
    for module in model.modules():
        if hasattr(module, "chunk_gated_delta_rule"):
            module.chunk_gated_delta_rule = _fixed_torch_chunk_gated_delta_rule
            patched += 1
        if hasattr(module, "recurrent_gated_delta_rule"):
            module.recurrent_gated_delta_rule = _fixed_torch_recurrent_gated_delta_rule
    logger.info("Patched %d linear attention layers", patched)


class Qwen35HF:
    default_dtype = torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        disable_quantization: bool = False,
        **kwargs,
    ) -> "Qwen35HF":
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        inference_config_path = os.path.join(
            model_name_or_path, "inference_config.json"
        )
        inference_config = None
        if os.path.exists(inference_config_path):
            inference_config = parse_inference_config(
                config_path=inference_config_path,
                allowed_resize_modes={
                    ResizeMode.STRETCH_TO,
                    ResizeMode.LETTERBOX,
                    ResizeMode.CENTER_CROP,
                    ResizeMode.LETTERBOX_REFLECT_EDGES,
                    ResizeMode.FIT_LONGER_EDGE,
                },
            )

        attn_implementation = _get_qwen3vl_attn_implementation(device)

        if os.path.exists(adapter_config_path):
            # Has adapter - load base model then apply LoRA
            base_model_path = os.path.join(model_name_or_path, "base")
            base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                base_model_path,
                device_map=device,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
            )
            # Apply LoRA, eval, convert to dtype
            model = (
                PeftModel.from_pretrained(base_model, model_name_or_path)
                .eval()
                .to(cls.default_dtype)
            )
            model = model.merge_and_unload()
        else:
            # No adapter - just load base model, eval, convert to dtype
            base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_name_or_path,
                device_map=device,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
            )
            model = base_model.eval().to(cls.default_dtype)

        _patch_model_linear_attn_layers(model)

        # Load processor with chat_template if available
        # Check both root and base/ directory for chat_template.jinja
        chat_template_path = os.path.join(model_name_or_path, "chat_template.jinja")
        if not os.path.exists(chat_template_path):
            chat_template_path = os.path.join(
                model_name_or_path, "base", "chat_template.jinja"
            )

        # Use base/ for processor when adapter exists - preprocessor configs differ
        if os.path.exists(adapter_config_path):
            processor_path = os.path.join(model_name_or_path, "base")
        else:
            processor_path = model_name_or_path

        if os.path.exists(chat_template_path):
            with open(chat_template_path, "r") as f:
                chat_template = f.read()
            processor = AutoProcessor.from_pretrained(
                processor_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                chat_template=chat_template,
                min_pixels=16 * 32 * 32,
                max_pixels=512 * 32 * 32,
            )
        else:
            processor = AutoProcessor.from_pretrained(
                processor_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                min_pixels=16 * 32 * 32,
                max_pixels=512 * 32 * 32,
            )

        return cls(
            model=model,
            processor=processor,
            inference_config=inference_config,
            device=device,
        )

    def __init__(
        self,
        model: Qwen3_5ForConditionalGeneration,
        processor: AutoProcessor,
        inference_config: Optional[InferenceConfig],
        device: torch.device,
    ):
        self._model = model
        self._processor = processor
        self._inference_config = inference_config
        self._device = device
        self.default_system_prompt = "You are a helpful assistant."
        self._lock = Lock()

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        skip_special_tokens: bool = True,
        enable_thinking: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        inputs = self.pre_process_generation(
            images=images,
            prompt=prompt,
            input_color_format=input_color_format,
            enable_thinking=enable_thinking,
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        return self.post_process_generation(
            generated_ids=generated_ids,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
        )

    def pre_process_generation(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        image_size: Optional[Tuple[int, int]] = None,
        enable_thinking: bool = False,
        **kwargs,
    ) -> dict:
        # Handle prompt and system prompt parsing logic from original implementation
        if prompt is None:
            prompt = "Describe what's in this image."
            system_prompt = self.default_system_prompt
        else:
            split_prompt = prompt.split("<system_prompt>")
            if len(split_prompt) == 1:
                prompt = split_prompt[0] or "Describe what's in this image."
                system_prompt = self.default_system_prompt
            else:
                prompt = split_prompt[0] or "Describe what's in this image."
                system_prompt = split_prompt[1] or self.default_system_prompt

        # Construct conversation following qwen3vl inference pattern
        # Pass the actual image in the conversation for proper vision token handling
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # processor.__call__() doesn't forward enable_thinking to the Jinja
        # template, so we call apply_chat_template separately then tokenize.
        text = self._processor.apply_chat_template(
            [conversation],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if isinstance(text, list):
            text = text[0]

        model_inputs = self._processor(
            text=[text],
            images=[images],
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to device and cast floating-point tensors to model dtype
        # to avoid dtype mismatches that cause CUDA illegal memory access errors
        model_inputs = {
            k: v.to(device=self._device, dtype=self.default_dtype)
            if isinstance(v, torch.Tensor) and v.is_floating_point()
            else v.to(device=self._device)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in model_inputs.items()
        }

        return model_inputs

    def generate(
        self,
        inputs: dict,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = INFERENCE_MODELS_QWEN3_5_DEFAULT_DO_SAMPLE,
        **kwargs,
    ) -> torch.Tensor:
        if max_new_tokens is None:
            max_new_tokens = INFERENCE_MODELS_QWEN3_5_DEFAULT_MAX_NEW_TOKENS
        input_len = inputs["input_ids"].shape[-1]

        with self._lock, torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                bos_token_id=self._processor.tokenizer.bos_token_id,
            )

        # Return only the newly generated tokens
        return generation[:, input_len:]

    def post_process_generation(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool = False,
        enable_thinking: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        # Decode the generated tokens
        decoded = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        )

        # Apply post-processing - clean up think tags and other artifacts
        result = []
        for text in decoded:
            # Clean common artifacts from all outputs
            text = text.replace("<|im_end|>", "")
            text = text.replace("<|endoftext|>", "")
            text = text.replace("assistant\n", "")
            text = text.replace(" addCriterion\n", "")

            if enable_thinking:
                # The generated output only contains NEW tokens (input is
                # trimmed in generate()). Since the input ends with "<think>\n",
                # the opening <think> tag is NOT in the decoded output.
                # Prepend it so the regex can parse thinking vs answer.
                text = "<think>" + text
                # Extract thinking and answer separately
                think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                    answer = re.sub(
                        r"<think>.*?</think>\s*", "", text, flags=re.DOTALL
                    ).strip()
                else:
                    # Model hit max tokens before producing </think>.
                    # Everything after <think> is thinking, no answer yet.
                    thinking = text.replace("<think>", "").strip()
                    answer = ""
                result.append({"thinking": thinking, "answer": answer})
            else:
                # Remove <think>...</think> blocks (Qwen3.5 reasoning format)
                text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
                result.append(text.strip())

        return result


def adjust_lora_model_state_dict(state_dict: dict) -> dict:
    return {
        refactor_adapter_weights_key(key=key): value
        for key, value in state_dict.items()
    }


def refactor_adapter_weights_key(key: str) -> str:
    if ".language_model." in key:
        return key
    return (
        key.replace("model.layers", "model.language_model.layers")
        .replace(".weight", ".default.weight")
        .replace(".lora_magnitude_vector", ".lora_magnitude_vector.default.weight")
    )
