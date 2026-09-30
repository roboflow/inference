"""vLLM proxy backend.

This package implements a serving mode where the inference server runs
CPU-only in front of a vLLM container (OpenAI-compatible API) that owns the
GPU and performs continuous batching + dynamic LoRA. The inference server
keeps doing auth/billing/model-resolution/image-preprocessing per request and
proxies generation to vLLM.

This module intentionally stays import-light (no torch / transformers /
safetensors imports) so that `inference.models.utils` can read the
`VLLM_PROXY_ENABLED` switch without pulling heavy dependencies.
"""

from inference.models.vllm_proxy.config import VLLM_PROXY_ENABLED

__all__ = ["VLLM_PROXY_ENABLED"]
