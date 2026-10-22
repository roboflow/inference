"""vLLM proxy backend.

This package implements a serving mode where the inference process runs
CPU-only in front of a vLLM container (OpenAI-compatible API) that owns the
GPU and performs continuous batching + dynamic LoRA. The process keeps doing
model resolution and image preprocessing per request and proxies generation
to vLLM.

The package root stays import-light (no torch / transformers / safetensors
imports) so that selecting the backend never pulls heavy dependencies.
"""
