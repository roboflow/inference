"""Active compatible vendors: Google Gemma and SpaceXAI.

Qwen35OpenRouterBlock and Qwen36OpenRouterBlock are retired and excluded from
BLOCKS. Their implementations remain; current providers return native predictions.

Historical implementation context: vendor-named vision models served over OpenAI-compatible APIs.

Ports of V1 ``roboflow_core/google_gemma@v4``,
``roboflow_core/qwen3_5_openrouter@v1``, ``roboflow_core/qwen3_6_openrouter@v1``
and ``roboflow_core/spacexai@v3``. Each block fixes one vendor's model table
and prompts; the requests reuse the shared chat helpers::

    google_gemma@v4 (OpenRouterBlockBase, the request loop of openrouter@v3)
      model label ──▶ OpenRouter slug; openrouter@v3 task messages
      api_key rf_key:* ─▶ platform proxy;  sk-or-.. ─▶ OpenRouter directly
      answer ──▶ decode_vlm_output(named_normalized) ──▶ predictions

    qwen3_5_openrouter@v1, qwen3_6_openrouter@v1 (own OpenRouter key only)
      model label ──▶ OpenRouter slug; V1 Qwen messages
      ──▶ chat.completions.create(model, messages, max_tokens, temperature)
      raw answer only; no provider routing, quantization filter or reasoning

    spacexai@v3 (xAI Responses API)
      task ──▶ JPEG, detail "auto"
      object-detection ──▶ lossless PNG at full size, detail "high", image first
      api_key rf_key:* ─▶ platform_client.post("apiproxy/xai", payload)
      xAI key          ─▶ OpenAI(base_url=xai_base_url).responses.create(store=False)
      a direct error naming the reasoning config ─▶ retried once without it, as V1
      answer ──▶ decode_vlm_output(xyxy_percent) ──▶ predictions

Differences from V1:

* A failed request raises ``ChatRequestError`` with the keys scrubbed and no
  chained exception. V1 raised the vendor's exception, whose text may contain
  the key.
* ``max_concurrent_requests`` must be at least 1; unset means 8 instead of an
  environment variable. After a failure, requests not yet started are skipped.
* ``inference_id`` is a hex UUID, as in every V2 block; V1 used the dashed form.
* Predictions are native V2 carriers; see ``decoding``.
* Qwen: a Roboflow-managed ``rf_key:`` key fails before any request (V1 sent
  it to OpenRouter as a bearer token). An answer without ``message.content``
  fails with the OpenRouter message instead of returning ``None`` as output.
"""


from roboflow_workflows.v2.blocks.language.google_gemma.v4 import GoogleGemmaBlock
from roboflow_workflows.v2.blocks.language.spacexai.v3 import SpaceXAIBlock


BLOCKS = (
    GoogleGemmaBlock,
    SpaceXAIBlock,
)
