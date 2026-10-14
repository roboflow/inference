"""Active OpenRouter vendors: Kimi, Meta Muse and Z.ai GLM.

LlamaVisionBlock is retired and excluded from BLOCKS. Its implementation and
shared helpers remain for existing internal references, without a drop-in promise.

Historical implementation context: vendor model blocks served through OpenRouter: Kimi, Meta Muse, Z.ai GLM, Llama.

Ports of V1 ``roboflow_core/kimi_openrouter@v3``, ``roboflow_core/meta_vlm@v3``,
``roboflow_core/zai_vlm@v2`` and ``roboflow_core/llama_vision@v2``. Each block
picks a model by its V1 label and sends OpenRouter chat completions through
``OpenRouterBlockBase``, which owns keys, privacy routing, quantization
filters, the reasoning retry and the Roboflow proxy::

    model_version label ──▶ VendorModel (slug, box format, reasoning rules)
    per image:
      kimi, llama ─▶ openrouter_task_messages      (V1 system + text + image)
      meta, zai   ─▶ image_first_task_messages     (one user message: image, text)
                     image as JPEG; shrunk 10% at a time over 9.5 MB base64
    OpenRouterBlockBase.ask_all ─▶ answers in image order
    kimi, meta, zai: answer ─▶ decode_vlm_output(model box format) ─▶ predictions
    llama: raw answer only (V1 v2 decodes nothing)

Reasoning, as V1: Kimi and Llama send none. Meta always sends an effort
(Muse rejects disabled reasoning). Z.ai disables reasoning by default; a
model that requires reasoning gets ``low`` instead of ``none``.

Differences from V1:

* A failed request raises ``ChatRequestError`` with the keys scrubbed.
* ``max_concurrent_requests`` must be at least 1; unset means 8 instead of an
  environment variable.
* Predictions are native V2 carriers; see ``decoding``.
* Every check runs again on the resolved values before any request: a key
  that resolved to ``None`` (V1 failed on it too; the shared OpenRouter base
  would send the ``OPENAI_API_KEY`` environment variable instead), an
  unknown model label (V1 Kimi and Llama raised ``KeyError``), a reasoning
  level the model rejects, a task field that resolved to ``None``, and
  invalid or duplicate ``classes`` of a decoding task (V1 sent the request
  first).
* A ``model_version`` selector is reported to workload introspection as the
  selector; V1 attached a label-to-slug resolver.
"""


from roboflow_workflows.v2.blocks.language.kimi_openrouter.v3 import KimiOpenRouterBlock
from roboflow_workflows.v2.blocks.language.meta_vlm.v3 import MetaVlmBlock
from roboflow_workflows.v2.blocks.language.zai_vlm.v2 import ZaiVlmBlock


BLOCKS = (KimiOpenRouterBlock, MetaVlmBlock, ZaiVlmBlock)
