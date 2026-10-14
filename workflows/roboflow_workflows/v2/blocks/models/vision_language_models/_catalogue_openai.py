"""OpenAI GPT vision models and OpenAI-compatible chat servers.

Ports of V1 ``roboflow_core/open_ai@v7`` and ``roboflow_core/openai_compatible@v1``.
They speak different OpenAI APIs::

    open_ai@v7 (one request per image, run_concurrently)
      task ──▶ ChatPrompt + upload (JPEG, or PNG ≤2048 px for detection)
           ──▶ Responses API payload {model, input, instructions?, text?, ...}
      api_key rf_key:* ─▶ platform_client.post("apiproxy/openai/v2", payload)
      api_key sk-...   ─▶ OpenAI(base_url=openai_base_url,
                                 http_client=openai_http_client).responses.create
      answer text ──▶ decode_vlm_output ──▶ predictions, error_status

    openai_compatible@v1 (one request per call)
      prompt + {{ $parameters.NAME }} ──▶ text; image parameters ──▶ image parts
      ──▶ chat completions at the step's base_url (allowlist checked first)
      failure ──▶ output "", error_status = scrubbed message

Both send through an ``httpx.Client`` resource; a test injects one with an
``httpx.MockTransport`` and nothing leaves the process. Images are encoded on
the host, once per request (``chat.encode_image``).

``model_version=None`` (the default) means V1's default for the task:
``gpt-5.1``, or ``gpt-6-astra`` for instance segmentation. V1 replaced an
omitted ``model_version`` the same way.

Differences from V1:

* A failed open_ai request raises ``ChatRequestError`` with the keys
  scrubbed; openai_compatible reports the scrubbed message in
  ``error_status``. V1 texts could contain the key.
* The OpenAI-compatible allowlist is the ``allowed_base_urls`` resource. Its
  default reads V1's ``OPENAI_COMPATIBLE_ALLOWED_BASE_URLS`` setting.
* ``max_concurrent_requests`` must be at least 1; unset means 8 (V1's
  default) instead of an environment variable.
* Predictions are native V2 carriers; see ``decoding``.
"""


from roboflow_workflows.v2.blocks.models.vision_language_models.open_ai.v7 import (
    OpenAIBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.openai_compatible.v1 import (
    OpenAICompatibleBlock,
)


BLOCKS = (OpenAIBlock, OpenAICompatibleBlock)
