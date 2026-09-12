# Zero data retention in LLM blocks

ZDR must already be enabled for the provider account/project associated with the
API key, including Roboflow-managed keys. The OpenAI and Gemini block options do
not provision or verify that account setting.

| Block | Configuration | Request behavior |
| --- | --- | --- |
| OpenAI (`roboflow_core/open_ai@v6`) | `zero_data_retention: true` | Sends `store: false`, both directly and through the Roboflow passthrough. |
| Gemini (`roboflow_core/google_gemini@v5`) | `zero_data_retention: true` | Uses the project's ZDR policy and rejects explicit context caching and Search/Maps grounding before sending a request. |
| OpenRouter (`roboflow_core/openrouter@v2`) | `privacy_level: "zdr"` | Restricts routing to ZDR endpoints with `provider.zdr: true` and `data_collection: "deny"`. Works with managed and custom keys. |

For example, an OpenAI step using a custom key supplied as a workflow input:

```json
{
  "type": "roboflow_core/open_ai@v6",
  "name": "caption",
  "images": "$inputs.image",
  "task_type": "caption",
  "api_key": "$inputs.openai_api_key",
  "zero_data_retention": true
}
```

OpenAI and Gemini default to `zero_data_retention: false` to preserve existing
workflow behavior. This does **not** disable ZDR already configured on the key.
OpenRouter retains its existing `privacy_level: "deny"` default, which excludes
training/data-collection providers but does not require zero retention.

Gemini's `generateContent` API has no per-request ZDR switch; the block does not
send an unsupported `store` or `zero_data_retention` parameter to Google. Its
normal inline-image requests remain compatible with a ZDR-enabled project.

OpenAI's `store: false` disables response storage; exclusion from abuse-monitoring
logs still depends on the account/project's ZDR configuration. OpenRouter may
reject requests when the selected model has no eligible ZDR endpoint.

Provider documentation:

- [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data)
- [Gemini zero data retention](https://ai.google.dev/gemini-api/docs/zdr)
- [OpenRouter zero data retention](https://openrouter.ai/docs/guides/features/zdr)
