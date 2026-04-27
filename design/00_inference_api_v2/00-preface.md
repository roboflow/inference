# Inference Server API 2.0
Current HTTP API has grown organically (to 60+ endpoints) has grown organically and suffers from and suffers from:
* **Inconsistent authentication:** API keys accepted via query params, JSON body fields, or middleware -- 6 different patterns across endpoints.
* **Model/resource IDs buried in bodies:** Model IDs are passed in request bodies rather than URLs, preventing load-balancer-level routing to backends that already have those models loaded.
* **Divergent response formats:** The same model (e.g., object detection) produces structurally different responses when called directly vs. through a workflow (e.g., "class" vs "class_name", flat vs nested prediction structures, presence/absence of parent metadata).
* **Overlapping endpoints:** Multiple ways to do the same thing (e.g., `/infer/workflows/{ws}/{wf}` and `/{ws}/workflows/{wf}`; model-specific paths like `/clip/embed_image` alongside generic `/infer/object_detection`).

## Design principles
* **Resource-identifying URLs** - Every URL encodes the model or workflow resource needed, enabling distributed routing without body parsing.
* **Header-only authentication** - API keys always in Authorization headers, never in bodies or query params.
* **Unified execution path** - Direct model inference takes input / produces results equivalent to single-step workflow
* **One way to do each thing** - No duplicate endpoints; for most cases single, opinionated execution path - with the exception of elements that state a trade-off for performance vs simplicity - simplistic methods for making requests should be available in favour of low entry-bar for clients, that should not discard sophisticated solutions designed to ensure maximum performance.
* **Coexistence** - `v2` mounts alongside `v1` under a `/v2` prefix; `v1` remains for backward compatibility during migration.

## Authorization
We wnat to standardise auth such that it's both secure and usable w/o body parsing - hence standart bearer token auth is proposed.

```
Authorization: Bearer <api_key>
```

**Original author of preamble: @Thomas**, some modifications introduced by @Paweł
