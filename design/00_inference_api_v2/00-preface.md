# Inference Server API 2.0

The current HTTP API has grown organically to 60+ endpoints and suffers from:

* **Inconsistent authentication.** API keys are accepted via query params, JSON body fields, or middleware — six different patterns across endpoints.
* **Model and resource IDs buried in bodies.** Identifiers travel inside request bodies rather than URLs, which prevents load-balancer-level routing to backends that already have the relevant models loaded.
* **Divergent response formats.** The same model (e.g., object detection) produces structurally different responses when called directly vs. through a workflow — `class` vs `class_name`, flat vs nested prediction structures, presence or absence of parent metadata.
* **Overlapping endpoints.** Multiple ways to do the same thing — e.g., `/infer/workflows/{ws}/{wf}` and `/{ws}/workflows/{wf}`; model-specific paths like `/clip/embed_image` alongside generic `/infer/object_detection`.

## Design principles

* **Resource-identifying URLs.** Every URL encodes the model or workflow resource being addressed, enabling distributed routing without body parsing.
* **Header-only authentication.** API keys always travel in the `Authorization` header, never in bodies or query params.
* **Unified execution path.** Direct model inference accepts the same inputs and produces the same results as a single-step workflow that wraps the same model.
* **One way to do each thing.** No duplicate endpoints. For most operations the API exposes a single, opinionated execution path. The one exception is performance vs. simplicity: where that trade-off is real, we expose both a simple low-entry-bar variant and an advanced high-performance variant. The two variants must never carry different semantics — only different transport.
* **Coexistence.** `v2` mounts alongside `v1` under a `/v2` prefix. `v1` remains in place for backward compatibility throughout the migration window.

## Authorization

Authentication is standardised so that it is both secure and usable without body parsing. Bearer-token auth is the only supported scheme:

```
Authorization: Bearer <api_key>
```

API keys are never accepted in query params or request bodies in `v2`.

---

*Original author of preamble: @Thomas. Modifications introduced by @Paweł.*
