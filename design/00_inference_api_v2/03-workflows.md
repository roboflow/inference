# Workflows endpoints

Three foundational questions matter more for workflows than the per-endpoint surface area:

1. How to represent and transport input data efficiently.
2. How to deliver responses that may be large and structurally heterogeneous.
3. How to extend the introspection API so that clients can discover which blocks are available in the current environment and how to use them.

This document focuses on those three. Where workflow endpoints mirror the model endpoints, we cross-reference `02-models-endpoints.md` rather than restating the design.

## `POST /v2/workflows/run`

### Input data representation

Running a workflow has always supported two modes:

* **Pre-defined workflow** — the workflow lives in the Roboflow registry and is referenced by `workflow_id` and `workflow_version` (shipped as query params alongside the other control parameters).
* **In-line workflow** — the workflow definition is carried in the request itself. In the JSON variant it sits alongside `inputs` as a `workflow_definition` field; in the multipart variant it occupies a designated `workflow_definition` part.

Beyond `workflow_definition`, the request body carries runtime inputs (images, scalar parameters). The v1 API supported only the JSON transport, with two consequences:

* All input images had to be base64-encoded inside the JSON payload — **slow to construct on the client, slow to send over the wire, slow to deserialise on the server.**
* The workflow could not be validated without first decoding the entire input — wasteful for the validation use case.
* That said, it was **convenient**: a single JSON document, no transport gymnastics.

`v2` keeps the convenience path and adds two faster alternatives. The three request transports introduced for `/v2/models/run` apply directly to `/v2/workflows/run`, with workflow-specific notes:

* **Query-only `POST`.** Suitable for pre-defined workflows whose runtime inputs are all simple types (URLs, scalars). The workflow is identified by `workflow_id` + `workflow_version` query params; runtime inputs are supplied as additional query params keyed by their workflow input names. Subject to the same URL-length caveat as for models.
* **JSON body `POST`.** Equivalent to the v1 shape with auth moved to headers. Supports both pre-defined and in-line workflows: in-line workflows carry their definition under `workflow_definition`. Most expressive option; slowest for image-heavy workloads because of base64 expansion.
* **Multipart `POST`.** Most efficient transport for image-heavy or video-heavy workloads. Designated parts: `workflow_definition` (when in-line, JSON), `inputs` (JSON, may reference sibling parts as `$part.<name>`), plus one binary part per image or other large blob. Pre-defined workflows can omit the `workflow_definition` part and supply `workflow_id` + `workflow_version` as query params.

The same `model_id` URL-encoding rules apply to workflow identifiers carried as query params.

### Output data representation

`/v2/workflows/run` reuses the response envelope and the `rich`/`compact` split defined for `/v2/models/run`. Workflow specifics:

* **Output keying.** A workflow produces multiple named outputs (those declared in its definition). The envelope's `outputs` array carries one entry per workflow output, keyed by output name. Element ordering matches the order of declarations in the workflow definition.
* **Batching.** When the request carries a batch of inputs (e.g., several images), each workflow output expands to a list-of-results aligned with the input batch by index. Empty results (e.g., a downstream filter dropped the input) are represented as explicit `null` entries at the corresponding index — never silently omitted, never compacted to a shorter list.
* **Heterogeneous output types.** Different workflow outputs may carry different prediction types (one detection, one classification, one embedding). Each is independently subject to the `response_style` choice; the choice is request-wide, not per-output.
* **Dense outputs and multipart responses.** When `response_format=multipart` is requested, only outputs whose representation actually benefits from binary transport (segmentation maps, embeddings, dense arrays) are emitted as separate parts. Scalar and small-list outputs remain inside the JSON envelope.
* **Empty workflows / short-circuits.** A workflow that short-circuits (e.g., a gate block rejects all inputs) still returns the full envelope with every declared output present and `null`-valued — the response shape is determined by the workflow definition, not by what executed.

The introspection endpoint (`/v2/workflows/interface`) is responsible for telling clients which outputs to expect; see below.

## `POST /v2/workflows/interface`

Mirrors `/v2/models/interface` (see `02-models-endpoints.md`) with two structural adjustments:

* The endpoint accepts a `POST` rather than `GET`, because in-line workflows ship their definition in the request body. Pre-defined workflows can be introspected with an empty body and `workflow_id` + `workflow_version` as query params.
* The `model_inputs` / `model_outputs` sections become `workflow_inputs` / `workflow_outputs`, keyed by the input/output names declared in the workflow definition. Each entry references the same type catalogue as the model interface.

The data-type representation catalogue and `definitions` mechanism are shared with models — a single source of truth for how, say, an `image` input is represented across both endpoints.

## Other endpoints

The remaining workflow endpoints (`validate`, `system/blocks`, `system/definition-schema`, `system/engine-versions`) are kept semantically identical to their `v1` counterparts. The only changes are:

* paths move under `/v2/`;
* responses are wrapped in the standard `v2` envelope;
* auth is header-only.

### Optional extension: `/validate` runtime readiness

`POST /v2/workflows/validate` can optionally perform a **runtime-readiness check** on top of structural validation: given the validated workflow, can *this* server actually execute it? The check covers:

* whether every required block is available at the requested engine version;
* whether the required model architectures are present;
* whether sufficient hardware resources (GPU, memory) are available;
* whether any block-level prerequisites (e.g., external API keys configured on the server) are met.

The check is gated by a `runtime_readiness=true` query parameter and returns block-by-block diagnostics so that the client can pinpoint exactly which step would fail at execution time. When omitted, the endpoint behaves identically to v1: structural validation only.
