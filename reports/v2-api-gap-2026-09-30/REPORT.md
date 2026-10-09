# V2 API gap report

Prepared for Damian Kosowski and the inference-core team on 30 September 2026.

The `feat/new-model-manager` branch has a substantial working model-serving foundation, but its HTTP contract does **not yet implement the proposal in PR #2277**. The main gaps are the public route names and methods, response envelope and defaults, input and output transports, model-package selection, and machine-readable interface discovery. Compatibility discovery is a stub. All six proposed V2 workflow endpoints are absent; legacy workflow functionality exists separately.

The immediate deliverable agreed in the meeting was this inventory, with enough evidence for someone else to take over the remaining work. The recommendations below are a proposed implementation sequence, not newly assigned owners or deadlines. Legacy parity, stream/pipeline migration and usage implementation remain separate workstreams.

## Scope and fixed references

| Source | Revision inspected | Role |
|---|---|---|
| [PR #2277][design-pr] | `de634b98bac204c96caa98a15dd7559dded361d5` | Draft design baseline: preface, API structure, models and workflows. The substantive specification is in the four added Markdown files; the PR body is an unfilled template. |
| Public `inference`, `origin/feat/new-model-manager` | `3d45b8712cc428eb01b714f3609346be7c92acc4` | Implementation baseline, fetched and checked out for this audit. |
| Private `inference-closed-plugins`, `origin/main` | `18f2c22543b499727093cf180d45e835ba1c911f` | Companion serving runtime, inspected and tested from a temporary source snapshot. The user's plugins working tree was left unchanged at `72842fa`. No remote `feat/new-model-manager` branch exists in this repository. |
| [Meeting transcript][meeting] | 30 September, especially 36:30 and 40:37–43:10 | Model endpoints are the initial focus. Compare against the draft design using the new-model-manager branch and both repositories. Produce the report before lower-priority implementation. |

The fetched public branch had been force-updated relative to the locally cached remote ref. All implementation links in this report are pinned to the inspected commit, so later branch changes do not alter the evidence.

**Evidence levels:** “Observed” means a local request or focused test exercised the behavior. “Source” means code inspection establishes the implementation path or absence. The HTTP contract probes mock authentication, registry access, URL downloads and model execution, while running the real middleware, routers, parsers, dispatch and serializers. Seven additional endpoint checks use a real local MMP. No trained-model inference, GPU execution, production traffic, or deployed-server validation was performed.

## What already works

- The application mounts V2 model and server routers alongside legacy routes. The framework resolves a model task and action, parses input, checks parameters, ensures the model is loaded, invokes the gateway and serializes the result. There are **42 task/action registrations across 15 task families**; registration is not a claim that every model was exercised. Families include detection, classification, segmentation, OCR, embeddings, VLMs, depth, gaze, keypoints and passthrough. [Dispatch][dispatch], [handler registry][registry]
- Query URL images, JSON base64 images, raw image bodies, and repeated multipart `image` uploads work in the offline HTTP probes. Scalar query parameters are coerced; the two-image multipart probe produced two results. Existing tests cover image-count/body limits and guarded URL fetching. [Parsers][parsers], [input tests][input-tests]
- Explicit rich and compact object-detection output both work, including corner coordinates and class names when prediction metadata provides them. Several other task families already have typed serializers, although their contracts differ from the proposal. [Typed serializers][typed], [observations][observations]
- V2 rejects query-only and body-only API keys without a Bearer header. Model inference performs a separate model authorization lookup; loaded-model interface discovery follows a different path, described below. [Middleware][app], [model authorization][model-stat]
- Direct and MMP gateways implement the same lifecycle/inference interface. Local tests verified direct gateway behavior, plugin gateway contracts, and MMP health/readiness/stats paths. These are reusable foundations for closing the HTTP gaps. [Direct gateway][gateway], [MMP gateway][mmp], [validation record][validation]

## Endpoint comparison

The proposed endpoint surface comes from [API structure][design-structure]. Unless stated otherwise, observed results below use a valid mocked Bearer identity and enable control-plane routes. Missing routes were also checked against registration source; a middleware response alone is not evidence that a route exists.

| Proposed method and path | Current implementation and evidence | Required follow-up |
|---|---|---|
| `POST /v2/models/run` | **Different contract.** Execution is at `POST /v2/models/infer`; `/run` is unregistered. `/infer` returns predictions in local probes. [Model routes][model-routes] | Establish the canonical path, then align transport, envelope, output and interface behavior described below. |
| `GET /v2/models/interface` | **Partial, different schema.** Loaded gateway data is returned verbatim; unloaded models receive task-level action descriptions. Both local paths return 200, but neither implements the proposed discoverable contract. [Interface implementation][interface] | Build the shared representation catalogue, model input/output schemas, definitions and filters. Make discovery consistent before and after loading. |
| `GET /v2/models/compatibility` | **Stub.** Always returns 501 `NOT_IMPLEMENTED` after middleware. [Compatibility stub][compatibility] | Implement architecture compatibility discovery. First settle response schema and runtime capability semantics. |
| `GET /v2/models/loaded` | **Different path.** `GET /v2/models` returns a `models` map from gateway stats; `/loaded` is absent. The current route returns 200 in the probe. [List route][model-list] | Align the path and define whether entries in loading/error states belong in a “loaded” response. |
| `POST /v2/models/load` | **Implemented at the proposed path.** Requires query `model_id`, delegates to gateway load and returns `{model_id,status}`. Mocked success is 200; missing ID is tested with real MMP. [Load route][model-load] | Specify lifecycle response/error semantics and validate real model loads on both backends. The draft provides no detailed lifecycle schema to certify. |
| `DELETE /v2/models/unload` | **Different method and split behavior.** Single-model unload is `POST /v2/models/unload`; unload-all is `DELETE /v2/models`. Omitting the ID from the POST gives 400. [Unload routes][model-unload] | Provide the proposed DELETE operation with optional model ID, or explicitly revise the design. Define partial-failure behavior. |
| `POST /v2/workflows/run` | **Absent.** Legacy inline `/workflows/run` and predefined `/{workspace_name}/workflows/{workflow_id}` routes exist. [Workflow routes][workflow-routes] | Add the V2 facade with the three transports, V2 envelope and named output/batch rules. |
| `POST /v2/workflows/interface` | **Absent.** Legacy `/workflows/describe_interface` and predefined `describe_interface` routes exist. | Reuse the V2 type catalogue for `workflow_inputs`, `workflow_outputs` and transport descriptions. |
| `POST /v2/workflows/validate` | **Absent.** Legacy `/workflows/validate` exists. | Add V2 authentication/envelope adaptation. Keep optional runtime readiness separate from structural validation. |
| `GET /v2/workflows/system/blocks` | **Absent.** Legacy `/workflows/blocks/describe` exists. | Expose existing semantics under the V2 path with the agreed envelope/authentication. |
| `GET /v2/workflows/system/definition-schema` | **Absent.** Legacy `/workflows/definition/schema` exists. | Add V2 adaptation. |
| `GET /v2/workflows/system/engine-versions` | **Absent.** Legacy `/workflows/execution_engine/versions` exists. | Add V2 adaptation. |
| `GET /v2/server/health` | **Implemented.** Returns 200 `{status:"ok"}` without authentication, including with a real MMP attached. [Server routes][server-routes] | Confirm that public liveness is an intended exception to the preface's header-only authentication rule. |
| `GET /v2/server/ready` | **Implemented with bounded meaning.** Checks stats connectivity and that configured preload IDs are `loaded`; no-preload MMP check passes. | Document that this does not establish the ability to run every available model/workflow. Confirm public probe policy. |
| `GET /v2/server/info` | **Partial.** Returns server name, model count and each model's state/device. No build version or general configuration metadata is returned, despite the design's stated purpose. | Define and return build/configuration metadata; retain deliberate disclosure limits. |
| `GET /v2/server/metrics` | **Different format.** Returns JSON gateway stats, including with real MMP. A source TODO explicitly defers Prometheus text support. | Implement the promised Prometheus-compatible export and specify content negotiation. |

**Control-plane default:** list/load/unload, info and metrics return 403 unless `ENABLE_CONTROL_PLANE_ROUTES=true`. The routes remain registered. This is an explicit implementation policy, not a missing-route defect; the proposal should state it. Health and ready are explicitly unauthenticated. [Middleware route sets][app]

**Missing-route response detail:** with the landing page mounted in this checkout, the probes saw 405 for missing POST routes and HTML 404 for missing GET routes. A deployment without that static mount can behave differently. The reliable finding is that the proposed routes are unregistered, rather than a universal claim about their error status.

## Cross-cutting model contract gaps

| ID | Expected by the proposal | Current behavior and consequence | Evidence |
|---|---|---|---|
| G1 | Default `response_style=rich`; common envelope with `outputs` entries and example `inference_id` metadata. | Defaults to `compact`; emits `type`, `model_info`, empty `usage`, and `predictions`. No `outputs` or inference IDs in the model response. A client generated from the design cannot consume this unchanged. | Observed; [common parameters][dispatch], [OD envelope][od-output] |
| G2 | JSON image inputs accept URL/base64 representations; multipart inputs can reference named siblings using `$part.<name>`. | JSON images accept only `type=base64`. The proposal's URL input returns 400 `INVALID_IMAGE`. Multipart only reads uploaded files named `image`; a valid JSON reference to `$part.frame` with a sibling `frame` upload returns 400 `EMPTY_BODY`. | Observed; [JSON parser][json-parser], [multipart parser][multipart-parser] |
| G3 | `response_format=json|multipart`, with a JSON `response` part and dense binary array parts. | No HTTP multipart response serializer. `response_format=multipart` is forwarded as a model parameter; the accepting fake gateway still yields JSON. With real model methods, the stray keyword may be ignored or rejected; that outcome was not tested. Internal plugin shared-memory transport does not provide HTTP multipart responses. | Observed/source; [dispatch][dispatch], [output serializers][od-output], [MMP result decoding][mmp] |
| G4 | Optional `model_package_id` selects a specific model package. | Parsed into `CommonRequestParams`, then unused by the server loading/inference path. The probe records the same model/instance/device load request with and without package selection. Neither gateway's `load`/`ensure_loaded` API takes a package ID. | Observed/source; [dispatch][dispatch], [direct gateway][gateway], [MMP gateway][mmp] |
| G5 | Repeatable `requested_output` filters returned outputs. | Not implemented as output selection. Query parsing reduces repeats to the last value and forwards it to model inference; the response is unfiltered in the probe. | Observed; [query decoding][dispatch] |
| G6 | A common machine-readable contract with `control_parameters`, `request_formats`, `model_inputs`, `model_outputs`, `$ref`/`definitions`, and format/style filtering. | Current interface is `{model_id,actions}` for loaded models and adds `model_type` for fallback. Fallback actions contain `task`, `params`, `output_schema`; loaded manager actions contain `method`, `default`, `params`, `response_type`. Filter parameters have no effect. Fallback advertises `images`, while the HTTP JSON/form input is named `image`, without a representation mapping. | Observed/source; [interface route][interface], [manager action metadata][actions], [OD introspection][od-interface] |
| G7 | Direct inference and an equivalent single-step workflow share input/output semantics. | Legacy workflow execution uses a gateway-backed models provider and legacy repacking; it does not use the V2 HTTP serializers. There is no V2 workflow surface on which to establish the promised equivalence. Shared model execution alone does not establish wire-contract parity. | Source; [workflow provider][workflow-provider], [workflow execution][workflow-execution] |

The interface fallback is task-level, whereas a loaded model's actions come from its concrete class/MRO. Until discovery is unified, loading a model can change both the action inventory and the schema of the discovery response. An unloaded VLM can receive the generic VLM action set, which is not proof that a particular model supports every listed action. [Interface fallback][interface], [handler registry][registry], [manager actions][actions]

## Prediction representation gaps

These differences matter independently of the outer envelope. Matching a `type` string is insufficient: several current payloads use the proposal's type identifier with different field names or semantics. Expectations below are from [model endpoint design][design-models]; implementations are in [typed serializers][typed] and their HTTP wrappers.

| Family | Current implementation | Gap against the proposal |
|---|---|---|
| Object detection | Compact `xyxy`, `class_id`, `confidence`; rich `left_top`/`right_bottom`, confidence and class ID. OD can recover class names from `image_metadata`. | Core coordinate representation aligns. Serializers do not emit the illustrated tracker/detection IDs. The design should clarify their optionality and provenance; absence is not proof that tracking itself is required for every model. |
| Single-label classification | Compact uses `confidences` and `top_classes_ids`; rich uses `candidates` and `top`. HTTP wrapper passes `class_names=None`. | Proposal uses `confidence`, `predicted_class_ids`, `confidence_threshold`, and rich `predicted_classes` with class names. The serializer does not apply a confidence threshold; the underlying base class explicitly documents top-1 semantics. Define empty-result/threshold semantics before adding a conforming adapter. |
| Multi-label classification | Compact uses `confidences` and `detected_classes_ids`. A rich request deliberately falls back to compact output. Class names are null through the HTTP wrapper. | Does not share the proposed single-/multi-label structure; rich representation and threshold metadata are absent. |
| Instance segmentation | Compact includes dense masks or per-detection COCO RLE dictionaries, depending on the raw mask object. Rich puts each mask under `mask`. | No proposed compact cropped-RLE envelope (`rles`, `crop_shapes`, `offsets`, image size). Rich does not expose the proposed `rle_mask` and top-level `image_size`; class names are not supplied by the wrapper. Dense-mask behavior was observed; COCO conversion is established by source. |
| Semantic segmentation | Uses `roboflow-semantic-segmentation-compact-v1`, `segmentation_map`, `confidence`, and null class names. Rich selects the same serializer. | Proposal uses `roboflow-semantic-segmentation-v1`, `pixel_scores`, class names, and optional multipart binary maps. |
| Embeddings and depth | Dense arrays are serialized inside typed JSON wrappers. Depth ignores the rich/compact distinction. | JSON arrays provide a usable baseline; efficient binary multipart output is missing. The proposal does not fully specify every wrapper, so wrapper naming should be settled explicitly. |
| Text and structured OCR | Text is `{type,text}`. Structured OCR has its own `batch` of `text`/`regions`, with optional region `texts`, and always uses its compact serializer. | Proposal says text-only output remains a string and structured OCR is a detection variant with distinct text content. Final OCR/text placement within the common envelope needs clarification. OCR representation differences are source findings; real OCR was not executed. |

Keypoints, interactive segmentation, gaze and the VLM action set extend beyond the detailed prediction examples in the draft. They are existing capabilities to preserve, but the current draft is too thin to label every one conforming or defective.

## Additional defects reproduced during the audit

These are observable input-handling problems, separate from design choices:

1. **A JSON array body produces HTTP 500.** `extract_json_base64()` calls `body.get` before checking that the top-level JSON value is an object. Probe: `POST /v2/models/infer?model_id=audit/1` with `[]`. Add structural validation and return a client error. [JSON parser][json-parser]
2. **Malformed multipart `inputs` JSON is silently discarded.** With a valid `image` upload and `inputs={bad-json`, the request still returns 200 and runs inference with empty parameters. A malformed threshold or other input can therefore fall back to defaults instead of being rejected. [Multipart parser][multipart-parser]

Both are recorded in [observations.json][observations]. Authentication and actual inference are mocked, but the failing parser and HTTP response paths are real.

## Companion repository findings

`rf-serving` supplies the MMP gateway, subprocess backend and decoders through entry points. It does not replace the public V2 routers or their serializers. `rf-legacy-bridge` adapts the legacy host; it is not an alternative V2 implementation. Consequently, route/envelope/parser fixes belong primarily in the public repository, while changes to gateway arguments or transport must be coordinated across both repositories. [Plugin packaging][plugin-package], [gateway resolution][gateway-resolver]

The current private package pins `inference-models==0.39.0rc3`, `inference-model-manager==0.5.0`, and `inference-server==0.4.0`. Its CI is pinned to public commit `1194b97d287af4017763872f1563f82b0963ffc5`, rather than the public tip used here. Local combined-source tests provide evidence for the inspected pair, but do not replace that pinned CI's full wheel/hardware validation. [Private CI][plugin-ci]

Both gateways expose package-unaware lifecycle signatures and return the loaded-model interface from their stats action directory. MMP's internal pickled/shared-memory result is deserialized before the public handler constructs the HTTP response. This is why adding HTTP multipart output requires work above that internal transport. [Direct gateway][gateway], [MMP gateway][mmp]

## Decisions required before implementation

The design is still a draft. Record agreed changes in the design rather than silently treating every existing implementation difference as a new requirement.

- **Canonical API and compatibility window:** retain the proposed `/run`, `/loaded`, and DELETE unload operation, or explicitly amend them. Decide whether current experimental paths need temporary aliases.
- **Envelope and output identity:** define output names, batch placement, inference-ID generation, optional metadata and how text/dense values fit the envelope. The draft provides examples rather than a complete formal schema. Usage fields should be coordinated with Grzegorz's separate workstream.
- **Authentication policy:** confirm public health/readiness and default-disabled control-plane routes. Loaded interface discovery currently returns gateway metadata after general key validation, without the per-model lookup used by fallback; decide the intended model-metadata authorization policy. No cross-workspace access was tested here.
- **Classification semantics:** define confidence thresholds, empty single-label results and any top-N input. Top-N is raised as a problem in the draft but is not assigned a precise query parameter/contract.
- **Lifecycle and compatibility schemas:** specify compatible architecture reporting, loaded-state filtering, unload-all failure status and error payloads. The outline alone is insufficient for exact acceptance tests.
- **Existing open questions:** keep raw-array fallback encoding, nested `$part` access and type-version policy as decisions. Their unresolved status does not excuse missing basic multipart or part-level references. Runtime-readiness validation is explicitly optional, and video stream processing is explicitly deferred.
- **Correct draft examples:** the multipart example uses unquoted `$part.image1` tokens inside JSON, which is invalid JSON; the audit uses quoted reference strings. The cropped-mask example has inconsistent lengths between RLEs, crop shapes and offsets. Fix examples before turning them into executable fixtures.

## Recommended follow-up sequence

Priority here means order for finishing V2, not priority over the team's legacy release work.

| Order | Deliverable | Acceptance evidence |
|---|---|---|
| 1 | Agree the contract decisions and turn the draft into versioned schemas and example fixtures. | Every supported transport/style has a valid fixture, with explicit optional fields and unresolved features excluded or labeled. |
| 2 | Align model paths/methods, default style, envelope, reserved parameters and discovery schema (G1, G5, G6). Fix the two parser defects. | Tests generated from the agreed contract pass; malformed inputs return client errors; discovery describes the actual HTTP inputs and is stable before/after loading. |
| 3 | Wire package selection through both gateways and define compatibility discovery (G4 and the stub endpoint). | Distinct packages are actually selected on cold/warm paths with well-defined cache/routing identity; compatibility is checked against server capabilities. Both gateway contract suites pass. |
| 4 | Complete typed serializers and shared metadata handling. | Classification empty/threshold cases, class names, rich multi-label output, segmentation masks, OCR, batches and optional IDs match schemas. Test actual model outputs as well as synthetic fixtures. |
| 5 | Implement shared input/output transport codecs (G2, G3). | Equivalent query/JSON/multipart inputs produce equivalent results; URL images, named binary parts, dense arrays and multipart response decoding round-trip. |
| 6 | Build the V2 workflow facade on those shared contracts (G7), coordinating its scope with Paweł. | Predefined/inline workflows, named outputs, order, null batch positions and short-circuit outputs match the design; direct inference and a single-step workflow are compared. |
| 7 | Finish server metadata and Prometheus metrics, then validate deployed behavior. | CPU/direct and MMP tests use real representative models, with GPU and staging checks where needed. Preserve existing V1 parity checks as a separate release gate. |

The reusable pieces are the current dispatch/handler architecture, model manager, direct/MMP gateway pair, guarded image fetching and legacy workflow host. The work is predominantly contract completion and adapters around those pieces, with cross-repository changes required for package selection. The report does not establish that merely renaming routes would make V2 complete.

## Validation and limitations

**318 focused existing tests passed** across five runs:

| Run | Result | What it establishes |
|---|---|---|
| Public server routes, parsers, dispatch, auth, selected handlers and gateway signature | 148 passed | Current implementation behavior, mostly with mocked registry/model execution. |
| Direct gateway and model-stat tests | 92 passed | Direct lifecycle/dispatch behavior and authorization-resolution logic. |
| Typed serializers and default action registry | 47 passed | Current serializer structures and model-action registration. |
| Plugin gateway contract, state vocabulary, load timeouts and worker errors | 24 passed | Private gateway compatibility and error handling in focused tests. |
| Local real-MMP endpoint integration | 7 passed, 4 deselected | Health, readiness without preloads, info, JSON metrics and missing-ID validation. Auth is mocked; no model weights loaded. |

The additional audit script recorded **51 HTTP observations**, including both happy paths and intentional failure cases. It records evidence rather than declaring proposal conformance. The route inventory and handler registry are captured alongside responses. [Probe source][probe], [observations][observations], [exact commands and logs][validation]

The original Python 3.10 environment failed collection because SciPy 1.15.3 could not load a macOS binary. A temporary environment used SciPy 1.14.1 and the branch-required test dependencies; an old pytest-asyncio fixture incompatibility was also resolved there. The development environment and product source were not edited. Final runs still emit existing Pydantic deprecation warnings.

The workflows extra was not installed in this test environment. Workflow endpoint absence was checked in the source registration and router definitions as well as the local route inventory; **legacy workflow execution was not tested**. Full trained-model correctness, numerical parity, all 42 actions, GPU/subprocess model execution, frontend/client compatibility, real API-key authorization, and production/staging behavior remain unverified.

[design-pr]: https://github.com/roboflow/inference/pull/2277
[design-structure]: https://github.com/roboflow/inference/blob/de634b98bac204c96caa98a15dd7559dded361d5/design/00_inference_api_v2/01-general-api-structure.md
[design-models]: https://github.com/roboflow/inference/blob/de634b98bac204c96caa98a15dd7559dded361d5/design/00_inference_api_v2/02-models-endpoints.md
[meeting]: https://app.avoma.com/meetings/f81d1ac7-dfb7-4be7-a8ec-43ce4ddac5d8/transcript
[model-routes]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/routers/v2_models.py#L43
[interface]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/routers/v2_models.py#L66
[compatibility]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/routers/v2_models.py#L151
[model-list]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/routers/v2_models.py#L165
[model-load]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/routers/v2_models.py#L187
[model-unload]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/routers/v2_models.py#L219
[server-routes]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/routers/v2_server.py
[app]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/app.py#L140
[dispatch]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/framework/dispatch.py#L55
[registry]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/framework/registry.py
[parsers]: https://github.com/roboflow/inference/tree/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/framework/input_parsers
[json-parser]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/framework/input_parsers/json_base64.py#L26
[multipart-parser]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/framework/input_parsers/multipart.py#L16
[input-tests]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/tests/unit_tests/test_v2_infer_input.py
[model-stat]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/framework/model_stat.py
[typed]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_model_manager/inference_model_manager/serializers_typed.py
[od-output]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/handlers/object_detection/output_serializer.py#L30
[od-interface]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/handlers/object_detection/introspection.py
[actions]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_model_manager/inference_model_manager/dispatch.py#L150
[gateway]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/gateway.py#L293
[gateway-resolver]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/gateway_resolver.py
[workflow-routes]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/workflows/router.py
[workflow-provider]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/workflows/models_provider.py#L620
[workflow-execution]: https://github.com/roboflow/inference/blob/3d45b8712cc428eb01b714f3609346be7c92acc4/inference_server/inference_server/workflows/execution.py#L62
[plugin-package]: https://github.com/roboflow/inference-closed-plugins/blob/18f2c22543b499727093cf180d45e835ba1c911f/rf_serving/pyproject.toml
[plugin-ci]: https://github.com/roboflow/inference-closed-plugins/blob/18f2c22543b499727093cf180d45e835ba1c911f/.github/workflows/unit_tests.yml
[mmp]: https://github.com/roboflow/inference-closed-plugins/blob/18f2c22543b499727093cf180d45e835ba1c911f/rf_serving/rf_serving/gateway/mmp_gateway.py#L224
[probe]: probe.py
[observations]: observations.json
[validation]: VALIDATION.md
