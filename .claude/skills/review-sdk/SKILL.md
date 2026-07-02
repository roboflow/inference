---
name: review-sdk
description: Review guidance for PRs changing inference_sdk/ (HTTP client, webrtc client, entities, errors, request/response utils) — enforces the InferenceHTTPClient public-API contract, sync/async parity, API-key redaction, error-wrapping, client-mode gating, and companion version/docs/test requirements.
---

# Reviewing sdk changes

## Scope
Triggers on any PR touching:
- `inference_sdk/**` — the pip-installable `inference-sdk` package: `http/client.py` (the `InferenceHTTPClient`), `http/entities.py`, `http/errors.py`, `http/utils/**` (encoding, loaders, requests, executors, request_building, post_processing, aliases), `config.py`, `webrtc/**`.
- `tests/inference_sdk/**` — unit/integration/e2e tests for the above.
- `docs/inference_helpers/inference_sdk*` and `docs/inference_sdk/**` — SDK reference docs.
- `.release/pypi/inference.sdk.setup.py`, `requirements/requirements.sdk.*.txt`.

NOTE the legacy name: this package used to be `inference_client/` (renamed in #85). There is **no** `inference_client/` directory in the current tree — treat any such path as stale.

OUT of scope (other skills): server-side model code under `inference/models/**` and `inference/core/**` (even when a PR touches both, e.g. #212 added core-model endpoints + SDK methods together — review only the `inference_sdk/` side here), Workflows blocks, and `inference_cli/`.

## What this surface is
`inference_sdk` is a **thin, dependency-light HTTP/WebRTC client** that talks to a Roboflow inference server (local or hosted) — it must not import heavy server/model code. The contracts a reviewer protects:

- **`InferenceHTTPClient` is the public API.** Constructed via `InferenceHTTPClient(api_url=..., api_key=...)` or `.init(...)`. `client_mode` (`HTTPClientMode.V0`/`V1`) is auto-derived from `api_url` by `_determine_client_mode` (`ALL_ROBOFLOW_API_URLS` → V0/legacy hosted routes; everything else → V1). Public state: `inference_configuration`, `client_mode`, `selected_model`.
- **Every public inference method ships a sync + async pair** (`infer`/`infer_async`, `ocr_image`/`ocr_image_async`, `clip_compare`/`clip_compare_async`, `sam2_segment_image`/`_async`, …). ~23 async methods mirror the sync surface (`inference_sdk/http/client.py`). A new method without its async sibling is a defect.
- **Errors are a stable public taxonomy** (`inference_sdk/http/errors.py`): `HTTPClientError` (base) with subclasses `HTTPCallErrorError`, `InvalidInputFormatError`, `InvalidModelIdentifier`, `ModelNotInitializedError`, `ModelTaskTypeNotSupportedError`, `ModelNotSelectedError`, `APIKeyNotProvided`, `EncodingError`, `WrongClientModeError`, `InvalidParameterError`, `FeatureDeprecatedError`, plus `RetryError`. Callers catch these; renaming/removing one is a breaking change.
- **`InferenceConfiguration`** (`http/entities.py`) is a `frozen=True` dataclass whose fields map to server query/body params via `to_*_parameters()` methods (V0 legacy vs V1 per-task). The internal→external name mapping tables are the wire contract.
- **`ImagesReference = Union[np.ndarray, PIL.Image.Image, str]`** is the accepted input type across the client; loaders/encoders convert to base64 for the wire.
- **API keys must never leak** in exceptions, logs, or `response.url` (see Pitfalls).

## Standards enforced here
- **Every new sync client method needs an async twin, both `@wrap_errors`/`@wrap_errors_async` decorated.** The sync decorator maps `RetryError`/`HTTPError`/`ConnectionError`; the async one maps `ClientResponseError`/`ClientConnectionError` (`client.py` lines ~121-173). #255 was a one-line bugfix precisely because `infer_async` was wrongly wrapped with the sync `@wrap_errors` instead of `@wrap_errors_async`.
- **Never call bare `response.raise_for_status()`.** Use `api_key_safe_raise_for_status(response=...)` (`http/utils/requests.py`) so the key is stripped from `response.url` before raising. #212 flipped a `response.raise_for_status()` to `api_key_safe_raise_for_status`.
- **Any string derived from an error/URL that surfaces to the user must pass through `deduct_api_key_from_string(...)`** — applies to `description`, `api_message`, and connection-error messages (#248, #255).
- **V1-only endpoints must gate with `self.__ensure_v1_client_mode()`** (raises `WrongClientModeError`) before building the request — core-model methods (clip/cogvlm/sam/doctr/gaze/lmm, load/list models) all do this (`client.py`, e.g. lines ~722, 970; CogVLM comment "Lambda does not support CogVLM"). Docstrings for these methods must list `WrongClientModeError` under `Raises:`.
- **New endpoint methods follow the established shape**: resolve model alias (`resolve_roboflow_model_alias`), load/encode input via `load_static_inference_input` (+ `_async`), build payload with `inject_images_into_payload` / `inject_nested_batches_of_images_into_payload`, apply config via `InferenceConfiguration.to_*_parameters`, post, then `api_key_safe_raise_for_status`. See the core-model methods added in #212.
- **`InferenceConfiguration` field additions must be wired into the relevant `to_*_parameters()` mapping table AND documented.** #1521 added `workflow_run_retries_enabled` to the dataclass, defaulted it from a `config.py` env-flag (`WORKFLOW_RUN_RETRIES_ENABLED` via `str2bool`), and documented it in `docs/inference_helpers/inference_sdk.md`. Fields are `Optional[...] = None` unless a hard default is intended; the dataclass is `frozen=True`.
- **Deprecation is signalled, not silently broken.** Use `@deprecated(reason=...)` / `@experimental(info=...)` (`utils/decorators.py`) — they emit `InferenceSDKDeprecationWarning`, gated by `INFERENCE_WARNINGS_DISABLED`. Hard removals raise `FeatureDeprecatedError(feature=..., reason=..., removal_release=..., replacement=...)`, not a bare `ValueError` (see `test_detect_gazes_deprecated.py`). `infer_from_workflow` is the canonical `@deprecated` example (superseded by `run_workflow`).
- **New public symbols must be exported.** Top-level re-exports in `inference_sdk/__init__.py` (`InferenceHTTPClient`, `InferenceConfiguration`, `VisualisationResponseFormat`); WebRTC classes in both the `from .sources import ...` block AND `__all__` of `inference_sdk/webrtc/__init__.py` (#2200 added `LocalStreamSource` to both).
- **Type-hinted public signatures + Google-style docstrings** with `Args:`/`Returns:`/`Raises:` on every public method (see any method in `client.py`). Keep `ImagesReference` as the image param type.
- **Keep the SDK dependency-light.** Runtime deps are pinned in `requirements/requirements.sdk.http.txt` and `requirements.sdk.webrtc.txt`; a new third-party import needs a corresponding pin. Do not import `inference.core.*` heavy modules into `inference_sdk` (only lightweight cross-refs like a shared exception, as #1521 imported one server exception on the *server* side, not into the SDK).

## Required companions
- **Version bump: `inference/core/version.py`** (`__version__`). The `inference-sdk` package derives its version from there — `.release/pypi/inference.sdk.setup.py` copies `inference/core/version.py` → `inference_sdk/version.py` at build time and imports `__version__` from core. Every merged SDK PR here bumps it (#248 `rc17→rc19`, #255 `rc21→rc22`, #1521 `0.54.1→0.54.2`). Do NOT hand-edit `inference_sdk/version.py` — it is generated. Block a behavioural SDK change with no `inference/core/version.py` bump.
- **Tests under `tests/inference_sdk/unit_tests/**`.** #212 added/updated `http/test_client.py`, `test_entities.py`, and every `http/utils/test_*.py`; #255/#248 (key redaction) belong in `http/utils/test_requests.py`; config/env-flag changes → `unit_tests/test_config.py`; deprecations → a dedicated test like `test_detect_gazes_deprecated.py`; WebRTC → `unit_tests/webrtc/` (+ `integration_tests/`/`e2e_tests/webrtc/`). A new method with no test is blocked.
- **Docs.** New/changed public methods, config fields, or model helpers must update `docs/inference_helpers/inference_sdk.md` and/or the sub-pages (`inference_sdk/core_models.md`, `configuration.md`, `workflows.md`, `model_management.md`). #212 updated `docs/foundation/*.md` + the SDK http_client page; #1521 documented the new config field.
- **Requirements pins** in `requirements/requirements.sdk.http.txt` / `.webrtc.txt` for any new runtime dependency.
- There is **no dedicated SDK changelog file** — the version bump + docs are the changelog surface here.

## Common pitfalls & past regressions
- **#255 — wrong error decorator on an async method.** `infer_async` was decorated `@wrap_errors` (sync) instead of `@wrap_errors_async`; async exceptions (`ClientResponseError`) escaped un-redacted/un-wrapped. Check: every `async def` public method uses `@wrap_errors_async`.
- **#248 — API key leaked via `api_message`.** `api_message=error.message` passed the raw aiohttp message (containing `api_key=...`) straight through; fix wrapped it in `deduct_api_key_from_string`. Check: no error field, log line, or re-raised URL carries an un-redacted key.
- **#255 (broader) — key leaking in errors generally.** Any new code path that stringifies an exception, URL, or request into a user-facing message must redact. Grep the diff for `str(error)`, `.message`, `response.url`, `raise_for_status` without the safe wrapper.
- **#1521 — transient Workflow failures weren't retried / weren't surfaced cleanly.** `_run_workflow` was switched from raw `requests.post` + `raise_for_status` to `send_post_request(..., enable_retries=...)`, and `RetryError` added to `wrap_errors`. Check: workflow/HTTP calls that can be retried honour `workflow_run_retries_enabled` and that `RetryError` is mapped to `HTTPCallErrorError`/`HTTPClientError`.
- **Missing async sibling / missing V1 gate.** New V1-only endpoint added without `__ensure_v1_client_mode()` will produce confusing 404s against hosted V0 URLs instead of a clear `WrongClientModeError`.
- **Silent hard-removal.** Removing a helper by making it raise `ValueError` (instead of `FeatureDeprecatedError`, or `@deprecated` if it should still work) breaks callers without a migration signal — MEMORY note: only specific helpers escalate to `FeatureDeprecatedError`; don't over- or under-apply it.
- **Editing generated `inference_sdk/version.py`** instead of `inference/core/version.py` — the edit is overwritten at build time and the real package version won't move.

## Review checklist
1. Does a behavioural change bump `inference/core/version.py`? (block if not) — and is `inference_sdk/version.py` left untouched (generated)?
2. New sync client method → is there a matching `_async` method, and are both decorated with the correct `@wrap_errors` / `@wrap_errors_async`?
3. Any new HTTP call: uses `api_key_safe_raise_for_status`, not bare `raise_for_status`? All user-facing error/URL strings pass through `deduct_api_key_from_string`?
4. V1-only endpoint: calls `self.__ensure_v1_client_mode()` and documents `WrongClientModeError` in the docstring?
5. New/changed error class: subclasses `HTTPClientError`, and no existing error class renamed/removed (backward-compat)?
6. `InferenceConfiguration` field: added to the right `to_*_parameters()` mapping(s), correct `Optional`/default, dataclass still `frozen=True`, and documented?
7. Deprecation/removal uses `@deprecated`/`@experimental`/`FeatureDeprecatedError` appropriately (warning vs hard error), not a bare exception?
8. New public symbol exported in `inference_sdk/__init__.py` (and `webrtc/__init__.py` `__all__` for WebRTC)?
9. Tests added/updated under `tests/inference_sdk/unit_tests/**` covering the new path (client, entities, utils, redaction, deprecation, or webrtc as applicable)?
10. Docs updated under `docs/inference_helpers/inference_sdk*` for any public API/config/model change?
11. New runtime dependency pinned in `requirements/requirements.sdk.*.txt`? No heavy `inference.core`/model imports pulled into the SDK?
12. Public signatures fully type-hinted with Google-style docstrings; images typed as `ImagesReference`?

## Key files & entry points
- `inference_sdk/http/client.py` — `InferenceHTTPClient`, `wrap_errors`/`wrap_errors_async`, `_determine_client_mode`, all endpoint methods.
- `inference_sdk/http/errors.py` — public exception taxonomy.
- `inference_sdk/http/entities.py` — `InferenceConfiguration`, `HTTPClientMode`, `ImagesReference`, param-mapping tables.
- `inference_sdk/http/utils/requests.py` — `api_key_safe_raise_for_status`, `deduct_api_key_from_string`, image payload injection.
- `inference_sdk/http/utils/{loaders,encoding,executors,request_building,post_processing,aliases}.py` — input loading, `send_post_request`/retries, model-alias resolution.
- `inference_sdk/config.py` — env flags (`WORKFLOW_RUN_RETRIES_ENABLED`, WebRTC timeouts), `InferenceSDKDeprecationWarning`.
- `inference_sdk/utils/decorators.py` — `deprecated`/`experimental`.
- `inference_sdk/webrtc/{sources,session,client,config}.py` — WebRTC streaming API.
- `.release/pypi/inference.sdk.setup.py` — proves version derives from `inference/core/version.py`.
- `docs/inference_helpers/inference_sdk.md` (+ `inference_sdk/` sub-pages).

## Reference PRs
- [#212](https://github.com/roboflow/inference/pull/212) — feature: extend SDK client to (almost) all core models (clip/cogvlm/sam/doctr/gaze); canonical new-endpoint shape + V1 gate + tests + docs.
- [#2200](https://github.com/roboflow/inference/pull/2200) — feature: WebRTC `LocalStreamSource` local stream processing; export in `webrtc/__init__.py` `__all__` + example.
- [#1521](https://github.com/roboflow/inference/pull/1521) — feature: workflow retries via SDK; `InferenceConfiguration.workflow_run_retries_enabled`, `RetryError` mapping in `wrap_errors`, `send_post_request`, docs + version bump.
- [#255](https://github.com/roboflow/inference/pull/255) — security/bugfix: API key leaking in errors; wrong `@wrap_errors` (sync) on `infer_async` → `@wrap_errors_async`.
- [#248](https://github.com/roboflow/inference/pull/248) — bugfix: redact API key from `api_message` via `deduct_api_key_from_string`.
- [#85](https://github.com/roboflow/inference/pull/85) — refactor: rename `inference_client/` → `inference_sdk/` (package, tests, requirements, release setup, docs, mkdocs).
- [#70](https://github.com/roboflow/inference/pull/70) — feature: baseline HTTP client for the inference server (original `InferenceHTTPClient`).
- [#1417](https://github.com/roboflow/inference/pull/1417) — WIP/DO-NOT-MERGE: pass images as numpy arrays; reference for `ImagesReference` handling direction (not merged).

## Related topic skills

When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):

- `review-topic-backward-compat-and-versioning`
- `review-topic-external-contract-and-silent-fallback`
- `review-topic-test-hygiene`
