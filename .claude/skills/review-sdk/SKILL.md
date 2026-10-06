---
name: review-sdk
description: >-
  Load for PRs touching inference_sdk/** (http/client.py InferenceHTTPClient, http/entities.py,
  http/errors.py, http/utils/**, config.py, webrtc/**), tests/inference_sdk/**,
  docs/inference_helpers/inference_sdk*, .release/pypi/inference.sdk.setup.py, or
  requirements/requirements.sdk.*.txt. Diff signals: a new client method, a new/renamed
  HTTPClientError subclass, a new InferenceConfiguration field, response.raise_for_status,
  api_key redaction, @wrap_errors_async, or __ensure_v1_client_mode.
---

# Reviewing sdk changes

## Scope
PRs touching:
- `inference_sdk/**` — the pip-installable `inference-sdk` package: `http/client.py`, `http/entities.py`, `http/errors.py`, `http/utils/**`, `config.py`, `webrtc/**`.
- `tests/inference_sdk/**`, `docs/inference_helpers/inference_sdk*` (top-level `inference_sdk.md` + `inference_sdk/` sub-pages), `.release/pypi/inference.sdk.setup.py`, `requirements/requirements.sdk.*.txt`.

Legacy name: the package was `inference_client/` (renamed in #85). Treat any `inference_client/` path as stale.

OUT of scope (other skills): server-side model code under `inference/models/**` and `inference/core/**` (review only the `inference_sdk/` side even in mixed PRs like #212), Workflows blocks, and `inference_cli/`.

`inference_sdk` is a **thin, dependency-light HTTP/WebRTC client** for a Roboflow inference server. It must not import heavy `inference.core.*` / model code.

## Review checklist
Severity tags: **BLOCK** (fix before merge) / **FLAG** (raise it) / **NIT** (optional).

- **BLOCK** — `inference_sdk/version.py` left untouched (it is generated at build time; never hand-edited).
- **BLOCK** — New HTTP call uses `api_key_safe_raise_for_status`, never bare `response.raise_for_status()`; every user-facing error/URL string passes through `deduct_api_key_from_string`.
- **BLOCK** — New sync client method has a matching `_async` sibling, each decorated with the correct `@wrap_errors` (sync) / `@wrap_errors_async` (async).
- **BLOCK** — V1-only endpoint calls `self.__ensure_v1_client_mode()` before building the request.
- **BLOCK** — No existing `HTTPClientError` subclass renamed/removed (public taxonomy; callers catch these).
- **FLAG** — `InferenceConfiguration` field wired into the right `to_*_parameters()` mapping(s), correct `Optional`/default, dataclass still `frozen=True`.
- **FLAG** — Deprecation/removal uses `@deprecated` / `@experimental` / `FeatureDeprecatedError` (warning vs hard error), not a bare `ValueError`.
- **FLAG** — New public symbol exported in `inference_sdk/__init__.py` (and `webrtc/__init__.py` `__all__` for WebRTC).
- **FLAG** — Tests added/updated under `tests/inference_sdk/unit_tests/**` for the new path.
- **FLAG** — New runtime dependency pinned in `requirements/requirements.sdk.*.txt`; no heavy `inference.core`/model import pulled into the SDK.
- **FLAG** — Sync HTTP path preserves per-thread `requests.Session` reuse (does not reintroduce a per-call `Session()` or force single requests through a `ThreadPoolExecutor`).
- **NIT** — Docs updated under `docs/inference_helpers/inference_sdk*` for public API/config/model changes.
- **NIT** — Public signatures fully type-hinted with Google-style docstrings; image params typed as `ImagesReference`; `WrongClientModeError` listed under `Raises:` for V1-gated methods.

### Not blocking
- Do NOT demand an `inference/core/version.py` bump — inference releases are versioned separately from feature/bugfix PRs.
- A pure-internal refactor (no public method/error/config/signature change and no behavioural change) does not require docs or new tests — only that existing tests still pass.
- A method that is intrinsically V1-only *and* has no V0 route does not need a V0 fallback; the `WrongClientModeError` gate is the correct behaviour, not a gap.
- Cosmetic docstring/typing nits (last two items) never block a merge on their own.

## Standards

- **Sync + async pairing.** Every public inference method ships a sync + async twin (`infer`/`infer_async`, `ocr_image`/`ocr_image_async`, `clip_compare`/`clip_compare_async`, `sam2_segment_image`/`_async`, …). The sync `wrap_errors` maps `RetryError`/`HTTPError`/`ConnectionError`; the async `wrap_errors_async` maps `ClientResponseError`/`ClientConnectionError`. A sync method decorated `@wrap_errors_async` (or vice-versa) lets un-redacted async exceptions escape (broke in #255; that PR was a one-line decorator swap on `infer_async`).

- **API keys never leak.** Any string derived from an error, URL, or request that surfaces to the user must pass through `deduct_api_key_from_string` (`http/utils/requests.py`) — `description`, `api_message`, connection-error messages (raw `api_message=error.message` leaked the key in #248). Never call bare `response.raise_for_status()`; use `api_key_safe_raise_for_status(response=...)`, which strips the key from `response.url` before raising (#212). Grep the diff for `str(error)`, `.message`, `response.url`, and `raise_for_status` without the safe wrapper.

- **Client-mode gating.** `client_mode` (`HTTPClientMode.V0`/`V1`) is auto-derived from `api_url` by `_determine_client_mode` (`ALL_ROBOFLOW_API_URLS` → V0 legacy hosted; else V1). V1-only endpoints (clip/cogvlm/sam/doctr/gaze/lmm, load/list models) must gate with `self.__ensure_v1_client_mode()` (raises `WrongClientModeError`) before building the request, and list it under `Raises:`. Skipping the gate produces confusing 404s against a V0 URL instead of a clear error.

- **New-endpoint shape.** Resolve alias (`resolve_roboflow_model_alias`), load/encode input via `load_static_inference_input` (+ `_async`), inject via `inject_images_into_payload` / `inject_nested_batches_of_images_into_payload`, apply config via `InferenceConfiguration.to_*_parameters`, post, then `api_key_safe_raise_for_status` (canonical example: the core-model methods in #212).

- **`InferenceConfiguration` is the wire contract.** `frozen=True` dataclass (`http/entities.py`); fields map to server query/body params via `to_*_parameters()` (V0 legacy vs V1 per-task). A field addition must be wired into the relevant mapping table, defaulted `Optional[...] = None` unless a hard default is intended, and documented — #1521 added `workflow_run_retries_enabled`, defaulted it from the `config.py` env flag `WORKFLOW_RUN_RETRIES_ENABLED` (via `str2bool`), and documented it.

- **Error taxonomy is public.** `http/errors.py`: `HTTPClientError` (base) with subclasses incl. `HTTPCallErrorError`, `WrongClientModeError`, `APIKeyNotProvided`, `FeatureDeprecatedError`, plus the standalone `RetryError`. Renaming/removing one is a breaking change. New errors subclass `HTTPClientError`.

- **Deprecation is signalled, not silently broken.** `@deprecated(reason=...)` / `@experimental(info=...)` (`utils/decorators.py`) emit `InferenceSDKDeprecationWarning`, gated by `INFERENCE_WARNINGS_DISABLED`. A hard removal raises `FeatureDeprecatedError(feature=..., reason=..., removal_release=..., replacement=...)`, not a bare `ValueError` (`test_detect_gazes_deprecated.py`). `infer_from_workflow` is the canonical `@deprecated` case (superseded by `run_workflow`). MEMORY: only specific helpers escalate to `FeatureDeprecatedError` — don't over/under-apply.

- **HTTP session reuse + timeouts (sync path).** The sync executor keeps a per-thread `requests.Session` via `_get_thread_local_requests_session` and runs a single-request package on the caller thread (`make_parallel_requests` short-circuits when `len==1`, avoiding a one-off `ThreadPoolExecutor`) so sequential workflow frames reuse TCP/TLS connections. `requests.Session` is **not** thread-safe, so parallel requests stay isolated per worker thread and reset the session on exit (`_reset_thread_local_requests_session`). A change that reintroduces a per-call `Session()`, shares one `Session` across threads, or routes single requests through the pool regresses this (#2538 recovered ~71% sequential FPS by reusing sync sessions). Retries flow through `send_post_request(..., enable_retries=...)`, which maps `RetryError` → `HTTPCallErrorError` (#1521). Note the async path still opens a fresh `aiohttp.ClientSession()` per call — do not assume it reuses connections.

- **Exports.** Top-level re-exports in `inference_sdk/__init__.py` (`InferenceHTTPClient`, `InferenceConfiguration`, `VisualisationResponseFormat`); WebRTC classes in both the `from .sources import ...` block and `__all__` of `inference_sdk/webrtc/__init__.py` (#2200 added `LocalStreamSource` to both).

- **Dependency-light.** Runtime deps pinned in `requirements/requirements.sdk.http.txt` / `.webrtc.txt`; a new third-party import needs a pin. Do not import heavy `inference.core.*`/model modules into `inference_sdk`.

- **Signatures + docstrings.** Type-hinted public signatures with Google-style `Args:`/`Returns:`/`Raises:`. Image params typed `ImagesReference = Union[np.ndarray, PIL.Image.Image, str]`.

## Required companions
- **Versioning:** no `inference/core/version.py` bump is required (release-time concern). `.release/pypi/inference.sdk.setup.py` copies it → `inference_sdk/version.py` at build time; never hand-edit the generated file.
- **Tests:** `tests/inference_sdk/unit_tests/**` — `http/test_client.py`, `test_entities.py`, `http/utils/test_*.py` (key redaction → `test_requests.py`; session reuse → `http/utils/test_executors.py`), `test_config.py` for env-flags, a dedicated file for deprecations, `webrtc/` for WebRTC.
- **Docs:** `docs/inference_helpers/inference_sdk.md` and/or sub-pages (`inference_sdk/core_models.md`, `configuration.md`, `workflows.md`, `model_management.md`).
- **Requirements:** a pin in `requirements/requirements.sdk.http.txt` / `.webrtc.txt` for any new runtime dependency.
- There is **no dedicated SDK changelog** — docs are the changelog surface.

## Key files & Reference PRs
- `inference_sdk/http/client.py` — `InferenceHTTPClient`, `wrap_errors`/`wrap_errors_async`, `_determine_client_mode`, `__ensure_v1_client_mode`, endpoint methods.
- `inference_sdk/http/errors.py` — exception taxonomy. `inference_sdk/http/entities.py` — `InferenceConfiguration`, `HTTPClientMode`, `ImagesReference`, param-mapping tables.
- `inference_sdk/http/utils/requests.py` — `api_key_safe_raise_for_status`, `deduct_api_key_from_string`.
- `inference_sdk/http/utils/executors.py` — `_get_thread_local_requests_session`, `make_parallel_requests`, `send_post_request`.
- `inference_sdk/http/utils/{loaders,encoding,request_building,aliases}.py`, `inference_sdk/config.py`, `inference_sdk/utils/decorators.py`, `inference_sdk/webrtc/{sources,session,client,config}.py`.
- `.release/pypi/inference.sdk.setup.py`, `docs/inference_helpers/inference_sdk.md` (+ `inference_sdk/` sub-pages).

Reference PRs:
- [#212](https://github.com/roboflow/inference/pull/212) — core-model client methods; canonical new-endpoint shape + V1 gate + tests + docs.
- [#2538](https://github.com/roboflow/inference/pull/2538) — perf: per-thread `requests.Session` reuse in the sync executor (~71% sequential FPS gain).
- [#1521](https://github.com/roboflow/inference/pull/1521) — workflow retries; `workflow_run_retries_enabled`, `RetryError` mapping, `send_post_request`.
- [#255](https://github.com/roboflow/inference/pull/255) — API key leak; wrong `@wrap_errors` (sync) on `infer_async`.
- [#248](https://github.com/roboflow/inference/pull/248) — redact API key from `api_message`.
- [#2200](https://github.com/roboflow/inference/pull/2200) — WebRTC `LocalStreamSource` export.
- [#85](https://github.com/roboflow/inference/pull/85) — rename `inference_client/` → `inference_sdk/`.
- [#70](https://github.com/roboflow/inference/pull/70) — baseline HTTP client.

## Related topic skills
Load the matching topic skill when the PR also shows these concerns:
- `review-topic-backward-compat-and-versioning`
- `review-topic-external-contract-and-silent-fallback`
- `review-topic-test-hygiene`
