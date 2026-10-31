---
name: review-http-api-server
description: Review guidance for PRs touching inference/core/interfaces/http/** (http_api.py, error_handlers.py, orjson_utils.py, dependencies.py, request_metrics.py, middlewares/, builder/, handlers/) or inference_cli/server.py. Also loads on diffs adding a route/middleware, an except arm in with_route_exceptions / with_route_exceptions_async, a new exception in inference/core/exceptions.py needing HTTP mapping, a header constant in constants.py, or a route-gating env flag in inference/core/env.py.
---

# Reviewing http-api-server changes

## Scope
Trigger this skill when a PR touches any of:
- `inference/core/interfaces/http/**` — `http_api.py` (the FastAPI app factory `HttpInterface`, all route registrations, middleware wiring, serverless auth), `error_handlers.py`, `orjson_utils.py`, `dependencies.py`, `request_metrics.py`, `uvicorn_config.py`, `middlewares/{cors,gzip}.py`, `builder/routes.py`, `handlers/workflows.py`.
- `inference_cli/server.py` — the `inference server start|status|stop` typer CLI.
- Companion touch-points this surface owns the contract for: a new exception in `inference/core/exceptions.py` that needs an HTTP mapping; the status-code map in `inference/core/roboflow_api.py`; header constants in `inference/core/constants.py` / `request_metrics.py`; a route-gating env flag in `inference/core/env.py`.

OUT of scope (other skills own these): Workflows execution-engine internals, model implementations under `inference/models/`, stream-manager/pipeline internals, `inference_sdk` client. Only review the HTTP *surface* of those. Serverless/tenant auth semantics are co-owned with `review-topic-auth-and-tenant-security` — that skill owns the fail-open rule; this skill enforces that every route on the surface declares its auth story.

## Review checklist
Severity-tagged. Resolve BLOCK before merge; raise FLAG; NIT is optional.

- **BLOCK** — Serverless auth (`check_authorization_serverless` middleware in `http_api.py`) must fail *closed* on unexpected upstream status and must not cache it. The middleware only branches on 200 / 401 / 402 from `get_serverless_usage_check_async`; any other status must deny (not silently authorize) and must NOT be written into `cached_api_keys`. This exact fix landed then was **reverted** (#2528 → #2529) — treat any change to this middleware, `AuthorizationCacheEntry`, or the `AUTH_CACHE_TTL_SECONDS`/`SHORT_AUTH_CACHE_TTL_SECONDS` TTLs as high-risk; require tests asserting both the status AND that the response was not cached (`await_count`), plus maintainer sign-off.
- **BLOCK** — Every new/modified route states its auth story. A route added under `if not (LAMBDA or GCP_SERVERLESS)` is unreachable serverless; a route reachable serverless is subject to `check_authorization_serverless`. The PR must make explicit which applies and confirm the route isn't an unintended authenticated-surface addition or an unintended auth bypass.
- **BLOCK** — Every new/modified route is wrapped in the correct decorator from `error_handlers.py`: `with_route_exceptions_async` for `async def`, `with_route_exceptions` for sync. Import from `error_handlers`, not `http_api` (#1512).
- **BLOCK** — A new exception surfaced to a route has an explicit `except` arm with an intentional `status_code` in BOTH `with_route_exceptions` and `with_route_exceptions_async`, mirrored — otherwise it falls through to a generic 500 (#2099). Add the `roboflow_api.py` `DEFAULT_ERROR_HANDLERS` entry too if it originates from a Roboflow API call.
- **BLOCK** — `except` arm ordering: specific exceptions precede the broad tuples that would otherwise swallow them (#998, #1104).
- **BLOCK** — Public JSON response shapes stay backward-compatible. No removed/renamed field or alias; `exclude_none=True` must not drop a contractual-but-empty field (#1599). Builder response shapes are Firestore-shaped and externally consumed (#1096/#1365).
- **BLOCK** — Builder file access is path-traversal-safe: `workflow_id` is validated with `re.match(r"^[\w\-]+$", ...)` and the on-disk path is `sha256(workflow_id.encode()).hexdigest()`, never built directly from user input (#1096).
- **FLAG** — New endpoint is gated behind an env flag from `inference/core/env.py` and excluded for `LAMBDA`/`GCP_SERVERLESS` where the Lambda authorizer can't support it (#1557, #1717).
- **FLAG** — New env var / flag / numeric bound is declared in `inference/core/env.py` (not read inline in `http_api.py`), with a sane default and, for tuning knobs, a rationale comment (#1611, #1717).
- **FLAG** — `response_model` type unions use `Union[...]`, not `X | Y` (FastAPI runtime-evals the annotation) (#1599).
- **FLAG** — A new response header is added to the CORS `expose_headers` list in the `PathAwareCORSMiddleware` block of `http_api.py`, and its constant lives in `constants.py` / `request_metrics.py`.
- **FLAG** — Request body/stream is read exactly once, via `parse_body_content_for_legacy_request_handler` in `dependencies.py` (#1518).
- **FLAG** — External numeric inputs (`confidence`, `usage_fps`, `frames`) are guarded before arithmetic: clamp order is correct for percentage-vs-fraction inputs (#1746), and non-numeric values are rejected with `isinstance(x, numbers.Number)` (#795).
- **FLAG** — Tests added/updated under `tests/inference/unit_tests/core/interfaces/http/` for the new status / field / header / branch.
- **NIT** — Exception log level matches severity: `logger.warning(...)` for client-caused/expected (402, missing key), `logger.exception(...)` for server faults; every caught exception is logged, not silently swallowed (#1104).
- **NIT** — CLI options follow `Annotated[..., typer.Option("--x/--no-x", help=...)]` and are threaded into `start_inference_container(...)` (#1024).
- **NIT** — Endpoint/CLI changes update `docs/api.md` / `docs/server_configuration/` / `docs/inference_helpers/`.

### Not blocking
- Do NOT demand an `inference/core/version.py` bump — inference releases are versioned separately from feature/bugfix PRs.
- Do NOT demand CORS `expose_headers` changes for headers that are internal-only or never returned cross-origin.
- Do NOT demand a new env flag for routes that are already unconditionally serverless-excluded and carry no perf/rollout risk.
- Response-header aggregation, Prometheus/GPU metrics endpoints, and serializer swaps are recurring revert magnets (#2222, #721, #724, #190) — ask for justification + tests, but a well-tested change here is not automatically a BLOCK.

## Standards
One canonical statement per rule. The checklist above references these.

- **Exception wrapping.** All routes route their errors through `with_route_exceptions` / `with_route_exceptions_async` in `error_handlers.py`; these decorators ARE the exception→`status_code` contract. The two wrappers must stay mirrored: a new mapped exception needs an arm in both, and specific `except` arms must precede broad tuples.
- **Error→status mapping.** New public errors get an explicit `status_code` in both wrappers; Roboflow-API-originating errors also get an entry in `DEFAULT_ERROR_HANDLERS` in `roboflow_api.py`. Missing arms fall through to 500.
- **Response backward-compat.** Responses are pydantic models serialized by `orjson_response(...)` / `orjson_response_keeping_parent_id(...)` in `orjson_utils.py` with `by_alias=True, exclude_none=True`. Public field/alias set is append-only; contractual-but-empty fields must survive `exclude_none` (use `orjson_response_keeping_parent_id` for `parent_id`).
- **Headers as contract.** Header constants live in `inference/core/constants.py` (`PROCESSING_TIME_HEADER`, `WORKSPACE_ID_HEADER`, …) and `request_metrics.py` (`REMOTE_PROCESSING_TIME_HEADER`). Anything returned cross-origin must be listed in the `expose_headers` of the `PathAwareCORSMiddleware` block in `http_api.py`.
- **Route gating & serverless.** New endpoints are gated by an `inference/core/env.py` flag via `str2bool(os.getenv("FLAG", "True"))` and excluded under `if not (LAMBDA or GCP_SERVERLESS)` when the Lambda authorizer can't carry them (only path params work). Magic numeric bounds become named env constants (e.g. `CONFIDENCE_LOWER_BOUND_OOM_PREVENTION`), not inline literals.
- **Serverless authorization.** The `check_authorization_serverless` middleware in `http_api.py` resolves an `api_key`, keys `cached_api_keys` on `(api_key, enforce_credits_verification)` via `AuthorizationCacheEntry`, and returns via `_authorization_error_response`. It handles 200 (authorize), 401 (deny), 402 (credits). Any other/unexpected upstream status must fail closed and must not be cached. Every route reachable in serverless is governed by this middleware; auth semantics are co-owned with `review-topic-auth-and-tenant-security`.
- **Legacy request parsing.** Legacy `/infer/...` request bodies are read exactly once through `parse_body_content_for_legacy_request_handler` in `dependencies.py`, returning `Optional[Union[bytes, UploadFile]]`.
- **Builder path safety.** `builder/routes.py` validates `workflow_id` (`^[\w\-]+$`) and resolves files by `sha256(...).hexdigest()`; response times are Firestore-shaped `{"_seconds": int(...)}`.
- **CLI.** `inference_cli/server.py` options are typer `Annotated[..., typer.Option("--flag/--no-flag", help=...)]` threaded into `start_inference_container(...)`.

## Key files & entry points
- `inference/core/interfaces/http/http_api.py` — `HttpInterface` app factory; route registrations; `check_authorization_serverless` middleware; `AuthorizationCacheEntry`; CORS `expose_headers` in the `PathAwareCORSMiddleware` block.
- `inference/core/interfaces/http/error_handlers.py` — `with_route_exceptions` / `with_route_exceptions_async`; the exception→status contract.
- `inference/core/interfaces/http/orjson_utils.py` — `orjson_response`, `orjson_response_keeping_parent_id`.
- `inference/core/interfaces/http/dependencies.py` — `parse_body_content_for_legacy_request_handler` (legacy body/multipart parsing).
- `inference/core/interfaces/http/builder/routes.py` — builder UI routes (path safety, Firestore-shaped responses).
- `inference/core/interfaces/http/middlewares/{cors,gzip}.py`, `request_metrics.py`, `uvicorn_config.py`.
- `inference/core/exceptions.py`, `inference/core/env.py`, `inference/core/constants.py`, `inference/core/roboflow_api.py` (`DEFAULT_ERROR_HANDLERS`).
- `inference_cli/server.py` — CLI. Tests: `tests/inference/unit_tests/core/interfaces/http/`.

## Reference PRs
- [#2528](https://github.com/roboflow/inference/pull/2528) / [#2529](https://github.com/roboflow/inference/pull/2529) — bugfix + revert: serverless-auth fail-closed on unexpected upstream status, no caching — high-risk area.
- [#2099](https://github.com/roboflow/inference/pull/2099) — `PaymentRequiredError` → 402 mirrored across sync+async handlers + `roboflow_api.py` status map.
- [#1557](https://github.com/roboflow/inference/pull/1557) — gate `GET /model/registry` behind `GET_MODEL_REGISTRY_ENABLED`.
- [#1717](https://github.com/roboflow/inference/pull/1717) — shared workflows thread pool behind `HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_*` flags.
- [#1512](https://github.com/roboflow/inference/pull/1512) — async handlers import `with_route_exceptions_async` from `error_handlers`.
- [#1611](https://github.com/roboflow/inference/pull/1611) — reject confidence 0 via `CONFIDENCE_LOWER_BOUND_OOM_PREVENTION` env const.
- [#1746](https://github.com/roboflow/inference/pull/1746) — `elif` skipped confidence lower-bound clamp for percentage inputs.
- [#1518](https://github.com/roboflow/inference/pull/1518) — multipart body read twice; centralized parsing in `dependencies.py`.
- [#1599](https://github.com/roboflow/inference/pull/1599) — retain empty `parent_id` in OCR response; `Union[...]` response_model.
- [#998](https://github.com/roboflow/inference/pull/998) — `WorkflowSyntaxError` needs its own `except` arm.
- [#1104](https://github.com/roboflow/inference/pull/1104) — versionless legacy model ids; log every caught exception.
- [#795](https://github.com/roboflow/inference/pull/795) — guard non-numeric `usage_fps`/`frames` before arithmetic.
- [#1096](https://github.com/roboflow/inference/pull/1096) / [#1365](https://github.com/roboflow/inference/pull/1365) — builder path-traversal fix (regex + sha256) and Firestore-shaped times.
- [#1024](https://github.com/roboflow/inference/pull/1024) — `--metrics-enabled/--metrics-disabled` CLI flag threaded to container adapter.
- [#2222](https://github.com/roboflow/inference/pull/2222) / [#721](https://github.com/roboflow/inference/pull/721) / [#724](https://github.com/roboflow/inference/pull/724) / [#190](https://github.com/roboflow/inference/pull/190) — historically-reverted header-aggregation / metrics / orjson swaps.

## Related topic skills
When the PR also exhibits these cross-cutting concerns, load the matching topic skill too:
- `review-topic-auth-and-tenant-security` (owns the serverless-auth fail-open rule)
- `review-topic-backward-compat-and-versioning`
- `review-topic-external-contract-and-silent-fallback`
- `review-topic-test-hygiene`
