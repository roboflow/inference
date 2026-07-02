---
name: review-http-api-server
description: Review guidance for PRs changing inference/core/interfaces/http/** (http_api.py, error_handlers.py, orjson_utils.py, dependencies.py, middlewares/, builder/, handlers/) or inference_cli/server.py — enforces HTTP API server standards, response/header/error contracts, env-flag gating, and required version-bump + test companions.
---

# Reviewing http-api-server changes

## Scope
Trigger this skill when a PR touches any of:
- `inference/core/interfaces/http/**` — `http_api.py` (the FastAPI app factory `HttpInterface`), `error_handlers.py`, `orjson_utils.py`, `dependencies.py`, `request_metrics.py`, `uvicorn_config.py`, `middlewares/` (cors, gzip), `builder/routes.py`, `handlers/workflows.py`
- `inference_cli/server.py` — the `inference server start|status|stop` typer CLI
- Companion touch-points that this surface owns the contract for: new exceptions in `inference/core/exceptions.py` that need an HTTP mapping, HTTP status mapping in `inference/core/roboflow_api.py`, header constants in `inference/core/constants.py`.

OUT of scope (other skills own these): Workflows execution-engine internals, model implementations under `inference/models/`, stream-manager/pipeline internals, `inference_sdk` client. Only review the HTTP *surface* of those.

## What this surface is
`http_api.py` builds one FastAPI `app` in the `HttpInterface` constructor. Routes are registered inline, heavily gated by env flags (`LAMBDA`, `GCP_SERVERLESS`, `LEGACY_ROUTE_ENABLED`, `CORE_MODEL_*_ENABLED`, `ENABLE_BUILDER`, `ENABLE_STREAM_API`, `GET_MODEL_REGISTRY_ENABLED`). Contracts a reviewer must protect:
- **Public JSON response shapes are backward-compatible.** Responses are pydantic models serialized via `orjson_response(...)`/`orjson_response_keeping_parent_id(...)` in `orjson_utils.py` with `by_alias=True, exclude_none=True`. Removing/renaming a field, or dropping a field that clients rely on, is a breaking change (see #1599 — `parent_id` must be retained even when empty).
- **Error → HTTP status mapping is centralized** in `error_handlers.py` via the `with_route_exceptions` (sync) and `with_route_exceptions_async` (async) decorators. Every route must be wrapped by one of them; the mapping of exception → `status_code` is the API's error contract.
- **Response headers are a contract.** Constants live in `inference/core/constants.py` (`PROCESSING_TIME_HEADER="X-Processing-Time"`, `WORKSPACE_ID_HEADER="X-Workspace-Id"`, etc.) and `request_metrics.py` (`REMOTE_PROCESSING_TIME_HEADER`). Any header returned cross-origin must also appear in the CORS `expose_headers` list (`http_api.py` ~L649-663).
- **Legacy `/infer/...` routes** and serverless/Lambda auth middleware have Lambda-authorizer constraints (only path params work) — new endpoints often must be excluded under `if not (LAMBDA or GCP_SERVERLESS)`.

## Standards enforced here
- **Every route wrapped in the right exception decorator.** Async routes use `with_route_exceptions_async`, sync use `with_route_exceptions`. Mismatched wrapping is a bug — #1512 fixed `builder/routes.py` to import `with_route_exceptions_async` from `error_handlers` (not from `http_api`) for its `async def` routes.
- **New public error → add a mapped handler with an explicit `status_code`.** Adding an exception without an `except` arm in BOTH the sync and async wrappers means it falls through to a generic 500. #2099 added `PaymentRequiredError` → 402 in both wrappers of `error_handlers.py`, plus the `402:` entry in `roboflow_api.py`'s status-code lambda map, plus workflow step-error handling. Handlers of the same error must be mirrored across sync+async arms.
- **Type unions use `Union[...]`, not `X | Y`, in `response_model=`.** #1599 changed `OCRInferenceResponse | List[...]` to `Union[OCRInferenceResponse, List[...]]` (runtime-eval compatibility for FastAPI response_model).
- **New endpoints are gated behind an env flag defined in `inference/core/env.py`** using `str2bool(os.getenv("FLAG", "True"))`. #1557 added `GET_MODEL_REGISTRY_ENABLED` and gated `GET /model/registry` with `if not LAMBDA and GET_MODEL_REGISTRY_ENABLED`. Perf toggles follow the same pattern (#1717 `HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_ENABLED`/`_WORKERS`).
- **Magic numeric bounds become named env constants**, not inline literals. #1611 replaced inline `confidence = 0` with `CONFIDENCE_LOWER_BOUND_OOM_PREVENTION` (env-configurable, default `0.01`) with a comment explaining the OOM rationale.
- **Exception log level signals severity.** In `error_handlers.py`: client-caused/expected conditions (402, missing key) use `logger.warning(...)`; unexpected server faults use `logger.exception(...)`. #1104 also added `logger.error("%s: %s", type(error).__name__, error)` to previously-silent `except` arms and standardized the caught var to `error`.
- **Builder file access must be path-traversal-safe.** `builder/routes.py` validates `workflow_id` with `re.match(r"^[\w\-]+$", ...)` and resolves the on-disk file via `sha256(workflow_id.encode()).hexdigest()` (#1096, code-scanning fix). Never build a path directly from user input.
- **Request-body parsing for legacy routes goes through `dependencies.py`.** The request body/stream must be read exactly once; multipart form is parsed there and returns `Optional[Union[bytes, UploadFile]]` (#1518).
- **CLI options are typer `Annotated[..., typer.Option(...)]` with a `--flag/--no-flag` form and a help string**, and must be threaded into `start_inference_container(...)` (#1024 `--metrics-enabled/--metrics-disabled`).

## Required companions
Block the PR if a change on this surface lands without its companion:
- **Version bump:** `inference/core/version.py` (`__version__`) is bumped on nearly every merged behavior change here (#2528, #1611, #1557, #1593). A functional change with no version bump is suspect — flag it.
- **Env var registration:** any new `os.getenv(...)` flag/const must be declared in `inference/core/env.py` (not read inline in `http_api.py`) — #1557, #1611, #1717, #1512.
- **Tests** under `tests/inference/unit_tests/core/interfaces/http/`: `test_http_api.py`, `test_error_handlers.py`, `test_orjson_utils.py`, `test_cors.py`, `test_model_response_headers.py`, `test_remote_processing_time_middleware.py`, `test_legacy_http_route_accepts_confidence_modes.py`, `test_builder.py`, `handlers/test_workflows.py`. New error mapping → assert the status_code (#2528 added a fail-closed 503 test); new response field → assert its presence/shape; middleware/header change → header assertion.
- **CORS expose_headers:** any new response header returned to browsers must be added to the `expose_headers` list in the `PathAwareCORSMiddleware` block of `http_api.py`.
- **Docs:** endpoint additions/changes should update `docs/api.md` and, for CLI, `docs/server_configuration/` / `docs/inference_helpers/`.

## Common pitfalls & past regressions
- **#1746** — `elif confidence < LOWER_BOUND` after an `if confidence >= 1: confidence /= 100` skipped the lower-bound clamp for percentage-form inputs. Chained `if/elif` on normalized numeric inputs: verify each branch independently applies. Check `test_legacy_http_route_accepts_confidence_modes.py`.
- **#1518** — multipart request body read twice: parsing content-type inline AND letting FastAPI re-read the stream corrupted the request. Confirm the body/stream is consumed exactly once, via `dependencies.py`.
- **#1599** — `exclude_none=True` silently dropped `parent_id` from OCR responses, breaking clients. When a field can be legitimately empty/None but is part of the contract, use `orjson_response_keeping_parent_id` (or equivalent) so it stays in the payload.
- **#998** — `WorkflowSyntaxError` was lumped with `WorkflowDefinitionError`; needed its own `except` arm to serialize the right error shape. Check that new workflow/error subclasses aren't swallowed by a broader `except (...)` tuple above them (order matters — specific before general).
- **#795** — malformed `usage_fps`/`frames` (non-numeric) crashed usage accounting; guard external numeric inputs with `isinstance(x, numbers.Number)` before rounding/arithmetic.
- **#1365 / #1096** — builder `createTime`/`updateTime` must be `{"_seconds": int(...)}` (Firestore-shaped) not a bare int; and file lookups must hash the id. Builder route response shapes have external consumers — don't change them casually.
- **#1104** — versionless legacy model ids and silent `except` arms: legacy routes must handle `dataset_id`+`version_id` composition and log every caught exception.
- **#2528 → #2529** — the serverless-auth "fail closed on unexpected upstream status" fix was merged then **reverted**. Treat changes to serverless/Lambda auth middleware caching + fail-open/closed semantics as high-risk: they must not cache unexpected upstream statuses and must be covered by tests asserting both status and that responses aren't cached (`await_count`). Expect extra scrutiny / a maintainer sign-off.
- **#2222 / #721 / #724 / #190** — cold-start header aggregation, Prometheus GPU metrics, and orjson swaps were all reverted historically. Response-header aggregation, metrics endpoints, and serializer swaps are recurring revert magnets — require strong justification + tests.

## Review checklist
1. Every new/modified route is decorated with `with_route_exceptions` (sync) or `with_route_exceptions_async` (async), imported from `error_handlers.py`.
2. Any new exception surfaced to a route has an explicit `except` arm with an intentional `status_code` in BOTH sync and async wrappers, plus the `roboflow_api.py` status map if it originates from a Roboflow API call.
3. `except` arm ordering: specific exceptions precede the broad tuples that would otherwise catch them (#998).
4. New endpoint is gated by an env flag from `env.py`, and excluded for `LAMBDA`/`GCP_SERVERLESS` where the Lambda authorizer can't support it.
5. New env var/flag/numeric bound is declared in `inference/core/env.py` (not read inline), with sane default and, for tuning knobs, a rationale comment.
6. `response_model` unions use `Union[...]`; response pydantic changes preserve existing fields/aliases; None-but-contractual fields aren't dropped by `exclude_none`.
7. New response header is added to CORS `expose_headers` and its constant lives in `constants.py`/`request_metrics.py`.
8. Request body/stream is read exactly once; user-supplied file paths (builder) are regex-validated and hashed.
9. `inference/core/version.py` bumped for behavior changes.
10. Tests added/updated in `tests/inference/unit_tests/core/interfaces/http/` covering the new status/field/header/branch; serverless-auth and metrics/header-aggregation changes carry extra test + justification.
11. CLI options follow the `Annotated[..., typer.Option("--x/--no-x", help=...)]` pattern and are passed through to the container adapter.
12. Log level matches severity (warning for client-caused, exception for server faults); every caught exception is logged.

## Key files & entry points
- `inference/core/interfaces/http/http_api.py` — `HttpInterface` app factory, all route registrations, middleware wiring, CORS `expose_headers`.
- `inference/core/interfaces/http/error_handlers.py` — `with_route_exceptions` / `with_route_exceptions_async`; the exception→status contract.
- `inference/core/interfaces/http/orjson_utils.py` — response serialization (`orjson_response`, `orjson_response_keeping_parent_id`).
- `inference/core/interfaces/http/dependencies.py` — legacy request body/multipart parsing.
- `inference/core/interfaces/http/builder/routes.py` — builder UI routes (path-safety, response shapes).
- `inference/core/interfaces/http/middlewares/{cors,gzip}.py`, `request_metrics.py`, `uvicorn_config.py`.
- `inference/core/env.py`, `inference/core/version.py`, `inference/core/constants.py`, `inference/core/roboflow_api.py`.
- `inference_cli/server.py` — CLI. Tests: `tests/inference/unit_tests/core/interfaces/http/`.

## Reference PRs
- [#2099](https://github.com/roboflow/inference/pull/2099) — feature: `PaymentRequiredError` → 402 mirrored across sync+async handlers + roboflow_api status map.
- [#1557](https://github.com/roboflow/inference/pull/1557) — feature: gate `GET /model/registry` behind `GET_MODEL_REGISTRY_ENABLED` env flag.
- [#1717](https://github.com/roboflow/inference/pull/1717) — perf: shared workflows thread pool behind `HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_*` flags.
- [#1512](https://github.com/roboflow/inference/pull/1512) — feature: async handlers use `with_route_exceptions_async` (correct decorator import).
- [#1611](https://github.com/roboflow/inference/pull/1611) — bugfix: reject confidence 0 via `CONFIDENCE_LOWER_BOUND_OOM_PREVENTION` env const.
- [#1746](https://github.com/roboflow/inference/pull/1746) — bugfix: `elif` skipped confidence lower-bound clamp for percentage inputs.
- [#1518](https://github.com/roboflow/inference/pull/1518) — bugfix: multipart body read twice; centralized parsing in `dependencies.py`.
- [#1599](https://github.com/roboflow/inference/pull/1599) — bugfix: retain empty `parent_id` in OCR response; `Union[...]` response_model.
- [#998](https://github.com/roboflow/inference/pull/998) — bugfix: `WorkflowSyntaxError` needs its own `except` arm (400).
- [#1096](https://github.com/roboflow/inference/pull/1096) — security: builder path-traversal fix (regex + sha256 file lookup).
- [#2528](https://github.com/roboflow/inference/pull/2528) / [#2529](https://github.com/roboflow/inference/pull/2529) — bugfix + revert: serverless-auth fail-closed (503, no caching) — high-risk area.
- [#1024](https://github.com/roboflow/inference/pull/1024) — feature: `--metrics-enabled/--metrics-disabled` CLI flag threaded to container adapter.

## Related topic skills

When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):

- `review-topic-auth-and-tenant-security`
- `review-topic-backward-compat-and-versioning`
- `review-topic-external-contract-and-silent-fallback`
- `review-topic-test-hygiene`
