---
name: review-topic-auth-and-tenant-security
description: Load when a PR touches `http_api.py` auth middleware (`check_authorization_serverless` / `check_authorization`), `get_serverless_usage_check_async`, api-key→workspace resolution (`get_roboflow_workspace`), assume-identity (`_add_assume_identity_headers`, `x-assume-identity-*`), the unauthenticated-route allowlist, api-key redaction (`api_key_safe_raise_for_status`, `deduct_api_key`), usage api-key hashing, or new `*_SECRET`/`*_TOKEN`/`API_KEY` env vars in `env.py`.
---

# Review topic: Auth, tenant-boundary & secret handling

## When this applies
Trigger on CONTENT/behaviour, not just directory. Load when the diff:
- Touches the auth/usage middleware in `inference/core/interfaces/http/http_api.py` (`check_authorization_serverless`, `check_authorization`) or the unauthenticated-route allowlist inside them.
- Calls `get_serverless_usage_check_async` or branches on its `status_code`.
- Resolves/compares api-keys, workspaces, or projects — `get_roboflow_workspace(_async)`, the `cached_api_keys` map / `AuthorizationCacheKey`, `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT`, `DEDICATED_DEPLOYMENT_WORKSPACE_URL`.
- Adds/reads assume-identity plumbing — `_add_assume_identity_headers`, `ASSUME_IDENTITY_ACCESS_TOKEN_HEADER` (`x-assume-identity-access-token`), `assume_identity_authorised_workspace_db_id`, `ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN`.
- Adds/forwards secrets or tokens — a new `*_SECRET`/`*_TOKEN`/`API_KEY` in `inference/core/env.py`, or a request/URL/header that may carry `api_key=` reaching a log, exception, response, metric label, or usage payload.

## Review checklist
Severity-tag every finding. Reference the numbered Standards below.

- **BLOCK** — Ambiguous/unexpected serverless usage-check outcome falls through to `call_next` (fail-open). The `status_code` branches cover 200/401/402; any other value (upstream 5xx, timeout, unmapped code) must deny and MUST NOT be cached. [S1]
- **BLOCK** — Missing `api_key` (query params AND JSON body) does not return 401 via `_authorization_error_response` / `_unauthorized_response`. [S2]
- **BLOCK** — An ambiguous/error auth result is written into `cached_api_keys` under a reusable key, so it can later be served as an allow. Only definitive outcomes are cached. [S3]
- **BLOCK** — A raw secret/api-key/token reaches a log line, exception `str()`, HTTP response, Prometheus label, or persisted usage payload in clear form. [S6]
- **BLOCK** — A revert re-opens a previously closed fail-open hole without a compensating control. [S1]
- **FLAG** — Auth cache keyed too broadly (not on `(api_key, enforce_credits_verification)` in the serverless path / `api_key` in the local path), risking one tenant's grant served to another. [S3]
- **FLAG** — A data/inference route added to the unauthenticated allowlist, or a prefix-match where an exact-match is required. [S4]
- **FLAG** — Workspace boundary not enforced against `DEDICATED_DEPLOYMENT_WORKSPACE_URL` / `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT`, or api-key resolvable to a workspace other than the caller's. [S5]
- **FLAG** — Assume-identity headers injected without the `ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN` + authorised-workspace gate. [S5]
- **FLAG** — Redaction seam (`api_key_safe_raise_for_status(_aiohttp)`) skipped on a Roboflow-API egress whose URL carries `api_key=`. [S6]
- **NIT** — New secret env var without a `None`/unset default, or exposed via an `/info`-style introspection endpoint. [S7]

### Not blocking
- Do NOT demand redaction refactors on paths that never carry a credential (internal URLs, static assets).
- Do NOT flag a correctly-keyed auth cache purely for its TTL choice — `AUTH_CACHE_TTL_SECONDS` (grants) vs `SHORT_AUTH_CACHE_TTL_SECONDS` (denials) is a deliberate split, not a bug.
- Do NOT require assume-identity gating on codepaths that never set `assume_identity_authorised_workspace_db_id`.
- The `deduct_api_key` convention deliberately reveals a 2-char prefix+postfix for keys ≥ 8 chars (`***` only, below that) — that is the intended redaction, not a leak.
- SSRF / user-supplied-URL validation is owned by `review-topic-input-boundary-security` — cross-ref it, do not review it here.

## Standards

**S1 — Fail-closed authorization.** In `check_authorization_serverless`, an ambiguous outcome (unexpected `get_serverless_usage_check_async.status_code`, upstream 5xx, timeout, missing field) must deny the request, not admit it. The current branches handle `200` (authorize), `401` (unauthorized), `402` (credits) and return explicitly; any other status falls through to `call_next` — a fail-open bug that silently grants free inference and is invisible in normal traffic (PR #2528 restored the deny-and-no-cache contract; PR #2529 reverted it, so treat reverts here as High-risk needing explicit justification).

**S2 — Missing credential ⇒ 401.** `api_key` is read from both query params and JSON body; when `None`, return 401 immediately (`_authorization_error_response(401, ...)` / `_unauthorized_response(...)`), never a pass-through, and never store/act on usage for a missing key (PR #772).

**S3 — Cache correctness.** The serverless path keys `cached_api_keys` on `AuthorizationCacheKey = (api_key, enforce_credits_verification)`; the local-deployment path keys on `api_key`. Never widen to a key that could serve tenant A's grant to tenant B. Denials are cached under `SHORT_AUTH_CACHE_TTL_SECONDS`; ambiguous/error results are NOT cached at all (PR #772).

**S4 — Unauthenticated-route allowlist.** The middlewares exclude a fixed list of paths (`/`, `/docs`, `/info`, `/healthz`, `/readiness`, `/metrics`, `/openapi.json`, `/model/registry`, and `/workflows/blocks/describe` only for GET / bodyless requests). Match is done against the raw ASGI `request.scope["path"]`, NOT `request.url.path`, so a malicious `Host` header cannot poison the path and slip an authenticated route in (CVE-2026-48710). Any addition must be genuinely safe to expose unauthenticated; prefer exact-match over prefix-match; never add a data/inference route.

**S5 — Tenant/workspace isolation.** A caller's api-key must only resolve to and act on its own workspace. `get_roboflow_workspace(_async)` fails (`WorkspaceLoadError`) on an empty workspace. In `check_authorization`, the resolved `workspace_id` must be in `allowed_workspaces` built from `DEDICATED_DEPLOYMENT_WORKSPACE_URL` + `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT`, else 401 (PR #988 simplified this to a pure workspace-ID match). `_add_assume_identity_headers` injects `ASSUME_IDENTITY_ACCESS_TOKEN_HEADER` / `ASSUME_IDENTITY_AUTHORISED_WORKSPACE_HEADER` only when BOTH `ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN` is set AND `assume_identity_authorised_workspace_db_id` is present.

**S6 — No secret exposure.** Secrets, api-keys, and tokens must never be logged, echoed in responses/exceptions, or persisted in clear form. Roboflow-API egress whose URL may carry `api_key=` must pass through `api_key_safe_raise_for_status` / `api_key_safe_raise_for_status_aiohttp` (they apply `API_KEY_PATTERN` → `deduct_api_key`) before raising/logging (PR #140, #255, #248, #188). Usage-tracking hashes keys via `_calculate_api_key_hash` and never stores them raw; do not mangle the hash used for usage identity (PR #772, #1831).

**S7 — New secret env vars.** A new `*_SECRET`/`*_TOKEN`/api-key in `inference/core/env.py` must default to `None`/unset (never a literal value), must not be printed at startup, and must not be returned by any `/info`-style introspection endpoint.

## Key files & reference PRs
- `inference/core/interfaces/http/http_api.py` — `check_authorization_serverless` (canonical fail-closed serverless path: 401 on missing/unauthorized key, explicit 401/402 branches, credential-scoped `cached_api_keys` on `AuthorizationCacheKey`) and `check_authorization` (workspace-boundary enforcement + minimal allowlist). TTLs: `AUTH_CACHE_TTL_SECONDS`, `SHORT_AUTH_CACHE_TTL_SECONDS`.
- `inference/core/roboflow_api.py` — `get_serverless_usage_check_async`, `get_roboflow_workspace(_async)`, `_add_assume_identity_headers`, `assume_identity_authorised_workspace_db_id`, `ASSUME_IDENTITY_ACCESS_TOKEN_HEADER`.
- `inference/core/utils/requests.py` — `api_key_safe_raise_for_status`, `api_key_safe_raise_for_status_aiohttp`, `deduct_api_key`, `API_KEY_PATTERN`.
- `inference/core/env.py` — secret/token declarations (`ROBOFLOW_INTERNAL_SERVICE_SECRET`, `ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN`, `MODAL_TOKEN_SECRET`, `PRELOAD_API_KEY`, `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT`) — all default to `None`/unset; pattern-match new secrets against these.
- `inference/usage_tracking/collector.py` — `_calculate_api_key_hash`, `_hashed_api_keys`.
- Reference PRs: #2528 / #2529 (serverless fail-open fix + revert), #772 / #1831 (missing-key usage + hash identity), #140 / #255 / #248 / #188 (api-key redaction), #988 (workspace-ID match), #2417 (self-hosted allowlist).
