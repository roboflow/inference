---
name: review-topic-secure-gateway-url-wrapping
description: Load when a PR adds or modifies an outbound HTTP call (requests.*, aiohttp, httpx, urllib), constructs a URL from API_BASE_URL / METRICS_COLLECTOR_BASE_URL / HOSTED_*_URL / any *.roboflow.com host, adds an env var or setting that holds an endpoint URL, builds an InferenceHTTPClient, or touches wrap_url / roboflow_secure_gateway_proxy_url_builder / SECURE_GATEWAY handling.
---

# Review topic: Secure-gateway URL wrapping

## Why this exists

Air-gapped deployments set `SECURE_GATEWAY` (legacy `LICENSE_SERVER`) and can
reach ONLY the gateway host. Every Roboflow-bound request must be rewritten to
`{gateway}/proxy?url=<quoted url>` via `wrap_url()`
(`inference/core/utils/url_utils.py`) on the server side, or via
`roboflow_secure_gateway_proxy_url_builder`
(`inference_models/inference_models/weights_providers/roboflow.py`) inside
`inference_models`. A single unwrapped call site dead-ends behind the gateway —
usually as a hang or a retry loop, discovered only in a customer's air-gapped
environment (see #2658, which fixed three such gaps; #2263 for the offline
loading fallout).

## When this applies

Load when the diff shows ANY of these content signals (paths are hints, not the
trigger):

- A new or moved outbound HTTP call: `requests.get/post/put/head/delete`,
  `aiohttp`/`ClientSession`, `httpx`, `urllib.request.urlopen`, websockets.
- A URL built from `API_BASE_URL`, `METRICS_COLLECTOR_BASE_URL`,
  `HOSTED_DETECT_URL` / `HOSTED_INSTANCE_SEGMENTATION_URL` /
  `HOSTED_CLASSIFICATION_URL` / `HOSTED_CORE_MODEL_URL` /
  `HOSTED_SEMANTIC_SEGMENTATION_URL`, or any literal `*.roboflow.com` host.
- A new env var, pydantic setting, or config field whose value is an endpoint
  URL that will later be requested.
- A new `InferenceHTTPClient(api_url=...)` construction in server-side code.
- Any change to `wrap_url`, `roboflow_secure_gateway_proxy_url_builder`,
  `SECURE_GATEWAY` / `LICENSE_SERVER` parsing, or `/proxy?url=` handling.

## Review checklist

Tag each finding. Fix BLOCK before merge; raise FLAG; NIT is optional.

- **BLOCK** — A new request to a Roboflow-owned endpoint whose URL does not
  pass through `wrap_url()` (server code) or the weights-provider
  `proxy_url_builder` (`inference_models`) before the socket is opened.
  Trace the URL from construction to the `requests`/`aiohttp` call; "it is
  wrapped somewhere upstream" must be verifiable at a specific line. (Rule 1.)
- **BLOCK** — An endpoint URL sourced from an env var or setting where only
  the DEFAULT value is wrapped. Overrides must be wrapped at consumption time
  (validator or call site), not at field-definition time — a user-supplied
  `TELEMETRY_*`-style override otherwise bypasses the gateway. (Rule 2.)
- **BLOCK** — A network call that runs at module import time (version checks,
  registry warm-ups) with no timeout, or without honoring
  `SECURE_GATEWAY` / `OFFLINE_MODE` / its dedicated disable flag. Behind a
  packet-dropping firewall an untimed import-time call stalls server startup
  for the OS TCP timeout. (Rule 3.)
- **BLOCK** — A new consumer of hosted inference endpoints (`HOSTED_*_URL`,
  via `InferenceHTTPClient` or otherwise) with no `SECURE_GATEWAY` gate. The
  SDK client joins request paths onto `api_url`, which cannot compose with
  `/proxy?url=<encoded>` — hosted targets are unreachable behind a gateway, so
  the path must fail fast or fall back (see the remote+hosted → local fallback
  in `inference/core/env.py`). (Rule 4.)
- **FLAG** — Hand-rolled wrapping (f-string building `/proxy?url=` directly)
  instead of calling `wrap_url()`. The helper is idempotent and handles
  scheme-qualified vs bare-host gateways; inline copies drift. (Rule 5.)
- **FLAG** — A wrapped URL persisted to disk/db (queues, caches) and re-wrapped
  on read, or an endpoint stored pre-wrapped where `SECURE_GATEWAY` could
  differ between writer and reader processes. Persist raw URLs; wrap at send
  time. (Rule 6.)
- **FLAG** — A new third-party integration (Google/Stability/Twilio/Modal/
  webhook-style) reachable from a workflow or server path with no timeout or
  with errors that propagate as crashes rather than typed failures — behind a
  gateway these hosts are unreachable and the block must degrade cleanly.
  (Rule 7.)
- **NIT** — A new outbound integration not mentioned in the `SECURE_GATEWAY`
  section of `docs/quickstart/docker_configuration_options.md` when its
  gateway behavior is user-visible.

### Not blocking

- Calls to localhost / 127.0.0.1 / Unix sockets / LAN devices (PLC, ONVIF,
  local event-ingestion) — the gateway constraint is about egress.
- User-supplied media/webhook URLs fetched on behalf of the user (these are
  covered by `review-topic-input-boundary-security`); flag only if a NEW
  Roboflow-owned endpoint hides among them.
- Test files, docstrings, and examples containing URLs.
- `inference_models` package files downloaded from provider-returned URLs —
  those arrive pre-wrapped from `parse_package_artefacts` when the gateway is
  configured; do not demand a second wrap.

## Standards (one canonical statement per rule)

1. **Wrap at the socket, verifiably.** Every Roboflow-bound URL must flow
   through `wrap_url()` / `proxy_url_builder` on a traceable line before the
   request executes. The audit that produced this skill found the entire
   `roboflow_api.py` surface wrapped but 42 workflow `run_remotely` blocks and
   the version check unwrapped (#2658) — coverage claims must be per-call-site,
   not per-module.
2. **Defaults are not configuration.** Wrapping applied to a default value
   (field default, module constant) silently exempts every override. Wrap in
   the validator or at the call site so env-provided endpoints route through
   the gateway too (`TelemetrySettings` fix in #2658).
3. **Import-time network calls need a bound and a switch.** Any call that can
   run during `import inference.*` must carry an explicit `timeout=` and be
   disabled by `SECURE_GATEWAY` / `OFFLINE_MODE` / its own flag. The GitHub
   version check ran untimed at import and could stall startup for minutes on
   air-gapped networks (#2658).
4. **Hosted endpoints cannot be proxied — gate them.** Until `inference_sdk`
   understands the gateway `/proxy` contract, any server-side path that would
   contact `HOSTED_*_URL` under `SECURE_GATEWAY` must be rerouted or refused
   with a clear warning at configuration time, not left to fail per request.
5. **One wrapper implementation per package.** Server code calls `wrap_url()`;
   `inference_models` code uses the weights-provider builder. New inline
   `/proxy?url=` string construction is drift waiting to happen.
6. **Wrap late, store raw.** `SECURE_GATEWAY` is process configuration;
   persisted URLs outlive processes. Anything writing URLs to sqlite queues,
   caches, or model artifacts stores the raw URL and wraps at send time —
   `wrap_url` idempotence is a safety net, not a design.
7. **Unreachable third parties must degrade, not crash.** Blocks contacting
   non-Roboflow services are expected to fail behind a gateway — with a
   timeout, a typed/structured error, and no retry storm.
