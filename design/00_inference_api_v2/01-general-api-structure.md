# API structure

This document lists the top-level shape of the `v2` API. Per-endpoint details are deferred to the subsequent documents.

## Models endpoints

* `POST /v2/models/run` — run model inference.
* `GET /v2/models/interface` — discover the request/response interface of a given model.
* `GET /v2/models/compatibility` — list model architectures compatible with the current server configuration.
* `GET /v2/models/loaded` — list models currently loaded into memory.
* `POST /v2/models/load` — load a given model.
* `DELETE /v2/models/unload` — unload a given model (or all loaded models, when no model is specified).

## Workflows endpoints

* `POST /v2/workflows/run` — run a workflow.
* `POST /v2/workflows/interface` — discover the request/response interface of a workflow. `POST` (rather than `GET`) is required because in-line workflows ship their definition in the request body.
* `POST /v2/workflows/validate` — validate a workflow definition (and optionally check runtime readiness — see `03-workflows.md`).
* `GET /v2/workflows/system/blocks` — describe blocks available on this server.
* `GET /v2/workflows/system/definition-schema` — return the JSON Schema for workflow definitions.
* `GET /v2/workflows/system/engine-versions` — list available engine versions.

## Video stream processing

TBD — specification deferred to a follow-up document.

## Server status

* `GET /v2/server/health` — liveness probe.
* `GET /v2/server/ready` — readiness probe.
* `GET /v2/server/info` — server build and configuration metadata.
* `GET /v2/server/metrics` — Prometheus-compatible metrics export.
