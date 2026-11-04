# Databricks SQL Image Source

`roboflow_core/databricks_sql_image_source@v1` runs a SQL statement on a
Databricks SQL warehouse and emits every returned row as a Workflow image, with
an optional key column as the image name and the remaining columns as metadata.
This page is the operator guide: how the block is enabled, bounded and
verified. The block's own description (shown in the Workflow builder) covers
its inputs and outputs.

## Enabling and disabling

| Variable | Default | Meaning |
|---|---|---|
| `DATABRICKS_SQL_IMAGE_SOURCE_ENABLED` | `true` self-hosted, `false` when `GCP_SERVERLESS` or `LAMBDA` is set | The rollout switch. An explicit value wins on every deployment. |
| `ALLOW_DATABRICKS_WORKFLOWS_SOURCE_TO_NON_GLOBAL_ADDRESSES` | `false` | Accept warehouse hosts that resolve to private or link-local addresses (Private Link on a self-hosted server). Never honoured on the hosted platform. |
| `DATABRICKS_SQL_IMAGE_SOURCE_MAX_RUN_BYTES` | `1073741824` (1 GiB) | Byte budget per block invocation: encoded bytes, decoded pixels, names and metadata. |
| `DATABRICKS_SQL_IMAGE_SOURCE_MAX_IMAGE_BYTES` | `33554432` (32 MiB) | Encoded size limit per image. Also sizes the connector's inline result chunk (twice this value). |
| `DATABRICKS_SQL_IMAGE_SOURCE_MAX_IMAGE_PIXELS` | `25000000` | Pixel limit per image, read from the header before decoding. |

Installing the connector (`pip install "roboflow-workflows[databricks]"`,
shipped in every Roboflow server image) and enabling the block are separate
decisions. When the block is disabled, or the connector is missing:

* `/workflows/blocks/describe` does not list it,
* a Workflow that references it, saved or submitted inline, fails at compile
  time with `Block ... is not available on this server: ...` naming the reason,
* no DNS lookup and no connection to Databricks is made.

The variables are read when the server process starts. Changing one is a
**redeployment or restart, not a live kill switch**: on the hosted platform
set it in the worker values and roll the deployment; on a self-hosted server
restart the container. During a rollback:

* jobs queued before the rollout that reference the block fail at compile
  time once they reach a restarted worker,
* jobs already running on a worker that has not restarted finish or hit their
  own `timeout_seconds` deadline (at most 600 s) plus the cleanup interval
  below; a worker that is terminated mid-run cancels nothing on the warehouse,
  the statement's server-side `STATEMENT_TIMEOUT` (equal to the deadline)
  ends it,
* `WORKFLOW_DISABLED_BLOCK_PATTERNS=databricks` remains a valid additional
  safeguard on multi-tenant workers and is applied by the same mechanism.

## Memory and time bounds

All budgets are **per block invocation**, not per process. Size them against
the worker's memory and the number of Workflow requests one process serves at
once:

```
per invocation ≈ MAX_RUN_BYTES                       retained rows
               + 2 x MAX_IMAGE_BYTES                 connector inline chunk (+ lz4 copy)
               + MAX_IMAGE_BYTES + 3 x MAX_IMAGE_PIXELS   decode temporaries of one image
```

With the defaults that is about 1.2 GiB per invocation. For the hosted
async-serverless workers the values files set 256 MiB / 16 MiB / 12 MP, about
340 MiB per invocation, leaving headroom for models and concurrent requests
within the per-process share of the pod's memory limit.

The connector's buffering is bounded explicitly: the cursor asks the warehouse
for inline chunks of at most 8 rows and `2 x MAX_IMAGE_BYTES` bytes
(`arraysize` / `buffer_size_bytes`), with a server-side `row_limit` of
`max_rows`, cloud-fetch result links disabled and refused if a warehouse
returns them anyway, and metadata capped per value (4096 characters), per
nesting (4 levels, 256 items) and per row (64 KiB).

`timeout_seconds` (5 to 600, default 120) is the overall deadline for
connecting, executing and polling, fetching and decoding. The statement is
submitted with `STATEMENT_TIMEOUT` equal to it, HTTP retries are limited to 3
attempts with at most a 5 s delay inside the remaining time, and redirects are
refused. When the deadline passes a watchdog cancels the statement
(`CancelOperation`) and closes the session; the step's thread then fails
within one socket timeout (30 s) plus one retry delay (5 s). That 35 s is the
**cleanup interval**: after it the step has returned, its execution slot is
free and no connector thread remains.

## Network policy

Only `*.cloud.databricks.com`, `*.gcp.databricks.com` and
`*.azuredatabricks.net` hosts are accepted, validated as a whole string, and
only SQL warehouse paths on the hosted platform (`/sql/1.0/warehouses/<id>`,
cluster paths are self-hosted only). The host is resolved once; every answer
must be a global address (unless the non-global flag is set on a self-hosted
server) and the first allowed answer is the one the connector connects to. The
connector's Thrift HTTP transport is pinned to that address while keeping the
hostname for TLS SNI, certificate verification and the `Host` header, so a DNS
change after validation cannot redirect the connection. Proxies, HTTP, disabled
TLS verification and redirects are refused by the transport; the connector's
second HTTP client (telemetry, feature flags, OAuth, cloud fetch, staging) is
refused for the duration of the run from any thread.

Transport inventory for `databricks-sql-connector` 4.x (verified on 4.8.0 with
`urllib3` 2.8): one `urllib3` pool for every Thrift call (session, execute,
polling, fetch, cancel, close, retries and reconnects) and one `PoolManager`
for everything else. Both are covered as described above; the REST (SEA)
backend is not used.

## Verification

Deterministic, in CI with the workflows unit tests:

* `workflows/tests/unit_tests/core_steps/integrations/test_databricks_sql_image_source.py`:
  manifest and destination validation, enablement gate (discovery, execution,
  direct submission with zero connection attempts), budgets and limits at
  their exact boundaries, the crafted-TIFF case, oversized metadata and names,
  batching, cleanup on every failure path, repeated failures.
* `test_databricks_transport_and_availability.py`: the pinned pool, the
  second-client guard, loader and parser behaviour.
* `test_databricks_real_transport.py`: the real connector against a local
  HTTPS Thrift warehouse (`databricks_thrift_harness.py`, test-only CA):
  import across fetch chunks, SNI / `Host` / single-destination assertions
  from socket-level observation, DNS changing after validation, retryable
  errors and forced reconnects, redirects to forbidden destinations (including
  HTTP and the metadata service address, refused at the socket), warehouse
  supplied result links, invalid certificates, proxies, slow-query
  cancellation within the cleanup interval, recovery and thread hygiene.
* `test_databricks_connector_contract.py`: the connector keywords, methods
  and transport hooks the block relies on, against the installed version.

Credentialed, against a real warehouse: `workflows/scripts/databricks_live_integration.py`
(see its docstring) runs import, bad image, oversized metadata, invalid
credential, slow-query cancellation verified through the warehouse's query
history, recovery and concurrent-job latency, in-process or through a server
or async-serverless producer URL, plus `--expect-disabled` and
`--expect-budget-failure` modes for deployments configured that way. The
GitHub workflow `Databricks SQL Image Source - live warehouse integration`
runs it on dispatch under the `databricks-live` environment. Record the image
digest, connector version and report in the release notes of any hosted
rollout.
