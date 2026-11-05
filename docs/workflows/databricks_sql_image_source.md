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
decisions. The transport supports connector **4.8.0**; the extra pins that version.
When the block is disabled, or the connector is missing or unsupported:

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
  rely on the warehouse statement timeout and verify its terminal state,
* `WORKFLOW_DISABLED_BLOCK_PATTERNS=databricks` remains a valid additional
  safeguard on multi-tenant workers and is applied by the same mechanism.

## Memory and time bounds

Each invocation runs DNS resolution, the SQL connector and image decoding in a
fresh Python subprocess. The parent checks availability and `disable_sinks`
before launch. Credentials and resolved configuration travel over a private
pipe; connector patches never alter clients in the parent process. The serialized
request (SQL, credentials and configuration together) is limited to 1 MiB.

All budgets are **per invocation**, and account for encoded image bytes, decoded
pixels, names and metadata. They are application-data budgets, not hard process
RSS limits. Connector buffers, Python imports, decode temporaries and filesystem
cache require additional memory. Measure the candidate image at the intended
concurrency before enabling hosted execution; do not infer a pod memory limit
from `MAX_RUN_BYTES` alone.

The child writes decoded arrays and a bounded manifest into a private temporary
directory. The parent loads them only after the child exits successfully, checks
array headers and file sizes before allocation, and removes the directory on
success or failure. Provision temporary storage for up to the invocation budget
plus file/manifest overhead per concurrent invocation. Disk exhaustion fails the
step; no partial output is returned.

The connector requests inline chunks of at most 8 rows and
`2 x MAX_IMAGE_BYTES` bytes, with a server-side `row_limit` of `max_rows`.
Metadata values are truncated to 4096 characters and capped at 4 nesting levels,
256 items per collection and 64 KiB per row. JPEG, PNG and WebP are accepted;
TIFF is rejected before pixel decoding. Total budget exhaustion fails the step,
while ordinary row conversion failures follow `skip_undecodable`.

`timeout_seconds` (5 to 600, default 120) begins before process launch and covers
imports, DNS, connection, execution, fetching, decoding and parent output loading.
The warehouse receives a statement timeout based on the remaining time. On
expiry the parent terminates the child, giving cleanup a single 35-second
allowance before forcing termination and reaping it. Connector cancellation and
closing can block on the connector's request lock, so they are best effort;
killing a client does not prove its warehouse query stopped. Verify warehouse
terminal state in the deferred staging acceptance run.

Parent output loading checks expiry between bounded operations and before
returning success. Arbitrary OS/filesystem stalls are outside this wall-clock
guarantee. A server restart is still required to change the rollout flag.

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
refused for the entire child-process lifetime, including its background threads.
Unrelated clients in the parent process remain independent.

Transport inventory for `databricks-sql-connector` 4.8.0 (verified with
`urllib3` 2.8): one `urllib3` pool for every Thrift call (session, execute,
polling, fetch, cancel, close, retries and reconnects) and one `PoolManager`
for everything else. Both are covered as described above; the REST (SEA)
and native kernel backends, including automatic fallback, are rejected before
backend construction.

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
history, recovery and concurrent-job latency, locally or through the synchronous server `/workflows/run` endpoint, plus `--expect-disabled` and
`--expect-budget-failure` modes for deployments configured that way. The
GitHub workflow `Databricks SQL Image Source - live warehouse integration`
runs it on dispatch under the protected `databricks-live` environment. Configure
its HTTPS server origin with the environment variable `DATABRICKS_LIVE_BASE_URL`
(GitHub environment variable of the same name); callers cannot choose a URL in
dispatch inputs. Requests refuse redirects. The synchronous runner does not
validate the async-serverless producer/job API: that path requires a separate
staging run after PR review. No credentialed staging test is part of local review.
Record the image
digest, connector version and report in the release notes of any hosted
rollout.
