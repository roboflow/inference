# Databricks SQL Image Source

`roboflow_core/databricks_sql_image_source@v1` reads selected columns from a table or view on a
Databricks SQL warehouse and emits every returned row as a Workflow image, with
an optional key column as the image name and explicitly selected columns as metadata.
This page is the operator guide: how the block is enabled, bounded and
verified. The block's own description (shown in the Workflow builder) covers
its inputs and outputs.

## Query contract

Configure `catalog`, `schema_name`, `table`, `image_column` (default `bytes`),
optional `image_name_column`, `metadata_columns` (default empty, at most 32) and
`max_rows` (1-1000). Each name is a single unquoted identifier, bounded to 255
characters. Embedded backticks are escaped within each component; dots are literal
parts of a component. The block executes only this template, extending the
projection with additional bound identifiers for the name and metadata columns:

```sql
SELECT IDENTIFIER(:column_0)
FROM IDENTIFIER(:source)
LIMIT :row_limit
```

Native parameters contain the quoted names and integer limit. SQL fragments,
expressions and `sql_statement` are not accepted; old workflow definitions must
migrate to these inputs. For example, use `catalog=main`, `schema_name=vision`,
`table=images_to_label`, `image_column=image_bytes`, `max_rows=100`.
Define filters and joins in a customer-managed Databricks view. A view preserves
selection flexibility; the fixed query and native parameter binding prevent input
from changing SQL structure. Requires SQL warehouse support for IDENTIFIER.

This does not prove authorization or tenant ownership, enforce read-only grants,
or bound view computation. Provision a dedicated identity with SELECT on the
intended sources and CAN USE on the warehouse. SQL LIMIT bounds returned rows,
not warehouse cost. Aggregate admission controls are described below.

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
request (credentials and configuration together) is measured before serialization
and limited to 1 MiB.

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
The transport enforces an HTTP body cap of
`max(1 MiB, 2 x MAX_IMAGE_BYTES) + 64 KiB` on every response, including chunked
responses. Oversized Content-Length is rejected before reading; no rejected body
is drained. HTTP compression is refused. Thrift strings are limited to this byte
cap, containers to 65,536 entries, total structural items to 131,072 and nesting
to 64 levels. The worker requests column-based results and rejects Arrow schemas,
Arrow/LZ4 batches and result links before their native decoders run. This remains
an application bound with Python object overhead, not an RSS guarantee.

Metadata values are truncated to 4096 characters and capped at 4 nesting levels,
256 items per collection and 64 KiB per row. JPEG and PNG are accepted;
WebP and TIFF are rejected before constructing their decoders. Total budget exhaustion fails the step,
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

The live runner creates temporary fixture views using `DATABRICKS_SETUP_TOKEN`
for every run, then submits only the fixed table/view reader configuration with
`DATABRICKS_TOKEN`. Grant that restricted identity SELECT on the fixture schema.
Views are dropped after the suite, with an exit cleanup retry on failures. Slow
case verification requires query history to expose the unique view identifier;
if it is absent, the case fails instead of claiming cancellation was verified.

## Aggregate admission control

Before launching the child or resolving the Databricks hostname, the parent
validates destination syntax and acquires one admission lease. Hosted execution
requires both the injected host Redis controller and an authenticated Roboflow
workspace resolved from the host API key. Workflow inputs cannot choose a tenant
ID, admission backend or quota. The Databricks token is not the tenancy key.
Authentication may use the trusted platform API; no Databricks DNS/network occurs
before admission.

Both the legacy and standalone server bind the controller to their existing
Workflows Redis cache client. There is no admission fallback during Redis errors.
A hosted worker whose cache fell back to memory refuses the Databricks block.
Self-hosted/standalone execution uses process-local limits when no Redis client
is bound and treats the installation as one tenant. Multi-process self-hosting
must bind Redis to aggregate across its workers.

Each setting below is a positive integer (maximum 100000), read at startup.
All workers sharing the quota namespace must use the same settings.

| Environment variable | Default | Scope |
|---|---:|---|
| `DATABRICKS_ADMISSION_TENANT_CONCURRENCY` | 2 | Active invocations per authenticated workspace |
| `DATABRICKS_ADMISSION_WORKSPACE_CONCURRENCY` | 8 | Active invocations per Databricks hostname, across tenants |
| `DATABRICKS_ADMISSION_WAREHOUSE_CONCURRENCY` | 4 | Active invocations per hostname + warehouse/cluster path |
| `DATABRICKS_ADMISSION_FLEET_CONCURRENCY` | 32 | Active Databricks invocations across the shared Redis scope |
| `DATABRICKS_ADMISSION_TENANT_RATE` | 30 | Admissions per tenant per rolling 60 seconds |
| `DATABRICKS_ADMISSION_WORKSPACE_RATE` | 120 | Admissions per destination hostname per rolling 60 seconds |
| `DATABRICKS_ADMISSION_WAREHOUSE_RATE` | 60 | Admissions per destination endpoint per rolling 60 seconds |
| `DATABRICKS_ADMISSION_FLEET_RATE` | 300 | Admissions per shared Redis scope per rolling 60 seconds |

One atomic Lua script checks all eight quotas using Redis time and records the
lease only if all pass. No unbounded queue is created. Rejected runs return a
controlled execution error; callers should back off. Successful admissions count
even if credentials, decoding or execution later fail. `/endpoints/<id>` and
`/warehouses/<id>` share a warehouse quota; case and a trailing hostname dot are
normalized. Destination identity is hostname-based, not proof of workspace
ownership or equivalence across different hostnames/IP aliases.

Concurrency is released after the worker supervisor finishes, including result
loading and termination cleanup. Rate entries last 60 seconds. Orphan leases
expire at the remaining invocation deadline plus 65 seconds (35-second cleanup
allowance plus scheduling tolerance); Redis keys expire too. Failed release keeps
the lease until expiry and logs a credential-free warning. OS stalls, orphaned
processes surviving beyond the lease, or Redis state loss can break a strict
physical-process ceiling: retain pod/process resource limits and verify shutdown
behavior in staging.

All hosted worker pools to be covered by a fleet quota must share the same Redis
scope. Use non-evicting Redis storage for these keys and monitor capacity/errors;
evicting, flushing or losing quota keys resets protection. Configure consistent
limits and size pod memory/temp storage for the maximum local concurrency before
enabling the block. Defaults are conservative starting values, not measured
production capacity. This PR neither audits deployed gateway limits nor changes
other Workflow blocks, ingress traffic, or remote-query cancellation guarantees.

Offline Lua regression command (test-only dependencies, not runtime dependencies):
`uv run --with 'fakeredis[lua]' pytest workflows/tests/unit_tests/core_steps/integrations/test_databricks_admission.py`.
Set `DATABRICKS_ADMISSION_TEST_REDIS_BINARY` to a local redis-server binary to
also run the eight-process test against a disposable Unix-socket Redis. It checks
atomic ceilings, rate persistence, orphan expiry and outage rejection.
A deployed shared Redis multi-worker staging check, operator values, overload/recovery
monitoring, and final-image validation remain rollout requirements.
