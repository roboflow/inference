---
name: review-cli-cloud-tooling
description: PRs changing inference_cli/ except server.py, container_adapter.py, tunnel_adapter.py, infer_adapter.py. Diff signals — typer.Option/typer.Argument, cloud_app/rf_cloud_app/benchmark_app/enterprise_app, get_workspace/handle_response_errors, @backoff.on_exception(..., RetryError), serialization_alias/alias= on pydantic Fields, REQUEST_TIMEOUT config constants, cloud_adapter.py (SkyPilot deploy), inference_cli/lib/enterprise/**, tests/inference_cli/**, docs/inference_helpers/cli_commands/**.
---

# Reviewing cli-cloud-tooling changes

## Scope
Triggers on changes under `inference_cli/` **except** `inference_cli/server.py`, `inference_cli/lib/container_adapter.py`, `tunnel_adapter.py`, `infer_adapter.py` (server/container/tunnel lifecycle — different skill). In scope:
- `inference_cli/benchmark.py`, `inference_cli/workflows.py`, `inference_cli/cloud.py`, `inference_cli/main.py`
- `inference_cli/lib/cloud_adapter.py` — SkyPilot cloud-deploy backing the `cloud` app.
- `inference_cli/lib/benchmark/**`, `inference_cli/lib/benchmark_adapter.py`
- `inference_cli/lib/workflows/**` (core, common, entities, local/remote image + video adapters)
- `inference_cli/lib/roboflow_cloud/**` (`common.py`, `config.py`, `errors.py`, `core.py`, `data_staging/**`, `batch_processing/**`)
- `inference_cli/lib/enterprise/**` (inference-compiler TRT extension)
- Companion tests under `tests/inference_cli/**` and CLI docs under `docs/inference_helpers/cli_commands/**`.

OUT of scope: the four lifecycle adapters above; the Workflows Execution Engine internals and blocks (separate skill); `inference_sdk`.

Two distinct cloud surfaces live here — do not conflate them:
- **`cloud` app** (`inference_cli/cloud.py` → `inference_cli/lib/cloud_adapter.py`): SkyPilot VM deploy (`deploy/status/start/stop/undeploy`). No Roboflow HTTP calls; older option/error style (see below).
- **`rf-cloud` app** (`inference_cli/lib/roboflow_cloud/**`): data-staging + batch-processing against the Roboflow API. This is where the HTTP/retry/pydantic-alias contracts apply.

## Review checklist

**BLOCK** — must be fixed before merge:
- HTTP: a new `rf-cloud` API op that does NOT go through `get_workspace()` + `handle_response_errors(operation_name=...)`, is not decorated `@backoff.on_exception(backoff.constant, RetryError, max_tries=3, interval=1)`, does not pass `timeout=REQUEST_TIMEOUT`, or does not re-raise connectivity/`Timeout` as `RetryError`. Raising a bare `Exception` where retry logic expects `RetryError` breaks retries.
- Wire schema: a request/response field whose `serialization_alias` (outbound camelCase) or `alias` (inbound) does not match the backend contract, or a casual rename of an existing alias.
- Breaking CLI contract: renaming/removing a `typer.Option` long or short flag, a command name, or narrowing a default. A colliding short flag within one command silently shadows (`-o` reuse, #2038).
- Benchmark integrity: fabricating metrics on the error path (`execution_time=0` counted a failure as a zero-latency success, #1150).

**FLAG** — reviewer should raise:
- New network command missing `api_key` resolution from `ROBOFLOW_API_KEY`, a `--debug-mode/--no-debug-mode` flag, or the standard `rf-cloud` error envelope (KeyboardInterrupt → exit 2; else debug re-raise / echo + exit 1).
- New request capability not threaded through all three layers: `Optional[...] = None` entity field → `api_operations` param → `core.py` command option (#2399).
- Closed choice set typed as a free-form `str` instead of a `str, Enum` (#2062).
- Magic number hardcoded in `api_operations` instead of an `os.getenv`-backed constant in `config.py`.
- Filesystem path handling: manual `"/"` split (#1051), glob missing the `*` wildcard (#1157), or non-deterministic/unlabeled aggregation (#891).
- `isinstance` vs `issubclass` misuse in exception-type checks (#354).
- Missing companions: tests, CI deps, docs, swagger (see Standards).
- Removal that leaves a dead export/command behind (#1211).

**NIT**:
- Accidental trailing-comma tuple (`x = (foo,)`) in option/arg construction.
- Missing `help=` string on a new option.

### Not blocking
- Do NOT ask for an `inference/core/version.py` bump — inference releases are versioned separately from feature/bugfix PRs; the bump is coordinated at release time.
- `inference_cli/version.py` `__version__` is auto-adjusted by the PyPI-build CI action — do NOT ask for a manual bump there.
- The `cloud` app (SkyPilot) deliberately uses `str` options, single-char shorts, and a bare `except Exception` → exit-1 envelope with no `debug_mode`. Do NOT demand the `rf-cloud` conventions on `cloud.py`/`cloud_adapter.py` — it is a separate, older surface.
- Speed/latency evidence for benchmark changes is verifiable via the PR description, not the diff.

## Standards

**CLI signature = public contract.** Option long/short names, defaults, and command names are user-facing. Removing/renaming an option or short flag is breaking; additions must default to `None`/existing behavior. New params: `Annotated[Optional[T], typer.Option("--long", "-short", help="...")] = None`, snake_case Python name, kebab-case flag. Short flags must be unique within a command — a duplicate silently shadows (benchmark `-o` reuse renamed to `-mpi`, #2038). (`inference_cli/lib/roboflow_cloud/data_staging/core.py`, #1516, #2143.)

**RF-cloud HTTP contract.** All `rf-cloud` API calls go through `get_workspace()` + `handle_response_errors(response, operation_name=...)` in `roboflow_cloud/common.py`, decorated `@backoff.on_exception(backoff.constant, RetryError, max_tries=3, interval=1)`, passing `timeout=REQUEST_TIMEOUT`. Connectivity/`Timeout` is re-raised as `RetryError`. In `handle_response_errors`: codes in `HTTP_CODES_TO_RETRY = {408,429,502,503,504}` → `RetryError`; 401 → `UnauthorizedRequestError`; any other `>= 400` → `RFAPICallError`. (`roboflow_cloud/common.py`, `config.py`; timeout made env-configurable in #1950.)

**Error taxonomy.** `roboflow_cloud/errors.py`: `RoboflowCloudCommandError` is the base; `RetryError` and `RFAPICallError` subclass it; `UnauthorizedRequestError` subclasses `RFAPICallError`. Retry keys off `RetryError` type — do not raise a bare `Exception` where a `RetryError` is expected.

**Command error envelope (rf-cloud).** Command bodies resolve `api_key` (falling back to `ROBOFLOW_API_KEY`), then wrap the delegate call in `try/except KeyboardInterrupt` → `print("Command interrupted.")` + `raise typer.Exit(code=2)`, and `except Exception as error:` → `if debug_mode: raise error` else `typer.echo("Command failed. Cause: {error}")` + `raise typer.Exit(code=1)`. Non-zero exit codes are the scripting contract. (`roboflow_cloud/data_staging/core.py`.)

**Pydantic wire schema.** Entities in `*/entities.py` map snake_case ⇄ camelCase JSON. Outbound bodies use `serialization_alias` (e.g. `batchId`, `partName`, `machineType`); inbound responses use `alias` (e.g. `displayName`, `shardId`). JSON keys are the server API contract and must match the backend. (`roboflow_cloud/{data_staging,batch_processing}/entities.py`.)

**New request field is optional + threaded end-to-end.** A new capability adds an `Optional[...] = None` field (with `serialization_alias`) on the entity, a matching param on the `api_operations` trigger fn, and the `core.py` command option — all three layers. (#2399, #2143, #2062.)

**Enums for closed choice sets.** Machine/backend/format options are `str, Enum` in `*/entities.py` (`MachineType`, `MachineSize`, `InferenceBackend`, `AggregationFormat`, `CompilationDevice`), so Typer validates the choice. (`batch_processing/entities.py`; `InferenceBackend` added in #2062.)

**Tunables live in `config.py`, env-overridable.** `REQUEST_TIMEOUT`, `MAX_SHARDS_UPLOAD_PROCESSES`, `MAX_DOWNLOAD_THREADS`, `MAX_IMAGE_REFERENCES_IN_INGEST_REQUEST` use `int(os.getenv("NAME", default))`. Do not hardcode magic numbers in `api_operations`. (`roboflow_cloud/config.py`; #1950.)

**Filesystem & aggregation robustness.** Use `os.path` for path decomposition (no manual `"/"` split — #1051 broke on Windows); local-image globs must include the `*` wildcard because `IMAGE_EXTENSIONS` are bare extensions (#1157); batch-result aggregation must be deterministic (`sorted(...)`) and stamp source provenance (the `image` key — #891 produced unordered/unlabeled rows).

**Benchmark error paths.** Do not inject fake metrics on failure (`execution_time=0` polluted latency stats, #1150), and use `isinstance` (not `issubclass` on an instance) for exception-type checks (#354).

**SkyPilot `cloud` deploy.** `cloud_adapter.py` lazily imports `sky` via `check_sky_installed()` (prompting `pip install inference[cloud-deploy]`) rather than importing at module top, so the base CLI imports without the heavy optional dep. Keep new `cloud`/SkyPilot code behind that guard; the `YAML_DEFS` templates (`gcp_cpu`/`gcp_gpu`/`aws_cpu`/`aws_gpu`) carry the container images and ports — treat changes to them as deployment-contract changes.

**Enterprise extension isolation.** `inference_cli/lib/enterprise/**` ships its own `LICENSE.txt`, is mounted as the `enterprise` typer app, and its heavy/optional TRT deps (under `enterprise/inference_compiler/`) must be import-guarded so the base CLI still imports. (#2186, #1679.)

## Required companions
- **Versioning** — no `inference/core/version.py` bump is required (release-time concern, see Not blocking). Ignore `inference_cli/version.py` (CI-managed).
- **Tests** — logic changes need unit tests under `tests/inference_cli/unit_tests/lib/...` mirroring the source tree; user-visible changes update integration tests under `tests/inference_cli/integration_tests/`. (#891 updated both.)
- **CI deps** — new runtime imports must be covered by the unit-test CI install list; `inference` is an optional dep, so new deps typically belong in `requirements/requirements.cli.txt`.
- **Docs** — user-facing command/option changes update `docs/inference_helpers/cli_commands/{benchmark,cloud,workflows,infer,server}.md`. Batch-processing API swagger is exposed externally - tag @Erol444.
- **Enterprise / inference_models** — inference-compiler PRs (#2186) also touch `inference_models/pyproject.toml`, `inference_models/docs/changelog.md`, and weights-provider code — verify they move together.

## Key files & entry points
- `inference_cli/main.py` — root Typer `app` + `--version` callback; mounts `server`/`cloud`/`benchmark`/`workflows`/`rf-cloud`/`enterprise` sub-apps.
- `inference_cli/cloud.py` + `inference_cli/lib/cloud_adapter.py` — SkyPilot VM deploy.
- `inference_cli/benchmark.py` / `inference_cli/lib/benchmark_adapter.py` + `inference_cli/lib/benchmark/**` — benchmarking.
- `inference_cli/workflows.py` / `inference_cli/lib/workflows/**` — workflow image/video processing & result aggregation.
- `inference_cli/lib/roboflow_cloud/{common,config,errors,core}.py` — shared HTTP/retry/config/error taxonomy + `rf_cloud_app`.
- `inference_cli/lib/roboflow_cloud/{data_staging,batch_processing}/{core,api_operations,entities}.py` — `rf-cloud` commands + wire schemas.
- `inference_cli/lib/enterprise/inference_compiler/**` — TRT compiler extension (own LICENSE).
- Tests: `tests/inference_cli/{unit_tests,integration_tests}/**`. Docs: `docs/inference_helpers/cli_commands/**`.

## Reference PRs
- [#2399](https://github.com/roboflow/inference/pull/2399) — feature: Asset Library metadata threaded entity→api_op→core; CI deps.
- [#2062](https://github.com/roboflow/inference/pull/2062) — feature: selectable `InferenceBackend` enum for batch processing (+ swagger).
- [#2143](https://github.com/roboflow/inference/pull/2143) — feature: add job name to batch jobs (+ swagger).
- [#1950](https://github.com/roboflow/inference/pull/1950) — feature: env-configurable `ROBOFLOW_API_REQUEST_TIMEOUT` in `config.py`.
- [#2186](https://github.com/roboflow/inference/pull/2186) / [#1679](https://github.com/roboflow/inference/pull/1679) — feature: enterprise inference-compiler TRT extension (LICENSE, guarded deps, inference_models changelog).
- [#1516](https://github.com/roboflow/inference/pull/1516) — bugfix: thread missing `page_size` through list-ingest-details.
- [#1157](https://github.com/roboflow/inference/pull/1157) — bugfix: glob wildcard for local benchmark images.
- [#1051](https://github.com/roboflow/inference/pull/1051) — bugfix: Windows-safe path split via `os.path`.
- [#891](https://github.com/roboflow/inference/pull/891) — bugfix: deterministic aggregation + `image` provenance key (unit+integration tests).
- [#1150](https://github.com/roboflow/inference/pull/1150) — bugfix: stop recording `execution_time=0` on benchmark error.
- [#354](https://github.com/roboflow/inference/pull/354) — bugfix: `isinstance` for exception check + formatted error status codes.
- [#1099](https://github.com/roboflow/inference/pull/1099) — feature: surface video-processing errors to the caller (`ErrorsInterceptor`).
- [#1211](https://github.com/roboflow/inference/pull/1211) — chore: remove stale data-staging export.
- [#2038](https://github.com/roboflow/inference/pull/2038) — bugfix: rename colliding benchmark short flag `-o`→`-mpi`.

## Related topic skills
When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):
- `review-topic-external-contract-and-silent-fallback`
- `review-topic-backward-compat-and-versioning`
- `review-topic-test-hygiene`
