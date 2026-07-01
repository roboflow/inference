---
name: review-cli-cloud-tooling
description: Review guidance for PRs changing inference_cli/ (except server.py) — the `inference` CLI: benchmark, workflows batch/video processing, `rf-cloud` data-staging + batch-processing, and enterprise inference-compiler. Enforces Typer command conventions, RF-cloud HTTP/retry/pydantic-alias contracts, error-handling, and version/test companions.
---

# Reviewing cli-cloud-tooling changes

## Scope
Triggers on changes under `inference_cli/` **except** `inference_cli/server.py` (that Docker-container / server-lifecycle surface is a different skill). In scope:
- `inference_cli/benchmark.py`, `inference_cli/workflows.py`, `inference_cli/cloud.py`, `inference_cli/main.py`
- `inference_cli/lib/benchmark/**` (api_speed, dataset, results_gathering, inference_models_speed)
- `inference_cli/lib/workflows/**` (core, common, entities, local/remote image + video adapters)
- `inference_cli/lib/roboflow_cloud/**` (common, config, errors, `data_staging/**`, `batch_processing/**`)
- `inference_cli/lib/enterprise/**` (inference-compiler TRT extension)
- Companion tests under `tests/inference_cli/**` and CLI docs under `docs/inference_helpers/cli_commands/**`.

OUT of scope: `inference_cli/server.py`, `inference_cli/lib/container_adapter.py`, `tunnel_adapter.py`, `infer_adapter.py` (server/container lifecycle skill); the Workflows Execution Engine internals and blocks (separate skill); `inference_sdk`.

## What this surface is
A Typer-based CLI (`app` in `main.py`) that mounts sub-apps: `server`, `cloud`, `benchmark`, `workflows`, `rf-cloud`, `enterprise`. The command layer (`*/core.py`, `benchmark.py`, `workflows.py`) is thin: it parses `typer.Option`s, resolves `api_key` (falling back to `ROBOFLOW_API_KEY`), and delegates to `api_operations.py` / adapter modules. Contracts a reviewer protects:
- **CLI signature = public contract.** Option long/short names, defaults, and command names are user-facing. Renaming/removing an option or its short flag is a breaking change; adding must be backward compatible (new options default to `None`/existing behavior).
- **RF-cloud HTTP contract.** All `rf-cloud` API calls go through `get_workspace()` + `handle_response_errors()` in `roboflow_cloud/common.py`, wrapped in `@backoff.on_exception(..., RetryError, max_tries=3)`, using `REQUEST_TIMEOUT` from `roboflow_cloud/config.py`. Retryable codes are `HTTP_CODES_TO_RETRY = {408,429,502,503,504}`; 401 → `UnauthorizedRequestError`; other 4xx/5xx → `RFAPICallError`.
- **Pydantic wire schema.** Request/response entities in `*/entities.py` map snake_case ⇄ camelCase JSON. Outbound bodies use `serialization_alias` (e.g. `batchId`, `partName`, `imagesMetadataPart`); inbound responses use `alias`. The JSON keys are the server API contract — they must match the backend, not be renamed casually.
- **Error taxonomy.** `RoboflowCloudCommandError` → `RetryError`, `RFAPICallError` → `UnauthorizedRequestError` (`roboflow_cloud/errors.py`). Retry logic keys off exception type; do not raise a bare `Exception` where a `RetryError` is expected.

## Standards enforced here
- **Typer option style.** New params are `Annotated[Optional[T], typer.Option("--long", "-short", help="...")] = None`, snake_case Python name, kebab-case CLI flag, with a `help=` string. Every command that hits the network takes `api_key` (defaulting to `ROBOFLOW_API_KEY`) and a `--debug-mode/--no-debug-mode` flag. Evidence: `roboflow_cloud/data_staging/core.py:30-56`, PR #1516, #2143.
- **Unique short flags per command.** Short flags must not collide within a command; a duplicate silently shadows. Evidence: PR #2038 renamed benchmark `--model_package_id` short flag `-o`→`-mpi` because `-o` already meant something else; be suspicious of reused single letters.
- **Standard command error envelope.** Command bodies wrap the delegate call in `try/except KeyboardInterrupt` → `print("Command interrupted.")` + `raise typer.Exit(code=2)`, and `except Exception as error:` → if `debug_mode: raise error` else `typer.echo("Command failed. Cause: {error}")` + `raise typer.Exit(code=1)`. Non-zero exit codes are the contract for scripting. Evidence: `data_staging/core.py:49-56`; `main.py:136-138`.
- **HTTP calls reuse the helpers, not raw requests handling.** New API ops call `get_workspace()` + `handle_response_errors(response, operation_name=...)`, are decorated with `@backoff.on_exception(backoff.constant, RetryError, max_tries=3, interval=1)`, and pass `timeout=REQUEST_TIMEOUT`. Connectivity/`Timeout` exceptions are re-raised as `RetryError`. Evidence: `roboflow_cloud/common.py:26-83`; PR #1950 made the timeout configurable via `ROBOFLOW_API_REQUEST_TIMEOUT`.
- **Tunables live in `config.py` and are env-overridable.** Constants like `REQUEST_TIMEOUT`, `MAX_SHARDS_UPLOAD_PROCESSES`, `MAX_DOWNLOAD_THREADS`, `MAX_IMAGE_REFERENCES_IN_INGEST_REQUEST` use `int(os.getenv("NAME", default))`. Do not hardcode magic numbers in api_operations. Evidence: `roboflow_cloud/config.py`; PR #1950.
- **Enums for closed choice sets.** Machine/backend/format options are `str, Enum` in `*/entities.py` (`MachineType`, `MachineSize`, `InferenceBackend`, `AggregationFormat`), so Typer validates the choice. Evidence: `batch_processing/entities.py:63-109`; PR #2062 added `InferenceBackend`.
- **New request field is optional + threaded end-to-end.** A new capability adds an `Optional[...] = None` field on the entity (with `serialization_alias`), a matching param on the `api_operations` trigger fn, and the `core.py` command option — all three layers. Evidence: PR #2399 (metadata mapping through entity→api_operations→core), #2062, #2143.
- **Enterprise extension isolation.** `inference_cli/lib/enterprise/**` ships its own `LICENSE.txt` and is mounted as the `enterprise` typer app; heavy/optional deps (TRT) must be import-guarded so the base CLI still imports. Evidence: PR #2186, #1679.

## Required companions
- **Version bump.** Nearly every merged PR here bumps `inference/core/version.py` `__version__` (e.g. #2399 `1.3.0`→`1.3.1`, #1051, #891). A functional CLI change with no version bump should be flagged. `inference_cli/version.py` also carries a `__version__` — check whether the PR's release convention touches it.
- **Tests.** Logic changes need unit tests under `tests/inference_cli/unit_tests/lib/...` mirroring the source tree (`roboflow_cloud/`, `workflows/`); user-visible behavior changes update integration tests under `tests/inference_cli/integration_tests/`. Evidence: PR #891 updated both unit `test_common.py` and integration `test_workflows.py` (column-count assertions) alongside the fix.
- **CI requirements.** If a change pulls in new runtime imports, the unit-test CI install list must cover them — PR #2399 extended `.github/workflows/unit_tests_inference_cli_x86.yml` to add `_requirements.txt`/`requirements.cpu.txt`. New deps also belong in `requirements/requirements.cli.txt`.
- **Docs.** User-facing command/option changes should update `docs/inference_helpers/cli_commands/{benchmark,cloud,workflows,infer,server}.md`. Batch-processing API-shape changes are mirrored in `docs/workflows/batch_processing/*_swagger.json` (updated in #2062, #2143, #1211).
- **Enterprise / inference_models companions.** inference-compiler PRs (#2186) also touch `inference_models/pyproject.toml`, `inference_models/docs/changelog.md`, and weights-provider code — verify those move together.

## Common pitfalls & past regressions
- **#2038 — colliding short flag.** Benchmark `--model_package_id` reused short `-o`; Typer let the later option shadow. Check every new short flag is unique within its command.
- **#1157 — glob without wildcard.** `glob(os.path.join(directory, e))` matched nothing because `IMAGE_EXTENSIONS` are bare extensions; fix was `f"*{e}"`. Watch local-image discovery / path-glob code in benchmark & workflows.
- **#1051 — Windows path split.** `result_path.split("/")[-2]` broke on Windows separators; replaced by `os.path.basename(os.path.dirname(path))`. Flag any manual `"/"` splitting of filesystem paths — use `os.path`.
- **#891 — non-deterministic aggregation + missing `image` key.** Batch results were aggregated without `sorted(all_processed_files)` and without stamping the source `image` name, so rows were unordered/unlabeled. Check batch-result aggregation sorts inputs and preserves provenance.
- **#1150 — fabricated `execution_time=0` on error.** On a failed request the code recorded `execution_time=0`, polluting latency stats; fix stopped passing it so errors don't count as zero-latency successes. Check benchmark error paths don't inject fake metrics.
- **#354 — `issubclass(exc, ...)` on an instance.** Exception handling called `issubclass` on an instance (needs `isinstance`), and error status codes weren't human-formatted. Check exception-type checks and stats formatting.
- **Trailing-comma tuples.** A stray trailing comma turns a value into a 1-tuple (`x = (foo,)`), silently corrupting option/arg construction. Watch for accidental trailing-comma tuples in CLI option/arg building.
- **#1516 — missing pagination param.** `list-ingest-details` couldn't set page size; the fix threaded `page_size` through core→api_operations→the paginating generator. When adding a listing command, verify pagination is actually plumbed, not just accepted.
- **#1211 — stale export left behind.** Removing a data-staging feature must remove its now-dead export/command, not leave a dangling symbol.

## Review checklist
1. Does the change touch only in-scope paths (not `server.py`/container/tunnel)? If it also touches server surface, defer that part.
2. New/changed `typer.Option`s: `Annotated[Optional[T], ...] = None`, kebab-case flag, `help=` present, short flag unique within the command, backward-compatible default.
3. Network commands: resolve `api_key` from `ROBOFLOW_API_KEY`, expose `--debug-mode`, and use the standard `try/except KeyboardInterrupt (exit 2) / except Exception (debug re-raise else exit 1)` envelope.
4. New API ops go through `get_workspace()` + `handle_response_errors(operation_name=...)`, carry `@backoff.on_exception(..., RetryError, ...)`, pass `timeout=REQUEST_TIMEOUT`, and re-raise connectivity/`Timeout` as `RetryError`.
5. New request fields are `Optional[...] = None` with correct `serialization_alias` (outbound camelCase) / `alias` (inbound), threaded through entity → api_operations → core command.
6. Closed choice sets are `str, Enum`, not free-form strings.
7. Filesystem paths use `os.path` (no manual `"/"` split); globs include `*`; listing/aggregation is deterministic (`sorted`) and preserves provenance.
8. No magic numbers in api_operations — tunables live in `config.py` as `os.getenv`-backed constants.
9. Error/benchmark paths don't fabricate metrics (no `execution_time=0`) and use `isinstance` for exception-type checks.
10. Companions present: `inference/core/version.py` bump; unit/integration tests updated; CI install list & `requirements.cli.txt` cover new deps; relevant docs / `batch_processing/*_swagger.json` updated.
11. Removals also delete dead exports/commands (#1211).

## Key files & entry points
- `inference_cli/main.py` — root Typer app + `--version`; sub-app wiring.
- `inference_cli/benchmark.py` / `inference_cli/lib/benchmark/{api_speed,dataset,results_gathering}.py` — benchmarking.
- `inference_cli/workflows.py` / `inference_cli/lib/workflows/{core,common,entities}.py` + `*_adapter.py` — workflow image/video processing & result aggregation.
- `inference_cli/lib/roboflow_cloud/common.py`, `config.py`, `errors.py` — shared HTTP/retry/config/error taxonomy.
- `inference_cli/lib/roboflow_cloud/{data_staging,batch_processing}/{core,api_operations,entities}.py` — `rf-cloud` commands + wire schemas.
- `inference_cli/lib/enterprise/inference_compiler/**` — TRT compiler extension (own LICENSE).
- `inference/core/version.py` — the version bumped by CLI PRs.
- Tests: `tests/inference_cli/{unit_tests,integration_tests}/**`. Docs: `docs/inference_helpers/cli_commands/**`, `docs/workflows/batch_processing/*_swagger.json`.

## Reference PRs
- [#2399](https://github.com/roboflow/inference/pull/2399) — feature: Asset Library metadata threaded entity→api_op→core; version bump + CI deps.
- [#2062](https://github.com/roboflow/inference/pull/2062) — feature: selectable `InferenceBackend` enum for batch processing.
- [#2143](https://github.com/roboflow/inference/pull/2143) — feature: add job name to batch jobs (+ swagger).
- [#1950](https://github.com/roboflow/inference/pull/1950) — feature: env-configurable `ROBOFLOW_API_REQUEST_TIMEOUT` in `config.py`.
- [#2186](https://github.com/roboflow/inference/pull/2186) / [#1679](https://github.com/roboflow/inference/pull/1679) — feature: enterprise inference-compiler TRT extension (LICENSE, guarded deps, inference_models changelog).
- [#1516](https://github.com/roboflow/inference/pull/1516) — bugfix: thread missing `page_size` through list-ingest-details.
- [#1157](https://github.com/roboflow/inference/pull/1157) — bugfix: glob wildcard for local benchmark images.
- [#1051](https://github.com/roboflow/inference/pull/1051) — bugfix: Windows-safe path split via `os.path`.
- [#891](https://github.com/roboflow/inference/pull/891) — bugfix: deterministic aggregation + `image` provenance key (unit+integration tests).
- [#1150](https://github.com/roboflow/inference/pull/1150) — bugfix: stop recording `execution_time=0` on benchmark error.
- [#354](https://github.com/roboflow/inference/pull/354) — bugfix: `isinstance` for exception check + formatted error status codes.- [#1099](https://github.com/roboflow/inference/pull/1099) — feature: surface video-processing errors to the caller (`ErrorsInterceptor`).
- [#1211](https://github.com/roboflow/inference/pull/1211) — chore: remove stale data-staging export.

## Related topic skills

When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):

- `review-topic-external-contract-and-silent-fallback`
- `review-topic-backward-compat-and-versioning`
- `review-topic-test-hygiene`
