# V2 API audit validation

These commands were run on 30 September 2026 from the public repository root. They inspect public commit `3d45b8712cc428eb01b714f3609346be7c92acc4` with the private plugin snapshot at `18f2c22543b499727093cf180d45e835ba1c911f`.

## Environment

The audit used Python 3.10.19 on macOS ARM64. The temporary environment at `/private/tmp/v2-api-audit/venv` reads existing dependencies from `/Users/damiankosowski/Projects/inference/.venv/lib/python3.10/site-packages` through a `.pth` file, with the following locally installed overrides/additions:

- SciPy 1.14.1, pyzmq 27.2.0, imagecodecs 2025.3.30.
- filetype 1.2.0, python-multipart 0.0.32, requests-mock 1.12.1, pytest-timeout 2.4.0.
- pytest 9.1.1, pytest-asyncio 1.4.0 and their resolved dependencies.

This is an audit environment, not a production lockfile or a clean-wheel installation test. `PYTHONPATH` forces the inspected public and private source trees to take precedence over installed copies. `observations.json` also records the imported app path and selected dependency versions. The private snapshot was produced using `git archive origin/main` after fetching the remote. No plugins working-tree files were changed.

Initial setup failures were environmental: SciPy 1.15.3 failed to load `_spropack.cpython-310-darwin.so`; missing filetype/imagecodecs dependencies stopped collection; the inherited old pytest-asyncio failed fixture setup with `FixtureDef.unittest`. These were resolved in the temporary environment before the successful runs below. The manager test configuration expects pytest-xdist; the focused manager run overrides `addopts` to execute serially.

## Commands and results

The variables below abbreviate the identical environment prefixes used for the final runs. The temporary directories are local execution artifacts and may need to be recreated for a later run. Use the pinned commits above, or record new commits when updating the report.

```bash
AUDIT_PY=/private/tmp/v2-api-audit/venv/bin/python
AUDIT_PLUGINS=/private/tmp/v2-api-audit/plugins
export MPLCONFIGDIR=/private/tmp/v2-api-audit/mpl
export XDG_CACHE_HOME=/private/tmp/v2-api-audit/cache
export PYTHONPATH="$PWD/inference_server:$PWD/inference_model_manager:$PWD/inference_models:$PWD"
```

### Public server behavior

```bash
"$AUDIT_PY" -m pytest \
  inference_server/tests/unit_tests/test_routers_v2_models.py \
  inference_server/tests/unit_tests/test_framework_dispatch.py \
  inference_server/tests/unit_tests/test_v2_infer_input.py \
  inference_server/tests/unit_tests/test_app_auth_scoping.py \
  inference_server/tests/integration_tests/test_dispatch_v2_infer.py \
  inference_server/tests/unit_tests/test_handler_object_detection.py \
  inference_server/tests/unit_tests/test_handler_embeddings.py \
  inference_server/tests/unit_tests/test_handler_vlm.py \
  inference_server/tests/unit_tests/test_handler_interactive_seg_input.py \
  inference_server/tests/unit_tests/test_gateway_contract.py \
  -q --import-mode=importlib
```

**148 passed, 22 warnings.** [Full log](validation/server-tests.log)

The “integration_tests” file in this group uses an ASGI app with a fake gateway and patched auth/registry; it is not trained-model integration. Its serializer expectations follow the current implementation, including `predictions`, which explains why green tests do not establish conformance to PR #2277.

### Direct gateway and registry resolution

```bash
"$AUDIT_PY" -m pytest \
  inference_server/tests/unit_tests/test_gateway.py \
  inference_server/tests/unit_tests/test_framework_model_stat.py \
  -q --import-mode=importlib
```

**92 passed, 22 warnings.** [Full log](validation/gateway-tests.log)

### Typed serializers and action registration

```bash
PYTHONPATH="$PWD/inference_model_manager:$PWD/inference_models:$PWD/inference_server:$PWD" \
  "$AUDIT_PY" -m pytest \
  inference_model_manager/tests/unit_tests/test_serializers_typed.py \
  inference_model_manager/tests/unit_tests/test_registry_defaults.py \
  -q --import-mode=importlib -o addopts=''
```

**47 passed.** [Full log](validation/manager-tests.log)

### Private gateway unit tests

```bash
PYTHONPATH="$PYTHONPATH:$AUDIT_PLUGINS/rf_serving" \
  "$AUDIT_PY" -m pytest \
  "$AUDIT_PLUGINS/rf_serving/tests/unit_tests/test_gateway_contract.py" \
  "$AUDIT_PLUGINS/rf_serving/tests/unit_tests/test_backend_state_vocabulary.py" \
  "$AUDIT_PLUGINS/rf_serving/tests/unit_tests/test_mmp_gateway_load_timeout.py" \
  "$AUDIT_PLUGINS/rf_serving/tests/unit_tests/test_mmp_gateway_worker_errors.py" \
  -q --import-mode=importlib
```

**24 passed.** [Full log](validation/plugin-tests.log)

### Real local MMP endpoint checks

```bash
ENABLE_CONTROL_PLANE_ROUTES=true \
PYTHONPATH="$PYTHONPATH:$AUDIT_PLUGINS/rf_serving" \
  "$AUDIT_PY" -m pytest \
  "$AUDIT_PLUGINS/rf_serving/tests/integration_tests/test_v2_endpoints.py" \
  -q --import-mode=importlib \
  -k 'health_returns_ok or ready_no_preload or info_returns_models or metrics_returns_json or load_missing_model_id or unload_missing_model_id or interface_missing_model_id'
```

**7 passed, 4 deselected, 22 warnings.** [Full log](validation/mmp-integration.log)

The fixture starts a real MMP loop, connects a real `MMPGateway` over a loopback socket and allocates shared memory. Auth is mocked. Selected cases do not request model weights. Unknown/stub model-load cases were deselected, so this run does not verify inference, cold model loading or model-registry access. The control-plane flag is explicitly enabled to avoid confusing the default 403 gate with endpoint failure.

### Design-specific observations

```bash
"$AUDIT_PY" reports/v2-api-gap-2026-09-30/probe.py \
  > reports/v2-api-gap-2026-09-30/observations.json
```

**51 observations recorded.** [Source](probe.py), [output](observations.json)

The probe uses the actual application with legacy/landing routes enabled according to the checkout's configuration. It records all registered V2 methods/paths, all task/action registrations, HTTP status/content type/body, and selected gateway arguments. Large HTML error bodies are truncated with their original character count recorded.

The following dependencies are replaced with deterministic test doubles: general API-key validation, model registry task/action lookup, model loading, model execution, gateway stats/interface and query-image downloads. Real parser and serializer implementations are not patched. The image bytes only satisfy the real format sniffing step; no image-decoding or numerical model claim follows from these probes.

## Remaining verification

- Full V2 contract tests should be derived from the agreed design rather than regenerated from existing responses.
- Execute representative real classification, detection, segmentation, embedding and OCR models with direct and subprocess backends, including cold/warm package selection and actual class metadata.
- Install the workflows extra and compare direct inference with equivalent single-step workflows once the V2 workflow facade exists.
- Verify deployed CPU/GPU behavior, SDK/client handling, authentication boundaries, binary payload round-trips and Prometheus scraping.
- Continue legacy request/response parity separately; it is not replaced by this report.

No product code was changed by the audit. The report, observation script, observation output and test logs are the deliverables.
