---
name: add-inference-model
description: Add a new core (pre-trained, non-user-fine-tuned) model to the inference repos. Covers the inference_models class + registry entry + tests, weight-zip preparation, the registration script against the Roboflow model registry, and the optional surfaces (workflow block, legacy-endpoint adapter). Trigger when the user asks to "add a model", "port a new model from HuggingFace", "wrap a transformers model", "expose X as a workflow block / model", or similar. This is a living skill — iterate on it each time a new model ships.
---

# Adding a new core model

This skill is the end-to-end playbook for shipping a new **pre-trained / core** model. Don't invoke it for:

- user-fine-tuned models (different path — workspace/dataset/version ids)
- new backends for an **existing** architecture (just extend the registry)
- bug fixes inside an existing model (ordinary code change)

## Where the code goes (read this first)

There are two top-level Python packages in this repo:

- `inference_models/inference_models/` — **the canonical place for new model implementations.** All new core models go here.
- `inference/` — the older package. Its `inference/models/<family>/` subfolders (e.g. `inference/models/sam3/`, `inference/models/yolov8/`) hold legacy model implementations that predate `inference_models`. **Treat those as deprecated — never add a new one.** The path forward is always `inference_models/` first, then cross-reference via the surfaces below.

A new model has up to **four surfaces**, and you only wire up the ones you need:

| # | Surface | Required? | Location |
| --- | --- | --- | --- |
| 1 | Model class + registry entry | **Always** | `inference_models/inference_models/models/<family>/` + `models_registry.py` |
| 2 | Weight zips + registry registration script | **Always** (so the model is actually loadable) | GCS test-assets bucket + PR to `roboflow/model-registry-sdk` |
| 3 | Workflow block | Only if it should appear in workflows | `inference/core/workflows/core_steps/models/foundation/<name>/v1.py` |
| 4 | Inference-models adapter | Only if it should serve on a plain `/infer` HTTP endpoint | Add a subclass in `inference/core/models/inference_models_adapters.py`, wire into `inference/models/utils.py` |

Quick guidance: plain HTTP `/infer` endpoint ⇒ add surface 4. Workflow visibility ⇒ add surface 3. Streaming video / stateful trackers typically only need 1-3 (state can't cross an `/infer` request boundary).

## Before scaffolding — survey existing models

`inference_models` has many models already. Before writing a single file, **read 2-3 same-backend, same-task siblings**. They carry patterns you should match (file layout, class naming, `from_pretrained` contract, how they handle device/dtype/quantization, how they shape the registry entry).

```
ls inference_models/inference_models/models/
ls inference_models/inference_models/models/auto_loaders/models_registry.py  # see every registered arch
```

Pick the 1-2 closest to the new model by (backend, task) and read:
- the model class file
- its registry entry
- its unit test under `inference_models/tests/unit_tests/models/test_<family>.py`
- its integration test + fixture if present

Only start scaffolding after you know which existing model yours most resembles. If nothing close exists (new backend, new task), flag that to the user — the skill's templates may not cover the gap.

## Discovery phase — ask the user

Before touching files, get concrete answers:

1. **Architecture name** (registry key string) — lower-case, hyphens OK, no slashes. This is the string matched in `models_registry.py`.
2. **Task type** — choose one of the concrete task constants defined in `inference_models/inference_models/models/auto_loaders/models_registry.py` (for example `OBJECT_DETECTION_TASK`, etc.), and verify the exact service-side accepted string against the model-registry API/schema docs or the model-registry SDK.
3. **Backend** — `HF` / `TORCH` / `ONNX` / `TRT` (or `TORCH_SCRIPT`, `MEDIAPIPE`, etc. — check `BackendType`). Determines which sibling you survey.
4. **Upstream weight source** — HF repo id, internal `.pt`, local files. **If HF and gated, stop** — the user needs to accept terms and supply an `HF_TOKEN` before any download.
5. **Variants** — one id like `clip`, or a family like `foo/{tiny,small,large}` with a default. Variants share one registry entry; variant resolution lives in the weights provider.
6. **Which surfaces?** Ask explicitly: workflow block? plain `/infer` adapter? If both, both get wired.
7. **Any existing legacy implementation under `inference/models/<family>/`?** If yes, note that the new `inference_models` implementation is the replacement — don't delete the legacy in the same PR unless the user asks, but do avoid depending on it.

Write the plan back to the user and get an OK before moving on.

## Step-by-step

### 1. Model class (surface 1a)

Create `inference_models/inference_models/models/<family>/__init__.py` and `<family>/<family>_<backend>.py`. The **only** hard contract is a classmethod:

```python
@classmethod
def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "YourModel": ...
```

`model_name_or_path` points at a directory containing every file registered for the model package.

If your model fits a standard category, extend the corresponding base class (see `inference_models/docs/contributors/adding-model.md` for the catalog: `ObjectDetectionModel`, `InstanceSegmentationModel`, `ClassificationModel`, `KeyPointsDetectionModel`, `SemanticSegmentationModel`, etc.). If it doesn't, a standalone class is fine — base classes exist for consistency, not as a hard requirement.

For **shared plumbing across several HF models** (sessioned video trackers, etc.), check `inference_models/inference_models/models/common/` for reusable bases before writing your own.

Read `inference_models/docs/contributors/adding-model.md` and `inference_models/docs/contributors/writing-tests.md` — they cover the `from_pretrained` contract in more depth than this skill.

### 2. Registry entry (surface 1b)

Edit `inference_models/inference_models/models/auto_loaders/models_registry.py`. Add:

```python
("<architecture>", <TASK_CONSTANT>, BackendType.<BACKEND>): LazyClass(
    module_name="inference_models.models.<family>.<family>_<backend>",
    class_name="<YourClass>",
),
```

The key is **only** `(architecture, task, backend)` — **not** the variant. All variants of the same family share one entry. The variant suffix in the model id (e.g. `foo/tiny`) is resolved server-side by the weights provider. Don't add `"<arch>-tiny"` as a separate architecture.

Use `RegistryEntry` (instead of `LazyClass` directly) if the model has optional features like fused NMS — see existing `yolov8` entries for the pattern.

### 3. Unit tests — inference_models side

Create `inference_models/tests/unit_tests/models/test_<family>.py`. Mock the backend library (transformers / onnxruntime / torch) so the test runs without weights. Copy the structure from a nearby test that targets the same backend.

**Run from `inference_models/` cwd** so pytest uses `inference_models/pytest.ini` (or pass `-c inference_models/pytest.ini` from the repo root):

```bash
cd inference_models
python -m pytest tests/unit_tests/models/test_<family>.py -W ignore
```

Running from the repo root without `-c inference_models/pytest.ini` silently mis-collects.

### 4. Integration test + fixture

Add a `..._PACKAGE_URL` constant and a `pytest.fixture(scope="module")` in `inference_models/tests/integration_tests/models/conftest.py` (follow existing patterns near other HF / torch packages). Add `inference_models/tests/integration_tests/models/test_<family>_predictions.py` marked `@pytest.mark.slow`. These run after step 6 uploads.

### 5. Workflow block (surface 3, optional)

Skip this section unless surface 3 is needed. Create `inference/core/workflows/core_steps/models/foundation/<family>/v1.py` + `__init__.py`. Read 1-2 existing blocks that match your pattern (stateless per-image vs. stateful per-video-session) before writing.

Block manifest fields to get right:

- `model_id` default — use the variant-qualified id, e.g. `"foo/small"` (not bare `"foo"`)
- `examples` — list every shipping variant
- `get_supported_model_variants()` — list every variant; **put the default first** (used as display name by the air-gapped cache scanner in `inference/core/cache/air_gapped.py`)

If the block holds per-video or otherwise per-request state, raise `NotImplementedError` in `__init__` when `step_execution_mode is StepExecutionMode.REMOTE` — remote sharding breaks stateful blocks. Fail at workflow-compile time, not at first-frame.

Register the block with the block loader (grep for an existing block's name in `inference/core/workflows/core_steps/loader.py` or similar to find the registration site).

Add unit tests at `tests/workflows/unit_tests/core_steps/models/foundation/test_<family>.py` — mock the inner `AutoModel.from_pretrained` and the model's inference call so the test isolates the block's branching/decision logic. Run from repo root:

```bash
python -m pytest tests/workflows/unit_tests/core_steps/models/foundation/test_<family>*.py -W ignore
```

### 6. Weight zips

For each variant, produce a **flat** zip — files at zip root, **no wrapping directory**. The test fixture `download_model_package` unzips the archive and calls `YourClass.from_pretrained(unzipped_dir)`; nested layouts break silently.

Typical fetch + zip from HF:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="…", local_dir=out_dir, allow_patterns=[...])
# then: zip every file at the root, no wrapping dir
```

Verify each zip with `unzip -l <zip> | head -10` — first column should be bare filenames, not `wrapper/config.json`.

**Smoke-test the zip before uploading** by extracting to a temp dir and loading:

```python
YourClass.from_pretrained(unzipped_dir, device="cpu")
```

Upload to `gs://roboflow-tests-assets/rf-platform-models/<arch>-<variant>.zip`. Confirm each URL returns 200:

```bash
curl -sI https://storage.googleapis.com/roboflow-tests-assets/rf-platform-models/<arch>-<variant>.zip | head -1
```

### 7. Registration script (surface 2)

Clone `roboflow/model-registry-sdk` if you haven't. Add a script at `scripts/core_models/register_<family>_models.py`. Browse `scripts/core_models/` for existing same-backend templates; copy the nearest one and swap constants.

Shape of the registration flow (see the SDK's `registration_helpers.execute_model_package_registration` and the bare methods on `TheGOATModelsServiceClient`):

```python
client = TheGOATModelsServiceClient(
    api_host=API_HOSTS[env],            # staging=api.roboflow.one, prod=api.roboflow.com
    service_secret=os.environ["MODELS_SERVICE_INTERNAL_SECRET"],
)
client.register_pre_trained_model(model_id=f"{arch}/{variant}", model_architecture=arch,
                                  model_variant=variant, model_access=..., task_type=...)
reg = client.register_model_package(file_handles=[...], package_manifest=...)
for spec in reg.file_upload_specs:
    upload_from_local_file(source_file=local, target_uri=spec.gcs_uri)
client.confirm_model_package_artefacts(..., seal_model_package=True)
```

Note: the base SDK's client **does not** have a `.init(current_environment=...)` classmethod — that helper only exists on the `exp-registry-migration` repo's vendored copy (and it reads from GCP Secret Manager). Script constructs the client directly from an env var + hardcoded host per `--env`.

Open a PR against `roboflow/model-registry-sdk`. Do not run against production yet.

### 8. Inference-models adapter (surface 4, optional)

Skip unless the user wants plain `/infer` endpoint support. Add a subclass of `Model` to `inference/core/models/inference_models_adapters.py` matching your task (there are per-task parents: object detection, instance segmentation, classification, keypoints, semantic segmentation, etc. — read the existing adapters in that file). In the adapter `__init__`, follow the existing adapter constructors in that file: they call `AutoModel.from_pretrained(model_id_or_path=..., ...)` and pass through the additional flags they need (for example `allow_untrusted_packages`, `allow_direct_local_storage_loading`, backend selection, etc.), then store the result; predict / infer methods delegate.

Register the adapter by model architecture in `inference/models/utils.py` so `/infer?model_id=<arch>/<variant>` resolves to it. Follow the pattern other entries in that file use.

Most new models on this path will NOT need surface 4 — workflow blocks (surface 3) cover the majority use case. Add 4 only if there's a concrete requirement.

### 9. Run registration against staging

The user sets `MODELS_SERVICE_INTERNAL_SECRET` once per shell:

```bash
export MODELS_SERVICE_INTERNAL_SECRET=$(gcloud secrets versions access latest \
    --secret=MODELS_SERVICE_INTERNAL_SECRET --project=878913763597)
# 878913763597 = staging project; confirm before running
```

Then a per-variant smoke test first, then the full set:

```bash
python scripts/core_models/register_<family>_models.py --env staging --variants <one>
python scripts/core_models/register_<family>_models.py --env staging
```

Verify via the staging API (needs a staging Roboflow API key):

```bash
curl -s "https://api.roboflow.one/models/v1/external/weights?modelId=<arch>/<variant>" \
    -H "Authorization: Bearer <staging-api-key>" | python3 -m json.tool
```

Expect `status: ok` and `modelPackages[0].packageFiles` listing every file with `md5Hash` set.

### 10. End-to-end verify

Run `AutoModel.from_pretrained("<arch>/<variant>", api_key=<staging-key>)` against staging (set `ROBOFLOW_ENVIRONMENT=staging` or `ROBOFLOW_API_HOST=https://api.roboflow.one`) and exercise the model with a real input. If surface 3 was built, also run it through `debugrun.py` / a short MP4 via `InferencePipeline`. If surface 4 was built, hit `/infer?model_id=<arch>/<variant>`.

**When running `debugrun.py` or the inference server from the repo root**, avoid letting the repo-root `inference_models/` directory shadow the editable-installed `inference_models` package. On newer Python versions that support it, you can use `PYTHONSAFEPATH=1` (or `python -P`) so Python does not auto-add the script directory to `sys.path`. **Do not rely on `python -P` on Python 3.10**. For Python 3.10, prefer running from an installed environment via `python -m ...` instead of invoking a repo-root script directly, or adjust your `PYTHONPATH` / working directory so the repo-root namespace package is not on `sys.path`.

## Gotchas (real, collected as hit)

Add to this list as new surprises surface.

- **HF gating**: some `facebook/*` repos (e.g. `facebook/sam3`) return 401 on every file without an `HF_TOKEN`. Accept terms on the model page + generate a token before any download.
- **Zip layout**: files at the zip root, no wrapping directory. The fixture unzips and calls `from_pretrained(that_dir)` — nested layouts break silently.
- **Nested-list shape for HF video processors**: some processor methods expect inputs at a very specific nesting depth (e.g. `input_boxes` at 3 levels `[image [boxes [coords]]]`, not 4). Unit tests that mock the processor won't catch wrong nesting — always include one integration or e2e test that exercises the real `from_pretrained` + predict path against real weights, even if tiny-variant.
- **State-requiring `.track()` / similar must raise on missing state**, not silently create an empty session. Empty-state-then-silent-success bugs are hard to detect.
- **Numpy array truthiness**: `dict.get(a) or dict.get(b)` raises on numpy arrays. Use explicit `"a" in d` / `"b" in d` checks, or a small `_first_present` helper.
- **SDK client auth**: `TheGOATModelsServiceClient.init(current_environment=...)` doesn't exist on the base SDK — only on `exp-registry-migration`'s vendored client. Our scripts construct the client directly from the env var + hardcoded host per `--env`.
- **Transformers import-time side effects**: some transformers model classes (e.g. SAM3 video) do `import torchvision` at module import. Missing torchvision surfaces as `ModuleNotFoundError: Could not import module 'Sam3VideoModel'` — misleading. Not a prod issue, but confuses local setup.
- **Stateful workflow blocks + remote execution**: if your block keeps per-video or per-request state, raise `NotImplementedError` in `__init__` when the execution mode is `REMOTE`. Failing at compile time beats failing on first frame.
- **`get_supported_model_variants` order**: the first entry is the display name for the air-gapped cache scanner. Put your default variant first.
- **`PYTHONSAFEPATH=1`** when running scripts from the repo root — see step 10.

## Verification checklist

Before declaring done:

- [ ] Architecture registered in `models_registry.py`; import + class resolve without error
- [ ] Every variant zip uploads and `curl -sI` returns 200
- [ ] `inference_models` unit tests pass (from `inference_models/` cwd)
- [ ] If surface 3: workflow-block unit tests pass (from repo root)
- [ ] Registration script merged or at least open as a PR against `roboflow/model-registry-sdk`
- [ ] `register_*_models.py --env staging` completes without errors (run per-variant smoke test first)
- [ ] Staging metadata API returns the model with every file + MD5 + sealed
- [ ] `AutoModel.from_pretrained("<arch>/<default>")` loads + runs against staging
- [ ] If surface 3: block runs end-to-end on a real input (image or MP4 via `InferencePipeline`)
- [ ] If surface 4: `/infer?model_id=...` returns a valid prediction
- [ ] `make style` clean
- [ ] At least one non-mock integration test exercises the real call path
- [ ] PR descriptions list remaining TODOs (other variants, production registration, additional surfaces deferred)

## Production registration

Only after staging is fully verified and the user explicitly approves:

```bash
export MODELS_SERVICE_INTERNAL_SECRET=$(gcloud secrets versions access latest \
    --secret=MODELS_SERVICE_INTERNAL_SECRET --project=481589474394)  # prod
python scripts/core_models/register_<family>_models.py --env production
```

## Iterating on this skill

Each new model added either confirms an assumption here (leave alone) or surfaces a gap (add a gotcha / template note). Non-HF backends (ONNX, TRT, TORCH) are underrepresented in today's templates — the next model through a non-HF path should add a step-1 note for its backend.
