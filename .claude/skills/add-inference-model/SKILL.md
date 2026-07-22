---
name: add-inference-model
description: Trigger when the user asks to "add a model", "add a core / pre-trained model", "port a model from HuggingFace", "wrap a transformers model", or "expose X as a workflow block / model". Also when a diff adds a subfolder under inference_models/inference_models/models/<family>/, a new (architecture, task, BackendType) key in auto_loaders/models_registry.py, an adapter in inference/core/models/inference_models_adapters.py, or a block under core_steps/models/foundation/<family>/.
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
| 1 | Model class + registry entry | **Always** | `inference_models/inference_models/models/<family>/` + `auto_loaders/models_registry.py` |
| 2 | Weight packaging + registry registration | **Always** (so the model is actually loadable) | Internal tooling maintained by the Roboflow Inference Core team — see "Surface 2" below |
| 3 | Workflow block | Only if it should appear in workflows | `inference/core/workflows/core_steps/models/foundation/<family>/v1.py` |
| 4 | Inference-models adapter | Only if it should serve on a plain `/infer` HTTP endpoint | Subclass in `inference/core/models/inference_models_adapters.py`, wired into `inference/models/utils.py` |

Quick guidance: plain HTTP `/infer` endpoint ⇒ add surface 4. Workflow visibility ⇒ add surface 3. Streaming video / stateful trackers typically only need 1-3 (state can't cross an `/infer` request boundary).

## Before scaffolding — survey existing models

`inference_models` has many models already. Before writing a single file, **read 2-3 same-backend, same-task siblings**. They carry patterns you should match (file layout, class naming, `from_pretrained` contract, how they handle device/dtype/quantization, how they shape the registry entry).

```
ls inference_models/inference_models/models/
```

Read `inference_models/inference_models/models/auto_loaders/models_registry.py` to see every registered arch. Pick the 1-2 closest to the new model by (backend, task) and read:

- the model class file
- its registry entry
- its unit test under `inference_models/tests/unit_tests/models/test_<family>_<backend>.py`
- its integration test + fixture if present (`test_<family>_<backend>_predictions.py`)

Only start scaffolding after you know which existing model yours most resembles. If nothing close exists (new backend, new task), flag that to the user — the skill's templates may not cover the gap.

## Discovery phase — ask the user

Before touching files, get concrete answers, agreed with codeowners — you must enforce acknowledgement of the proposed values. If you run as a CI agent, comment on the GH PR tagging codeowners to get the response and make that required to pass your review:

1. **Architecture name** (registry key string) — lower-case, hyphens OK, no slashes. This is the string matched in `models_registry.py`.
2. **Task type** — choose one of the concrete task constants defined in `inference_models/inference_models/models/auto_loaders/models_registry.py` (e.g. `OBJECT_DETECTION_TASK = "object-detection"`, `INSTANCE_SEGMENTATION_TASK`, `CLASSIFICATION_TASK`, `KEYPOINT_DETECTION_TASK`, `VLM_TASK`, `EMBEDDING_TASK`, etc.), and confirm the exact accepted task string with the Roboflow Inference Core team (who own the model-registry tooling).
3. **Backend** — check the `BackendType` enum (imported into `models_registry.py`): `HF` / `TORCH` / `ONNX` / `TRT` / `TORCH_SCRIPT` , etc. Determines which sibling you survey.
4. **Upstream weight source** — HF repo id, internal `.pt`, local files. **If HF and gated, stop** — the user needs to accept terms and supply an `HF_TOKEN` before any download.
5. **Variants** — one id like `clip`, or a family like `foo/{tiny,small,large}` with a default. Variants share one registry entry; variant resolution happens server-side (see step 2).
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

For **shared plumbing across several models** (sessioned video trackers, CUDA/ONNX/TRT helpers, etc.), check `inference_models/inference_models/models/common/` (`hf_streaming_video.py`, `cuda.py`, `onnx.py`, `torch.py`, `trt.py`, `model_packages.py`, and the `roboflow/` post-processing helpers) for reusable bases before writing your own.

Read `inference_models/docs/contributors/adding-model.md` and `inference_models/docs/contributors/writing-tests.md` — they cover the `from_pretrained` contract in more depth than this skill.

### 2. Registry entry + variant resolution (surface 1b)

Edit `inference_models/inference_models/models/auto_loaders/models_registry.py`. Add:

```python
("<architecture>", <TASK_CONSTANT>, BackendType.<BACKEND>): LazyClass(
    module_name="inference_models.models.<family>.<family>_<backend>",
    class_name="<YourClass>",
),
```

(`LazyClass` is imported from `inference_models.utils.imports`; note the `module_name` uses the `inference_models.models...` import path, not `inference_models.inference_models...`.)

The key is **only** `(architecture, task, backend)` — **not** the variant. All variants of the same family share one entry. The variant suffix in the model id (e.g. `foo/tiny`) is resolved server-side: the Roboflow weights provider (`inference_models/inference_models/weights_providers/roboflow.py`) reads it from `RoboflowModelMetadata.model_variant` (the `modelVariant` field on the registry-served metadata). Don't add `"<arch>-tiny"` as a separate architecture.

Use `RegistryEntry` (wrapping a `LazyClass` in `model_class=...`) instead of a bare `LazyClass` if the model has optional features — e.g. fused NMS via `supported_model_features={"nms_fused"}`. See the existing `yolov8` entries for the pattern.

### 3. Unit tests — inference_models side

Create `inference_models/tests/unit_tests/models/test_<family>_<backend>.py` (the file name carries the backend, e.g. `test_sam2_torch.py`, `test_glm_ocr_hf.py`). Mock the backend library (transformers / onnxruntime / torch) so the test runs without weights. Copy the structure from a nearby test that targets the same backend.

**Run from `inference_models/` cwd** so pytest uses `inference_models/pytest.ini` (or pass `-c inference_models/pytest.ini` from the repo root):

```bash
cd inference_models
python -m pytest tests/unit_tests/models/test_<family>_<backend>.py -W ignore
```

Running from the repo root without `-c inference_models/pytest.ini` silently mis-collects.

### 4. Integration test + fixture

Add a package-URL constant and a `pytest.fixture(scope="module")` in `inference_models/tests/integration_tests/models/conftest.py` (follow existing patterns near other HF / torch packages). Add `inference_models/tests/integration_tests/models/test_<family>_<backend>_predictions.py` marked `@pytest.mark.slow`. These run once the model package has been published via the internal registration tooling (Surface 2).

Start from loading a **local** package while experimenting, then follow up by uploading that package as a `*.zip` asset and adding a fixture that downloads it at test time, as other integration tests do.

### 5. Changelog companion (required)

Any PR that ships model code in `inference_models` must record the change. The
package version is selected and bumped later by maintainers:

- Add an entry under `## Unreleased` in `inference_models/docs/changelog.md`, using the appropriate `### Added`, `### Changed`, `### Fixed`, or `### Removed` subsection.
- Do not edit `inference_models/pyproject.toml` solely to bump its version.

### 6. Workflow block (surface 3, optional)

Skip this section unless surface 3 is needed. If needed, create the workflow block by following the **`create-workflow-block`** skill (`.claude/skills/create-workflow-block/SKILL.md`) — the end-to-end playbook for authoring a Workflows block (manifest, kinds, `run()`, loader registration, versioning, tests). For a model block: put it under `inference/core/workflows/core_steps/models/foundation/<family>/`, use the variant-qualified model id (e.g. `"foo/small"`) as the `model_id` default, and wire `run()` to `AutoModel.from_pretrained`.

Run the block's unit tests from the repo root:

```bash
python -m pytest tests/workflows/unit_tests/core_steps/models/foundation/test_<family>*.py -W ignore
```

### 7. Surface 2 — weight packaging & registry registration (internal tooling)

**This is handled by internal tooling maintained by the Roboflow Inference Core team.** Preparing the model's weight package, uploading it, and registering the model into the Roboflow model registry (across staging and production) are done with internal scripts, service credentials, and infrastructure that are **out of scope for this skill and not documented here.** Coordinate with the Roboflow Inference Core team to get a new model packaged and registered.

The one part that touches **your** code: the model package is a **flat** set of files — every file at the package root, no wrapping directory — because the loader materializes the package into a plain directory and calls `YourClass.from_pretrained(that_dir)`. Make sure your `from_pretrained` loads from a directory of files with no nested wrapper; nested layouts break silently. Smoke-test the load path locally before handing off for registration:

```python
YourClass.from_pretrained(local_dir, device="cpu")
```

Once the model has been registered by the internal tooling, the integration/e2e tests (steps 4 and 9) and `AutoModel.from_pretrained("<arch>/<variant>")` can load it.

### 8. Inference-models adapter (surface 4, optional)

Skip unless the user wants plain `/infer` endpoint support. Add a subclass of `Model` to `inference/core/models/inference_models_adapters.py` matching your task (there are per-task parents: object detection, instance segmentation, classification, keypoints, semantic segmentation, etc. — read the existing adapters in that file). In the adapter `__init__`, follow the existing adapter constructors: they call `AutoModel.from_pretrained(model_id_or_path=..., ...)` and pass through the flags they need (e.g. `allow_untrusted_packages`, `allow_direct_local_storage_loading`, backend selection), then store the result; predict / infer methods delegate.

Register the adapter by model architecture in `inference/models/utils.py` so `/infer?model_id=<arch>/<variant>` resolves to it. Follow the pattern other entries in that file use.

Most new models on this path will NOT need surface 4 — workflow blocks (surface 3) cover the majority use case. Add 4 only if there's a concrete requirement.

### 9. End-to-end verify

Once the model has been registered (via the internal tooling, step 7), load it and exercise it end to end:

- `AutoModel.from_pretrained("<arch>/<variant>")` (`AutoModel` lives in `inference_models/inference_models/models/auto_loaders/core.py`) with a real input.
- If surface 3 was built, run the workflow block on a real image or a short MP4 via `InferencePipeline` (e.g. `debugrun.py`).
- If surface 4 was built, hit `/infer?model_id=<arch>/<variant>`.

**When running `debugrun.py` or the inference server from the repo root**, avoid letting the repo-root `inference_models/` directory shadow the editable-installed `inference_models` package. On newer Python versions that support it, use `PYTHONSAFEPATH=1` (or `python -P`) so Python does not auto-add the script directory to `sys.path`. **Do not rely on `python -P` on Python 3.10.** For Python 3.10, prefer running from an installed environment via `python -m ...` instead of invoking a repo-root script directly, or adjust your `PYTHONPATH` / working directory so the repo-root namespace package is not on `sys.path`.

## Gotchas (real, collected as hit)

Add to this list as new surprises surface.

- **HF gating**: some `facebook/*` repos (e.g. `facebook/sam3`) return 401 on every file without an `HF_TOKEN`. Accept terms on the model page + generate a token before any download.
- **Package layout**: files at the package root, no wrapping directory. The loader materializes the package and calls `from_pretrained(that_dir)` — nested layouts break silently.
- **Nested-list shape for HF video processors**: some processor methods expect inputs at a very specific nesting depth (e.g. `input_boxes` at 3 levels `[image [boxes [coords]]]`, not 4). Unit tests that mock the processor won't catch wrong nesting — always include one integration or e2e test that exercises the real `from_pretrained` + predict path against real weights, even if tiny-variant.
- **State-requiring `.track()` / similar must raise on missing state**, not silently create an empty session. Empty-state-then-silent-success bugs are hard to detect.
- **Numpy array truthiness**: `dict.get(a) or dict.get(b)` raises on numpy arrays. Use explicit `"a" in d` / `"b" in d` checks, or a small `_first_present` helper.
- **Transformers import-time side effects**: some transformers model classes (e.g. SAM3 video) do `import torchvision` at module import. Missing torchvision surfaces as `ModuleNotFoundError: Could not import module 'Sam3VideoModel'` — misleading. Not a prod issue, but confuses local setup.
- **Stateful workflow blocks + remote execution**: if your block keeps per-video or per-request state, raise `NotImplementedError` in `__init__` when the execution mode is `REMOTE`. Failing at compile time beats failing on first frame.
- **`get_supported_model_variants` order** (workflow-block manifest classmethod, consumed by the air-gapped scanner in `inference/core/cache/air_gapped.py`): the first entry is the display name for the cache scanner. Put your default variant first.
- **`PYTHONSAFEPATH=1`** when running scripts from the repo root — see step 9.
- **Never register `model_config.json` as a package artefact** (hit with yolo26-depth): the auto-loader generates that file itself for offline-loader compatibility, and a registered artefact with the same name fails every load with `CorruptedModelPackageError` ("collides with the config file that inference is supposed to create"). Registered packages carry only the weights + `inference_config.json` (+ per-task extras like `class_names.txt`); `model_config.json` belongs only in local/test packages. If it slips in: unseal → `artefacts/remove` → re-seal fixes it without re-uploading.
- **Ultralytics exports for fixed-size heads**: mirror the predictor's real pre-processing in `inference_config.json` — ultralytics letterboxes with padding value 114 (not 0), RGB, `/255`, no normalization. Also mirror sibling manifests for the dynamic-ONNX `incompatibleProviders: ["CoreMLExecutionProvider"]` exclusion (dynamic-shape graphs break CoreML).

## Verification checklist

Before declaring done:

- [ ] Architecture registered in `models_registry.py`; import + class resolve without error
- [ ] `YourClass.from_pretrained(local_dir)` loads from a flat directory of files (smoke-tested)
- [ ] `inference_models` unit tests pass (from `inference_models/` cwd)
- [ ] `inference_models/docs/changelog.md` entry added under `## Unreleased` (version bump left to maintainers)
- [ ] If surface 3: workflow-block unit tests pass (from repo root)
- [ ] Model packaged & registered via the internal Inference Core team tooling (staging, then production after approval) — coordinate with the team
- [ ] `AutoModel.from_pretrained("<arch>/<default>")` loads + runs once registered
- [ ] If surface 3: block runs end-to-end on a real input (image or MP4 via `InferencePipeline`)
- [ ] If surface 4: `/infer?model_id=...` returns a valid prediction
- [ ] At least one non-mock integration test exercises the real call path
- [ ] PR descriptions list remaining TODOs (other variants, additional surfaces deferred)

## Iterating on this skill

Each new model added either confirms an assumption here (leave alone) or surfaces a gap (add a gotcha / template note). Non-HF backends (ONNX, TRT, TORCH) are underrepresented in today's templates — the next model through a non-HF path should add a step-1 note for its backend.
</content>
</invoke>
