---
name: review-inference-models-pkg
description: Load when a PR touches inference_models/inference_models/** (models/base/, models/common/, models/auto_loaders/**, weights_providers/**, runtime_introspection/**, errors.py, configuration.py), inference_models/pyproject.toml, inference_models/uv.lock, or inference_models/docs/**. Diff signals: a REGISTERED_MODELS entry, LazyClass/RegistryEntry, from_pretrained, to_supervision, a BaseInferenceModelsError subclass, an INFERENCE_MODELS_* env var, torch.jit.load/script.
---

# Reviewing inference-models-pkg changes

## Scope
Load when a PR changes any of:
- `inference_models/inference_models/**` — model backends; `models/base/*.py` prediction entities + base model classes; `models/common/` (`onnx.py`, `trt.py`, `torch.py`, `rle_utils.py`, `roboflow/`); `models/auto_loaders/**`; `weights_providers/**`; `runtime_introspection/**`; `errors.py`; `configuration.py`; `constants.py`; `entities.py`.
- `inference_models/pyproject.toml`, `inference_models/uv.lock`.
- `inference_models/docs/**` (`changelog.md`, `errors/`, `models/`, `api-reference/`, `contributors/`, `how-to/`).

OUT of scope (defer to sibling skills): `inference/core/**` model adapters, workflow blocks, the HTTP API, and the `ENABLE_TENSOR_DATA_REPRESENTATION` `_tensor` block pivot. On a cross-cutting PR, review only the `inference_models` side here.

## Review checklist
Severity tags: **BLOCK** = must fix before merge; **FLAG** = raise it; **NIT** = optional.

1. **BLOCK** — Functional change but no version bump in `inference_models/pyproject.toml` (`version = "0.29.x"`) OR no `docs/changelog.md` entry. See *Required companions*.
2. **BLOCK** — Deps changed in `pyproject.toml` but `uv.lock` not regenerated, or shared server deps not mirrored into `requirements/*.txt`.
3. **BLOCK** — New model/backend not registered in `REGISTERED_MODELS`, or missing `from_pretrained(**kwargs)`, resolvable `LazyClass`, integration tests, or `docs/models/*`. See *Standards §1–2*.
4. **BLOCK** — Prediction-entity field shape/semantics (`xyxy`/`class_id`/`confidence`/`mask`) changed non-additively, or a backend / `to_supervision()` / `docs/api-reference/*` left un-updated. See *Standards §3*.
5. **BLOCK** — `errors.py` class renamed/removed without updating the `inference/core` step-error → HTTP mapping. See *Standards §5*.
6. **BLOCK** — Global torch state leak (autocast / dtype / grad context escaping its scope), or a global-mutating loader (`torch.jit.load/script`) not serialized behind the shared lock. See *Standards §4, §7*.
7. **FLAG** — New load-time param passed positionally instead of threaded as an optional `**kwargs` forwarded from `AutoModel`/manager. See *Standards §1*.
8. **FLAG** — New/renamed error class missing its `help_url` anchor or `docs/errors/*` section. See *Standards §5*.
9. **FLAG** — New env var not `INFERENCE_MODELS_*`-named, not validated (`InvalidEnvVariable`), or not in `docs/how-to/environment-variables.md`. See *Standards §6*.
10. **FLAG** — ONNX/TRT batch bounds not respecting both min and max, missing TRT flags (`engine_host_code_allowed`), or no graceful handling of inconsistent artefacts. See *Standards §8*.
11. **FLAG** — Runtime-introspection change can hard-crash negotiation on a missing CUDA/Jetson probe, or has no unit test. See *Standards §9*.
12. **FLAG** — Third-party (`supervision`/`transformers`/`flash-attn`) constructor/kwarg use not `inspect.signature`-guarded (relies on a pinned version instead). See *Standards §3, §10*.
13. **NIT** — New integration/GPU assertion uses brittle exact-equality instead of tolerant bounds (`>=`, loosened tolerance). See *Standards §11*.

### Not blocking
- Do NOT demand a **speed-verification** or **prediction-correctness** artefact from the diff — those live in the PR description / attached evidence, not the tree (see *Required companions*). Confirm they were provided; do not block on their absence from the diff.
- Do NOT block a doc-only, changelog-only, or test-only PR for a missing version bump.
- Do NOT treat a `supervision`-constructor guard as an inference_models concern — that guard lives in the workflows viz block (#2467); on this surface #2467 was only additive `keypoints_detection.py` enrichment.
- Do NOT block a clean **revert** (#2241, #2087) for removing registry/CI/deps/changelog together — that is the correct rollback shape.

## Standards

1. **`from_pretrained` + `**kwargs` contract.** Every model class is loaded via `AutoModel.from_pretrained` (`models/auto_loaders/core.py`) and must implement its own `from_pretrained` classmethod accepting `**kwargs`, so negotiation-derived params can be forwarded. New load-time params are threaded as optional kwargs, not positional (e.g. the TorchScript lock in #2373, `rf_detr_max_input_resolution` in #2145). Canonical: `docs/contributors/adding-model.md`.

2. **Register the model.** New arch/task/backend combos land in `REGISTERED_MODELS` (`models/auto_loaders/models_registry.py`) keyed by `(ModelArchitecture, TaskType, BackendType)` → `LazyClass(module_name, class_name)` or `RegistryEntry`. Feature-gated backends set `RegistryEntry.supported_model_features` / `required_model_features` (e.g. `{"nms_fused"}`). Auto-loader ranking (`models/auto_loaders/ranking.py`) puts ONNX above Torch (`TRT`=7, `ONNX`=6, `TORCH`=5) — do not silently reorder it (#2047).

3. **Prediction-entity shape stability.** Entities live in `models/base/*.py` — `Detections` (`object_detection.py`), `InstanceDetections` (`instance_segmentation.py`), `KeyPoints` (`keypoints_detection.py`), `ClassificationPrediction`/`MultiLabelClassificationPrediction` (`classification.py`). Each is a torch-tensor dataclass with a `to_supervision()` consumed by `inference/core` adapters and Workflows. Do not mutate `xyxy`/`class_id`/`confidence`/`mask` semantics; add new representations as `Union`/opt-in (RLE masks added as `Union[torch.Tensor, InstancesRLEMasks]` + `InstanceSegmentationMaskFormat = Literal["dense","rle"]`, #2260; `KeyPoints` gained `covariance`/`detection_confidence` additively, changelog `0.29.7`). Any field change touches every backend + `to_supervision()` + `docs/api-reference/*`.

4. **Device / dtype discipline.** Tensors stay on-device until `to_supervision()` does `.cpu().numpy()`. No leaked global autocast — a bf16 `torch.autocast` that poisoned the whole process was a real bug (#2363, in `Sam3TrackerPredictor.__init__`, which entered autocast and never exited). Cast processor inputs to the resolved per-device dtype (Jetson bfloat16 fix, changelog `0.29.4`).

5. **Errors carry `help_url` to a real docs anchor.** `errors.py` is a tree under `BaseInferenceModelsError` (`help_url` optional). New classes get a `docs/errors/<page>.md` section and are raised with `help_url="https://inference-models.roboflow.com/errors/<page>/#<lowercased-classname>"` (see `configuration.py`, which raises `InvalidEnvVariable` with the `runtime-environment` anchor). The `inference/core` handlers (`workflows/execution_engine/v1/step_error_handlers.py`, `interfaces/http/error_handlers.py`) map specific classes — `ModelNotFoundError`, `UnauthorizedModelAccessError`, `ModelPackageRestrictedError`, `ModelPackageAlternativesExhaustedError` — to HTTP/workflow errors; renaming/removing one breaks that mapping (#2180 wired violated-env-constraint reporting through it).

6. **Env-var config is validated and documented.** New env flags use `INFERENCE_MODELS_*` naming, are read/validated in `configuration.py` (raising `InvalidEnvVariable` on bad values), and are documented in `docs/how-to/environment-variables.md` + changelog (Triton RF-DETR flags, changelog `0.29.6`).

7. **Concurrency safety for global-mutating loaders.** `torch.jit.load`/`torch.jit.script` mutate a process-global, non-thread-safe registry and must be serialized via `torchscript_global_lock` (`models/common/torch.py`), threaded from the model manager (#2373; `lock=None` is a no-op for single-threaded use). Any new global-state loader needs equivalent serialization.

8. **ONNX/TRT edge-cases.** Batch fast paths respect both `min_batch_size` and `max_batch_size` — a fixed-batch model got under-sized batches when only max was checked (#2112). TRT artefacts are inconsistent: output-name verification is skipped for known-bad artefacts (#1790 YOLO-NAS) and `engine_host_code_allowed` is set where required (#1791 YOLOv10). Watch RF-DETR class remapping / TRT post-processing and cache-path sanitisation (#2075/#1619/#2029).

9. **Runtime introspection degrades gracefully.** CUDA/Jetson/L4T detection (`runtime_introspection/core.py`) is best-effort; a failed probe must not crash negotiation (CUDA-version fix #2515; `JetsonTypeResolutionError`, #1796). New device support must not change platform defaults (#1436 kept MPS non-default on macOS).

10. **Third-party API drift is guarded, not pinned in code.** `transformers`/`flash-attn` drift silently broke OwlV2 / GLM-OCR (#2257/#2266); guard HF-backed models against version drift. supervision-constructor changes belong to the workflows viz block, not here (#2467).

11. **Tolerant test assertions.** New integration/e2e/GPU assertions use tolerant comparisons (`>=`, loosened tolerance), not exact box-count/class-set equality, which flakes on GPU/ONNX (#2026, #1646, #1649, #1809, #1638, #1843). Keypoint/segmentation slicing uses model-declared slot counts, not `-1` inference (#1626 broke multi-class keypoints with `5 + num_classes` / `view(..., -1, 3)`; fix uses fixed `6:` offset + `_key_points_slots_in_prediction`).

## Required companions
Block a functional change unless it carries these (condition → required file):

- **Any functional change** → version bump in `inference_models/pyproject.toml` (`version = "0.29.x"`) AND a `docs/changelog.md` entry under a new `` ## `0.29.x` `` heading with `Added`/`Fixed`/`Changed`.
- **Change reaches the server** → bump `inference/core/version.py` (`__version__`) too (as in #2260).
- **`pyproject.toml` deps changed** → `inference_models/uv.lock` regenerated (`uv sync`); shared server deps also mirrored into `requirements/*.txt` (#2047, #2449 security bumps, #2415/#2510 dep updates).
- **New error class** → `docs/errors/<page>.md` section + `help_url` anchor.
- **New model** → `REGISTERED_MODELS` entry + `docs/models/<model>.md` + a license file for the model-family dir, listed in `docs/models/index.md`.
- **New env var** → `docs/how-to/environment-variables.md` entry + changelog.
- **New model** → tests in `inference_models/tests/`: `integration_tests/models/test_<model>_predictions_<backend>.py` (mandatory — "the most important tests" per `writing-tests.md`), `e2e_platform_tests/` (AutoModel path), and `unit_tests/` for pure logic (ranking, negotiation, post-processing, introspection).

Non-diff-checkable — **verify via the PR description / attached evidence, not the diff** (block approval if absent from the PR, not from the tree):

- **Model registration acknowledgement** → new-model registration details must be acknowledged by code-owners and confirmed done in the PR's GH comments.
- **Speed verification** → new-model contributions must attach speed evidence on NVIDIA L4 GPU and MacBook (MacBook optional if unsupported), comparing our implementation vs. the original. Absence blocks approval.
- **Prediction correctness** → new or substantial model changes must show predictions are reasonable — ideally integration tests asserting meaningful prediction qualities; for modifications, existing tests suffice or manual evidence against the unchanged code. Absence blocks approval.

## Key files & entry points
- `models/auto_loaders/` — `core.py` (`AutoModel.from_pretrained`), `models_registry.py` (`REGISTERED_MODELS`), `auto_negotiation.py`, `ranking.py`, `model_cache_paths.py`.
- `models/base/*.py` — prediction entities + base model classes + `to_supervision()`.
- `models/common/` — `onnx.py`, `trt.py`, `torch.py` (`torchscript_global_lock`), `rle_utils.py`, `roboflow/post_processing.py`.
- `errors.py`, `configuration.py`, `constants.py`, `entities.py`; `runtime_introspection/core.py`; `weights_providers/{core,roboflow,trt_manifest,entities}.py`.
- `inference_models/pyproject.toml`, `inference_models/uv.lock`.
- `inference_models/docs/{changelog.md,errors/,models/,api-reference/,contributors/adding-model.md,contributors/writing-tests.md,how-to/environment-variables.md}`.
- Core-side error mapping (out of scope to change here, but the contract to protect): `inference/core/workflows/execution_engine/v1/step_error_handlers.py`, `inference/core/interfaces/http/error_handlers.py`.

## Reference PRs
- [#2373](https://github.com/roboflow/inference/pull/2373) — serialize TorchScript load/script behind a global lock.
- [#2363](https://github.com/roboflow/inference/pull/2363) — eject leaked bf16 autocast from `Sam3TrackerPredictor`.
- [#2467](https://github.com/roboflow/inference/pull/2467) — additive `keypoints_detection.py` enrichment (supervision-kwarg guard itself is out of scope).
- [#2112](https://github.com/roboflow/inference/pull/2112) — ONNX batch fast path respects min+max bounds.
- [#1626](https://github.com/roboflow/inference/pull/1626) — multi-class keypoint post-processing slot indexing.
- [#2260](https://github.com/roboflow/inference/pull/2260) — additive RLE mask representation + version/changelog companions.
- [#2145](https://github.com/roboflow/inference/pull/2145) — RF-DETR max input resolution + `ModelPackageRestrictedError` + errors doc.
- [#2180](https://github.com/roboflow/inference/pull/2180) — report model-loading failures from violated env constraints (error → core mapping).
- [#2047](https://github.com/roboflow/inference/pull/2047) — AutoLoader ranks ONNX over Torch (ranking + unit tests + docs).
- [#2515](https://github.com/roboflow/inference/pull/2515) — graceful CUDA-version acquisition in runtime introspection.
- [#2449](https://github.com/roboflow/inference/pull/2449) — security dep bumps with mirrored requirements + uv.lock + changelog.

## Related topic skills
When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):
- `review-topic-prediction-integrity`
- `review-topic-backward-compat-and-versioning`
- `review-topic-concurrency-and-resource-safety`
- `review-topic-external-contract-and-silent-fallback`
- `review-topic-test-hygiene`
