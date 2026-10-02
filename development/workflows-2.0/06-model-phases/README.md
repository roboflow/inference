# Model phases: a trained classifier as a phased block

From the inference checkout, in the existing `roboflow-inference-new` environment:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:workflows:inference_models:stream_vision \
  python development/workflows-2.0/06-model-phases/run_demo.py \
  --case all --output-dir /tmp/workflows-model-phases
```

This writes `evidence.json` and `gallery/index.html`. Use `--list` to see the
cases. Each case checks its own observations; the command exits non-zero on
any mismatch.

## Weights

The examples use torchvision ResNet-18 `IMAGENET1K_V1` (ImageNet-1K, 1000
classes), file `resnet18-f37072fd.pth`, 46,830,571 bytes, SHA-256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
`run_demo.py` never downloads. It looks in `--weights-dir`, or in torch's
checkpoint directory (`$TORCH_HOME/hub/checkpoints`, by default
`~/.cache/torch/hub/checkpoints`), and verifies size and hash.

On a machine without the file, fetch it once, explicitly:

```sh
python development/workflows-2.0/06-model-phases/prepare_assets.py \
  --download-to ~/models/resnet18
# then: run_demo.py --weights-dir ~/models/resnet18 ...
```

Without `--download-to`, `prepare_assets.py` only verifies. Images are existing
repository files (`dogs.jpg`, `dog.jpeg`, `car.jpg`); `assets.py` pins their
hashes.

## The block

`model_demo/flip_averaged_classifier@v1` is one logical block with one set of
Params and outputs, and three implementations:

| Implementation | Requires | Phases | Note |
| --- | --- | --- | --- |
| `mps` | `mps` | yes | Network on Apple Metal; `result` copies outputs to the CPU |
| `cpu-batched-views` | `cpu`, `batched_views` | none | One `[2, 3, 224, 224]` forward; runs as `run` in phase mode |
| `cpu` | `cpu` | yes | Default for `Target.cpu()` |

The compiler picks the first one whose requirements the target meets. The
phased implementations share one graph:

```text
image ─ tensor ─┬──────────── logits ─────────┐
                └─ flipped ─ flipped_logits ──┴─ probabilities ─ result
```

`run()` calls the same phase methods, so `CompileOptions(block_execution="run")`
and `"phases"` give bitwise-identical predictions on one device. Outputs are
native `ClassificationPrediction` objects whose `images_metadata` comes from
`ImageData.prediction_metadata()`, plus `top_class` and `confidence`.
A failing phase is named in both modes: the engine raises
`StepExecutionError` with `.phase` set and the original exception as
`__cause__`. A direct `run()` call raises `PhaseFailure` (also with
`.phase` and `__cause__`).

## Cases

| Case | What it shows |
| --- | --- |
| `classify` | Two photos, crops of each, and a nested workflow that classifies only crops with mean brightness ≥ 120. The beagle photo reads basset then beagle; its head crop reads beagle. The dark backpack crop runs no phase at all. Run and phase mode predictions are compared field by field. |
| `direct` | `CpuResNet18(...).run(image=...)` and `run_phases(...)` outside the engine, equal to the engine's phase mode. |
| `selection` | Choice per target, with the reason for every alternative. `cpu-batched-views` has no phases, so phase mode runs it whole. A `cuda` target is rejected at compile time. The workload names the weights without reading them. |
| `mps` | Runs on this host's MPS device when available (skipped otherwise). Run and phase mode are bitwise equal on MPS; CPU and MPS agree within 1e-3. CUDA is not supported here. |
| `active` | A source replays the three images at 0/40/80 ms. Predictions keep each frame's PTS and root; a lower band is classified in a child workflow unless dark; a window collects the labels with their own PTS. |
| `mutation` | `model_demo/redact_region@v1` greys the image in place in its `result` phase. An unordered reader warns, `mutation_conflicts="error"` rejects, and the ordered reader classifies the redacted pixels. |
| `phase-error` | A region outside the image fails in phase `bounds`; the error names the step and phase in both modes. |
| `private-phase` | `$steps.classify.probabilities` is not an output; selecting it fails. |
| `authoring-errors` | A phase cycle, an unknown phase parameter and an implementation restating the contract all fail when the class is defined. |

## Files

| File | Content |
| --- | --- |
| `classifier.py` | The logical block and its implementations |
| `resnet18.py` | Model primitives shared by every implementation |
| `redaction.py` | The in-place mutating phased block |
| `sources.py` | `model_demo/still_frames`, a finite timed image source |
| `assets.py`, `prepare_assets.py` | Pinned identities, verification, explicit download |
| `host.py` | Catalogue, compilation for a target and mode, passive and active runs |
| `native_comparison.py`, `observations.py` | Field-by-field prediction comparison and result reading |
| `model_cases.py`, `contract_cases.py`, `active_case.py`, `cases.py` | The cases |
| `gallery.py`, `run_demo.py` | Gallery page and command line |

Tests: `workflows/tests/unit_tests/execution_engine/v2/test_model_phase_examples.py`
uses untrained weights for plumbing checks; its real-model test runs this demo
when the pinned weights are present (`RESNET18_WEIGHTS_DIR` or torch's
checkpoint directory) and is marked `slow`.
