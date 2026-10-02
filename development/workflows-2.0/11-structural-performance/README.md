# Structural performance: shared drawing preparation

The 10 batched detector workflow, with three ways to draw boxes and labels.
Box and label painters stay separate steps in every variant.

```
image path (all variants):  source ── boxes (clone) ── labels (in place) ── output

per-painter     detector predictions ──> boxes and labels (each reads the GPU)
shared-prep     detector ── prep ── drawing ──> boxes and labels (per-image calls)
batch-painters  detector ── prep ── drawing ──> boxes and labels (one call per batch)
```


| `--drawing` | Engine calls per run of B frames (B = 8) | Device-to-host transfers per run |
| --- | --- | --- |
| `per-painter` | 1 + 2B (17), 10 unchanged | 6 per image with detections, 2 per empty image |
| `shared-prep` | 2 + 2B (18) | 1 |
| `batch-painters` | 4 (4) | 1 |

`prep` returns one `DetectionDrawing` per image: a new read-only host snapshot
of `xyxy`, `class_id`, `confidence` and the label texts, built per call. It
transfers raw bytes, so no value is rounded. The `predictions` output stays
the detector's native `Detections` on its device.

`shared-prep` versus `per-painter` shows the effect of one shared prep.
`batch-painters` versus `shared-prep` shows painter delivery only: same prep,
same transfer, and the batch painters call the per-image painter for each
image, with its own readiness wait. What changes is the number of engine
calls and their engine coordination overhead. The image path stays sequential.

The batch wrappers are measurement probes, not a pattern to copy for every
block: they still paint one image at a time. This example's carrier fixes label
text to `<class> <confidence:.2f>` and requires class names even in boxes-only
runs. A general drawing API would need independent label formatting.

Transfer counts above come from the code; they are not measured transfer costs.
The number of calls does not count all per-image validation work.

Run from the inference checkout, as 10:

```sh
conda activate roboflow-inference-new
export PYTHONPATH=".:workflows:inference_models:stream_vision${PYTHONPATH:+:$PYTHONPATH}"
EXAMPLE=development/workflows-2.0/11-structural-performance
```

## Parity (exact)

```sh
# Any device, seconds: scripted model with 0-40 boxes per frame
python $EXAMPLE/check_parity.py fake --device cpu --output-dir /tmp/m46/parity-fake

# Real model on held frames of a 09 fixture: TensorRT on cuda, ONNX on cpu
python $EXAMPLE/check_parity.py run --fixture /fixtures/parity/fixture.pt \
    --output-dir /tmp/m46/parity-trt
```

Both run `v2_serial` and `v2_pipeline` with every drawing over the 10 batch
plan (single, partial and full batches). They require `torch.equal` of boxes,
class ids, confidences and annotated pixels against `v2_serial/per-painter`,
plus the 10 batch checks, pipeline stress and unchanged source frames.

```sh
python -m pytest $EXAMPLE/tests
```

## Held-frame timing

```sh
python $EXAMPLE/run_structural.py --fixture /fixtures/parity/fixture.pt \
    --output /tmp/m46/structural.json
python $EXAMPLE/run_structural.py --list-cases

# CPU smoke: scripted model, synthetic frames
python $EXAMPLE/run_structural.py --fake-model --device cpu --synthetic-frames 4 \
    --synthetic-size 360x640 --frames 16 --warmup 2 --windows 2 \
    --output /tmp/m46/structural-smoke.json
```

Cases are `<drawing>:b<B>:<serial|pipeN>`, default B 1 and 8, serial and
pipe2. Window `w` runs every case once, starting at case `w`. Before timing,
an untimed pass checks that all cases of one B produce identical outputs.
The JSON holds per-window frames/s, raw per-run timings, engine residual,
CUDA peak memory, RSS, GC counts, engine calls per run, and sha256 of the
loaded example modules and of the V2 engine sources. Compare only cases with
the same B, execution and `capacity_frames` (`comparison_groups`).

Serial timing covers call to return. The inherited pipeline timer covers only
**submit-return to ready-observation**: it excludes blocking inside submit and
may exclude work that finished before submit returned. It is not end-to-end
latency. Use total window time for held throughput and decoded-frame age in
the live runner for latency comparisons. Held CUDA peaks are diagnostic only:
all cases share a process, and the inherited pipeline timing callback can retain
results until cyclic GC. Earlier cases also leave allocator reservations. Use
fresh live processes and the lifetime probe for retention claims.

## Live run (Thor, 16 sources)

```sh
python $EXAMPLE/run_live.py --drawing shared-prep --mode v2_pipeline \
    --batch-size 8 --pipeline-depth 2 --sources 16 \
    --output-dir /tmp/m46/live/shared_b8_d2
```

Every option after `--drawing` is a 10 `run_batched.py` option; outputs are
10's. `per-painter` is 10 itself. Shared drawings need a V2 `--mode`.

## Payload lifetime (diagnostic)

```sh
python $EXAMPLE/inspect_lifetimes.py --runs 32
```

Runs a passthrough workflow 32 times with automatic GC switched off in its own
process, then prints how many run payloads are still alive before and after
`gc.collect()`. Alive before collection means reference cycles in the engine
keep them. GC is switched off only to make this visible, not for speed; the
script restores it.

## Python API

```python
import drawing_backend      # installs the 08/09/10 search paths
backends.configure_mode(mode, device=device)   # 09, before Workflows imports
backend = drawing_backend.build_for_model(
    mode, model=model, model_id=model_id, confidence=0.4,
    pipeline_depth=2, stages="full", drawing="batch-painters",
)
rows = backend.process_batch(frames, image_ids=ids)   # or submit_batch (v2_pipeline)
```

Same arguments and backend as 10 `batched_backend.build_for_model`, plus
`drawing`. `drawing_backend.v2_workflow(stages, drawing=...)` returns the
definition; `drawing_blocks.create_catalogue()` compiles all of them.
