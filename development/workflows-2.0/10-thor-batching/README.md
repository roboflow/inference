# Thor physical batching: V2 detection → boxes → labels in batches of N

The 09 benchmark with real model batches. N frames go into one workflow run;
the detector calls the model's `pre_process`, `forward` and `post_process`
once for all N. Box and label painters stay per-image blocks: N calls each.
Jetson Thor for live runs and TensorRT parity; the fake-model check runs on
any CPU.

```
N frames ─ one V2 run, WorkflowBatchInput image ─┬─ detector: 1 call  ─ forward(N x 3 x H x W)
                                                 └─ boxes: N calls ── labels: N calls
```

| `--mode` | What runs per batch |
| --- | --- |
| `v1_tensor` | V1 engine, stock blocks, one `engine.run` on a list (the 09 V1 batch runner) |
| `v2_serial` | V2 `session.run`, one batch at a time |
| `v2_pipeline` | V2 `session.pipeline`, up to `--pipeline-depth` batches at once; detector phases and painters are separate stages |

Run from the inference checkout with the sources on `PYTHONPATH`, as for 09.
This directory reuses the 09 sources, admission, metrics and TRT loading.

## Live run

```sh
PYTHONPATH=".:workflows:inference_models:stream_vision${PYTHONPATH:+:$PYTHONPATH}" \
  python development/workflows-2.0/10-thor-batching/run_batched.py \
    --mode v2_pipeline --batch-size 8 --pipeline-depth 2 --sources 16 \
    --warmup-seconds 10 --duration-seconds 30 \
    --output-dir /tmp/thor/batched/v2_pipeline_b8_d2_s16
```

Batch collection is 09's: take the first waiting frame, then wait up to
`--collect-timeout-ms` (default 3) for the batch to fill to `--batch-size`.
Admission lets at most `batch size x depth` frames in flight (depth is 1
except in `v2_pipeline`). Sources keep only their latest frame meanwhile.

Outputs are those of 09 `run_batched.py`: `summary.json`, `frames.csv` and
`batches.csv`. Each `frames.csv` result also names its `image_id`,
`batch_index`, `batch_size` and `batch_started_ns` (monotonic, when the backend
received the batch). With the frame's `admit_ns` and `ready_ns` that splits its
latency into collection wait and batch execution. Add
`--record-forward-batches` to count the batch sizes reaching TRT `forward`.
In `v2_pipeline`, `engine_run_ms` in `batches.csv` is submit-to-ready and
batches overlap, so do not sum it.

## Parity

```sh
# Any CPU, seconds: deterministic fake model
python development/workflows-2.0/10-thor-batching/check_parity.py fake \
    --output-dir /tmp/thor/batched-parity-fake

# Thor, TensorRT: the held 09 fixture (omit --fixture to capture a new one), one process per mode
python development/workflows-2.0/10-thor-batching/check_parity.py run \
    --fixture /fixtures/parity/fixture.pt --batch-size 8 --output-dir /tmp/thor/batched-parity
```

Both include probe batches of 1, 3 and 5 where they fit, reserving a full
batch of B whenever there are B frames, then the rest. For 16 frames at B = 8,
the plan is `[1, 3, 8, 4]`. `fake` mixes three frame sizes inside batches; `run`
shuffles the captured frames with a fixed seed, so batches mix sources. They check:

- the batch sizes reaching `forward` equal that plan;
- every row carries its own image id, its own frame size in the prediction
  metadata, its batch's index and size, and (fake) its own frame's box;
- batched results equal the 09 backend of the same engine, one frame per call:
  exactly with the fake model; within the 09 tolerances with a real model,
  because another batch size may select other kernels. Every non-identical
  frame is listed with its measured box and confidence differences;
- `v2_pipeline` only: the same pipeline backend, with `depth` batches pending
  in a shuffled order, returns for each batch exactly its rows (count and ids),
  equal to its own rows when batches go one at a time. In the fake check, no
  model phase runs twice at once;
- `run` only: all modes agree exactly with each other (09 `compare_modes`).

`run --device cpu` uses ONNX instead of TensorRT. It is a local check of the
code paths, not TensorRT parity, and its report says so.

## Diagnostic: where does frame time go?

`diagnose.py` times one ladder on the same held frames, without sources:
`forward` (network only) → `model` (pre, forward, post) → `blocks.<stage>`
(09 blocks, no engine) → `v2.<stage>` (09 per-image V2) → `v2b.<stage>`
(batched V2). Stages are `detector`, `boxes`, `full`. A case id reads
`<family>[.<stage>]:b<B>:<execution>`, e.g. `v2b.full:b8:pipe2`; `--list-cases`
prints the selected grid without importing anything heavy.

```sh
# Thor, TensorRT, held frames from the parity fixture: throughput numbers
python development/workflows-2.0/10-thor-batching/diagnose.py \
    --fixture /fixtures/parity/fixture.pt --output /tmp/thor/diagnose.json

# Any CPU: fake model and synthetic frames (--fixture takes any 09 fixture.pt too)
python development/workflows-2.0/10-thor-batching/diagnose.py --fake-model --device cpu \
    --synthetic-frames 4 --synthetic-size 360x640 --frames 8 --warmup 2 \
    --output /tmp/diagnose-smoke.json

python development/workflows-2.0/10-thor-batching/diagnose.py --list-cases --cases 'v2*'
```

`--nvtx` and `--profile-case ID` (an `nsys --capture-range=cudaProfilerApi`
window) make a run instrumented, and the JSON says so. Use instrumented runs to
find where time goes; take throughput only from runs without them.

## Python API

`batched_backend.build_for_model(mode, model=..., model_id=..., confidence=...,
pipeline_depth=2, stages="full")` returns a backend with
`process_batch(frames, image_ids=...)` and, for `v2_pipeline`,
`submit_batch(...)`. `pipeline_depth` counts batches in flight.
`stages="detector"` or `"boxes"` cuts the V2 workflow for ablations, and
`batched_backend.v2_workflow(stages, batched=False)` builds the per-image 09
workflow with the same cut. Call 09 `backends.configure_mode(mode)` first.
`thor_imports` is how 10 reaches 08/09 code: import it first; use
`thor_imports.load_09_module` only for 09 `run_batched` and `check_parity`,
whose names 10 reuses.
