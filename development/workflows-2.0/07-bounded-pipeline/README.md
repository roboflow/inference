# Bounded pipeline: overlapping runs and pulses of one session

From the inference checkout, in the existing `roboflow-inference-new` environment:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:workflows:inference_models:stream_vision \
  python development/workflows-2.0/07-bounded-pipeline/run_demo.py \
  --weights ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  --output-dir /tmp/workflows-bounded-pipeline
```

This writes `evidence.json` and `index.html`. Use `--list` for the examples
and `--case <name>` for one of them. Each example checks its own observations;
the command exits non-zero on any mismatch. Nothing is downloaded: the model
example verifies the pinned ResNet-18 weights of
[06-model-phases](../06-model-phases/README.md) (`--weights`, or `--weights-dir`).

## Turning it on

```python
from roboflow_workflows.execution_engine.v2.pipelining import PipelineOptions

# Passive: bounded submissions; run() is refused while the pipeline is open.
with session.pipeline(options=PipelineOptions(max_in_flight=3)) as pipeline:
    futures = [pipeline.submit({"image": image}) for image in images]
results = [future.result() for future in futures]

# Active: up to max_in_flight pulses at once; per-source overload policy.
run = session.start(
    handlers=handlers,
    admission_bound=2,
    pipeline=PipelineOptions(max_in_flight=2, overload="latest"),
)
```

Without `pipeline` options both stay serial, the reference behaviour.

## Writing a block for pipelined runs

When the host compiles with `CompileOptions(block_execution="phases")` and
pipelines, two pulses can be in different phases of the same block instance
at the same time. A stage runs one call at a time, in order, so a single
phase never runs twice at once. But `self` is shared by every phase.

| Your block | Safe when pipelined? |
| --- | --- |
| Plain `run()` (no phases), state on `self` | Yes: the whole call is one ordered stage. Counts stay 1, 2, 3, 4. |
| Data passed from one phase to the next as return values | Yes. Phases still overlap across pulses. |
| A `self` field written by one phase and read by another | **No.** Another pulse can overwrite it in between, and the result is silently wrong. Set `phase_overlap = False`, or pass the value as a return value. |
| A field that one phase alone reads and writes | Ordered within that phase's stage. The per-phase gate does not make every write to `self` safe: if any other phase reads or writes the field, see the row above. |
| A resource injected into several steps (a shared client, model or cache) | Not covered by any block's gate. Steps are gated separately, so the resource needs its own lock or coordination. |

`phase_overlap = False` makes the whole call one stage, including resolving
its host Futures. The next pulse enters the block only after this call
finished:

```python
class MyBlock(Block):
    phase_overlap = False   # phases share scratch on self
```

`--case authoring` shows the unsafe block and both fixes side by side.
Probe events force the overlap; nothing sleeps.

Test your own phased block with two submissions. The serial run and the
default `block_execution="run"` never show this bug:

```python
plan = compile_workflow(definition, catalogue=catalogue,
                        options=CompileOptions(block_execution="phases"))
with plan.create_session(resources).pipeline(options=PipelineOptions(max_in_flight=2)) as p:
    futures = [p.submit({"value": 0.0}), p.submit({"value": 1.0})]
```

To show the bug reliably, hold the first submission inside its later phase
until the second has entered the earlier one (an event, as in
`bounded_authoring.py`), then compare with `session.run`. A real observer or
logging can change the interleaving and hide the bug. Reproduce the scratch
bug without one.

## Examples

| Example | Workload | What it shows |
| --- | --- | --- |
| `timeline` | synthetic | Passive run 1 enters phase `first` while run 0 is held inside `second`; with `phase_overlap = False` it waits for the whole call. Scores equal serial runs. |
| `active-timeline` | synthetic | Frame `a1` overlaps `a0` by phase; source `a` is delivered in frame order only after `a0` is released, while unrelated source `b` is delivered completely meanwhile. Same results as serial. |
| `overload` | synthetic | One held consumer. `block`: all 10 frames, the reader waits for a slot. `latest`: the held frame and the newest one; 8 stale frames are counted as dropped; result age is reported. |
| `lifecycle` | synthetic | `stop` delivers every admitted frame; `cancel` stops admitted frames at their next stage (a running call finishes); a step failure is attributed to its frame. In each case `read = admitted + dropped + unadmitted`, `admitted = processed + cancelled`, and no run thread survives. A failed passive submission aborts the one behind it; the session runs again afterwards. |
| `authoring` | synthetic | Writing a block for pipelined runs. An intentionally unsafe phased block keeps scratch on `self` between phases: pipelined it returns `[10, 10]` instead of `[0, 10]`, with no error; its serial run is correct. Passing the scratch as a phase return value, or setting `phase_overlap = False`, gives `[0, 10]` under the same test. A run-mode counter with state on `self` counts `1, 2, 3, 4`. |
| `model` | trained ResNet-18 | The M3 classifier (imported, not copied) on CPU, and on MPS when available: every pipelined prediction is bitwise equal to the serial one and is matched to its input by its own `images_metadata`. The M3 active workflow (frames, gated child crops, window) gives the same frames and windows in both modes. |

**Synthetic** means blocks that compute nothing and record when each call
enters and leaves. Probe events force each interleaving, and the timeline is a
logical order, not a time axis. No example sleeps.

## Evidence limits

- The `model` seconds are single wall-clock observations on this host. They
  are not a performance claim and say nothing about CUDA.
- Counters count work items (runs, pulses, calls), never bytes or device memory.
- A passive pipeline holds at most `max_in_flight` running submissions plus
  one prepared submission waiting for a worker. Inputs and returned Futures
  stay the caller's: the engine does not copy inputs, so submitting one
  mutable object twice shares it between the two runs.
- A slow synchronous handler backpressures every source of the run: there is
  one bounded worker pool and no hidden queue. Without an observer, other
  pulses keep computing until they need a free worker, an admission slot or
  their own delivery turn.
- A session observer couples more strongly. Handlers, observer callbacks and
  the error handler of a pipelined run never run at the same time. So while a
  handler runs, every other worker waits at its next step callback, and a
  slow handler stops compute, not only delivery.

## Files

| File | Content |
| --- | --- |
| `bounded_authoring.py` | Shared-scratch trap, safe phase return values and whole-call fix |
| `bounded_probes.py` | Synthetic probe blocks, source and event recorder |
| `bounded_scheduling.py` | `timeline`, `active-timeline`, `overload`, `lifecycle` |
| `bounded_model.py` | Real-model comparison, importing `06-model-phases` |
| `bounded_report.py`, `run_demo.py` | Report page and command line |
| `workflows/*.json` | Probe workflow definitions |

Tests: `workflows/tests/unit_tests/execution_engine/v2/pipelining/test_bounded_pipeline_examples.py`
runs the synthetic examples, and the model example when the pinned weights are
present (marked `slow`).
