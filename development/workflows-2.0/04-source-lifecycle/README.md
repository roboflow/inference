# Finite sources and lifecycle

From the inference checkout, using an environment with the existing dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:workflows:inference_models:stream_vision \
  python development/workflows-2.0/04-source-lifecycle/run_demo.py \
  --case csv --output-dir /tmp/workflows-source-demo
```

The first command runs a standalone CSV source with no injected resources or other
sources. Use `--case all` to run every example. The demo asserts its observations,
prints group/source/pulse/PTS/observation/status,
and writes `evidence.json` when `--output-dir` is supplied. No camera, model, network,
or additional installation is needed. Use `--case` to run one example:

| Case | What to observe |
| --- | --- |
| `csv` | A plain CSV source converts three temperatures and finishes at EOF. |
| `csv-stop` | An ordinary CSV handler calls `session.stop()` when output arrives; admitted samples drain. |
| `independent` | Five tensor images and three CSV temperatures flow independently through one authored graph. |
| `lifecycle` | An explicit barrier probe proves image output arrives while the temperature reader is held, and images continue after temperature EOF. |
| `stop` | With two image pulses admitted, the first callback requests stop twice; both pulses drain and each source closes once. |
| `failure` | An intentional image handler error is attributed and surfaced after source cleanup. |
| `passive` | `InputValue` preserves source/PTS through an ordinary scalar block without adding a T axis; unwrapped input still works. |
| `invalid-join` | Compilation rejects an output group mixing the two independent sources. |

`workflows/independent_sources.json` is one complete authored graph. The host loads
it intact and supplies static configuration, resources and named handlers. Bright
images enter a gated nested resize workflow; dark images filter both the processed
and forwarded child outputs. Two temperature groups share one conversion step and
one output-free audit action. Its persistent count exposes duplicate execution.

`csv-stop` uses the same plain CSV source and an output handler calling
`session.stop()`. The session is available even if a fast callback runs before
`start()` returns its run handle. Acquisition may have admitted more samples
already, so stop drains them; the final callback count can vary. No startup
barriers or source modifications are needed for this host pattern.

`source_plugins.py` shows class-owned declarations and cooperative `open/read/close`.
The separate `lifecycle_probe.py` subclasses are selected only for the explicit
`lifecycle`, `stop` and `failure` cases. Their `DemoControl` resource uses event
barriers to make acquisition order repeatable; it is test instrumentation, not
required source infrastructure, and does not align samples. The CSV has explicit millisecond
media/observation values. Image timing stays in buffer metadata, while native
`ImageData` retains its own geometry. Sources transfer emitted payload ownership.

`None` from `read()` means EOF. A missing emission port is terminally absent;
`Emission({})` is an explicitly filtered pulse. Source/port completion and filtering
have separate runtime tests. Sequential handlers can backpressure the run; stop
requires cooperative readers or completion of native reads. Timestamps do not
provide alignment, temporal collection, or a production concurrent scheduler.
