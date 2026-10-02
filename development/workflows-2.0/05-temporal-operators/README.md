# Temporal operators: alignment, windows and temporal blocks

From the inference checkout, in the existing `roboflow-inference-new` environment:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:workflows:inference_models:stream_vision \
  python development/workflows-2.0/05-temporal-operators/run_demo.py \
  --case rig --output-dir /tmp/workflows-temporal-demo
```

This runs the main graph and writes `evidence.json`, `rig/compiled.json`,
`rig/results.json` and `rig/gallery/index.html`. Use `--list` to see every case
and `--case all` to run them all. Each case checks its own observations and the
command exits non-zero on any mismatch. No model, camera or network is needed.

## Words used here

- **N**: the camera axis of an aligned batch; **T**: the time axis a window adds;
  **C**: crops of one image; **K**: a collection of chosen members.
- **Pulse**: one emission of a source or operator, processed through its route.
- **Operator**: a root-level declaration under `"operators"` that turns pulses of
  one or more domains into new pulses. `v2/align@v1` pairs samples;
  `v2/window@v1` collects successive samples into T. Their outputs are selected
  with `$operators.<name>.<port>`.

## The main graph

[`workflows/rig.json`](workflows/rig.json) is one authored graph. The host adds
only the CSV path, handlers and nothing else.

```text
left camera 40 ms ─┐                         ┌─ sensor 100 ms ── convert (unaligned)
right camera 40 ms ┼─ align "rig" (batch) ── samples [N]
                   │     ├─ static_crop ─────────── halves [N,C]
                   │     ├─ grey_card ───────────── reference [N]
                   │     └─ window "clip" (size 4) collects frames and halves,
                   │           holds the reference
                   │           frames [N,T]   crops [N,C,T]   reference [N]
                   │           ├─ best_frame(frames, reference) ── [N], chosen PTS
                   │           ├─ mosaic(crops) ────────────────── [N,C], last PTS
                   │           └─ crop(frames) ── [N,T,C] ─ mosaic ── [N,T]
                   │                                  └─ mosaic ── [N], last PTS
                   │                                        └─ window "recollect" ── [N,T]
left camera ───────┴─ align "paired" (fields, sensor leads) ── {celsius, image}
```

What to look for in `rig/results.json`:

| Group | Observation |
| --- | --- |
| `paired` | Sensor readings at 0/100/200 ms get frames at 0/80/200 ms. 100 ms is 20 ms from both 80 and 120; ties take the earlier frame. No T axis. |
| `rig_samples` | Eight `[N]` batches: left at `40·i` ms, right at `40·i+5` ms. The batch root has no timestamp; each member keeps its own. |
| `temperatures` | The unaligned branch converts every reading once, independent of the cameras. |
| `clips` | `frames` is `[N,T]` with no window-level timestamp. `best` picks left 80 ms and right 45 ms in the first window: each camera chooses a different time, and the output PTS is the chosen frame's. Mosaics use the `last` member's PTS. |
| `recollected` | Mosaics reduced each window to one image per camera, so a second window can collect them again: `[N,T]` at 120/280 ms (left) and 125/285 ms (right). |

## Cases

| Case | What it shows |
| --- | --- |
| `rig` | The main graph above. |
| `eof-drop`, `eof-partial` | Size-3 windows over 8 pairs. The 2-pair tail is dropped at EOF (counted as `partial_dropped`) or emitted once with `partial="emit"`. |
| `policies` | One `[N,T]` group, four outputs: `first`, `last`, `selected` and `common_or_none` (no timestamp, because members differ). |
| `selection-k1` | Top-1 brightest frame is still a `[N,K]` collection. Collecting it again is a compile error; a mosaic over K reduces it to `[N]`. |
| `missing-drop`, `missing-partial` | The right camera lacks its 85 ms frame. `drop` skips the pair; `partial` emits it with the right position filtered, not compacted. |
| `late` | A 60 ms frame arrives after 85 ms. It is counted as `late` and never paired. |
| `clock-error` | The right camera uses another media clock. The run fails at stage `operator`, naming `rig` and the input. |
| `capacity` | A probe makes the right camera read all frames before the left one starts. With the default `max_pending` (32) the pairs are correct; with 4 the run fails and names the operator, input and bound. Ordering is only guaranteed while retained work fits the declared bound. |
| `admission-one` | `admission_bound=1` still gives full size-3 windows: the operator retains samples, the sources do not. |
| `nested` | Gated child workflows before `align` and after `window`. Dark left frames never enter the child; after the window, only the camera whose last frame is bright runs the second child. |
| `stop` | A handler stops the run on the first pair. Pulses already admitted still drain (how many depends on reader timing), every operator finishes, and every source and operator is closed. |
| `failure` | A block raises inside the second window's pulse. The error names the step, the operator `clip` and its pulse; nothing is emitted afterwards. |
| `restart` | One session started twice gives identical results: operator state belongs to each run. |
| `invalid-dynamic-crop` | `v2/crop` output is dynamic: equal crop counts do not prove stable identity, so a window rejects it. `v2/static_crop` declares stationary regions. |
| `invalid-second-t` | A window over frames that already have T is rejected: one T at most. |
| `invalid-middle-t` | `v2/best_frame` over `[N,T,C]` crops is rejected: a T-oriented collapse needs T last. |
| `invalid-lineage` | `v2/best_frame` comparing a window with the current rig pulse is rejected: equal `[N]` shapes from different pulses do not correspond. |
| `operator-in-child` | Operators are root declarations in this delivery; one inside a child workflow is rejected, not ignored. |
| `passive` | `session.run` with a recorded, timestamped `[N,T]` input works with the same temporal blocks. Operators are rejected in passive definitions; nothing spans two `run` calls. |

## Files

| File | Content |
| --- | --- |
| `sources.py` | `TensorCamera` and `CsvSensor`: plain finite sources. They never wait for, pair with or buffer another source. |
| `blocks.py` | Small ordinary demo blocks. None reads or writes temporal metadata; outputs declare a context policy instead. |
| `fixtures.py`, `data/sensor.csv` | Brightness/PTS tables that make the expected choices obvious. |
| `probes.py` | Read-order instrumentation used only by `capacity`. |
| `host.py`, `inspection.py`, `gallery.py` | Compile, start, collect and describe what the engine delivered. |
| `cases.py`, `run_demo.py` | Named cases and the command line. |

## Current limits

- Operators are declared at the root only. Child workflows can feed them and
  consume their outputs.
- Context policies are `common_or_none`, `first`, `last` and `selected`. There is
  no interval (span) policy yet.
- A window's `hold` field takes the value from the last arrival in the window.
- `align` accepts ungrouped inputs only.
