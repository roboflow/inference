# Workflows 2.0 development examples

Runnable examples for the explicitly selected V2 engine in
`workflows/roboflow_workflows/execution_engine/v2/`.

| Examples | What you can try |
| --- | --- |
| [Live object detection](08-live-detection/README.md) | Mac camera with platform YOLOv8n, separate tensor-native box and label blocks, serial/pipelined execution, and on-screen speed statistics |
| [Bounded pipeline](07-bounded-pipeline/README.md) | Opt-in overlap of runs and pulses: phase timelines, per-source order, block/latest overload, stop/cancel/failure, and serial versus pipelined ResNet-18 on CPU and MPS |
| [Model phases](06-model-phases/README.md) | Trained ResNet-18 with CPU, MPS and batched implementations; branch/join phases, run versus phase comparison, nested crops and gates, active frames and windows, mutation warnings and phase-named errors |
| [Temporal operators](05-temporal-operators/README.md) | Explicit alignment of two cameras and a sensor, time windows, static crops before and crops after T, per-camera best frame, last-PTS mosaics and recollection, plus EOF/missing/late/clock/capacity, stop/failure/restart and compile-time rejections |
| [Finite sources](04-source-lifecycle/README.md) | Independent image and CSV signal sources, timestamps, nested conditionals, output groups and stop/drain |
| [Passive foundation](01-passive-foundation/README.md) | Nested image crops, mosaics, a filtering gate, custom blocks, invalid definitions and metadata measurements |
| [Tensor-native media](03-tensor-native/README.md) | Nested crop/resize predictions, native family round-trips, sparse filtering, provenance and rendered own/root outputs |
| [Sequential parity](02-sequential-parity/README.md) | 45 cases run on the real V1 engine and on V2 with call-by-call comparison, plus 20 V2 capability examples |

Run commands from the repository root with
`PYTHONPATH=workflows:inference_models:stream_vision` and the existing
`roboflow-inference-new` environment. Each directory contains its code,
workflow definitions and usage instructions.
