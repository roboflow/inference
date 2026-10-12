# Workflows 2.0 development examples

Runnable examples for the explicitly selected V2 engine in
`workflows/roboflow_workflows/execution_engine/v2/`.

Start with [34 — Local alpha integration](34-local-alpha/README.md) for the
packaged Python API and reusable detection, tracking and drawing blocks.
Its CPU fixtures need no camera, downloaded model or network. Older performance
examples retain their experimental blocks so their comparisons remain reproducible;
they are not the recommended source of application imports.

From the repository root:

```bash
conda activate roboflow-inference-new
export PYTHONPATH=$PWD:$PWD/workflows:$PWD/inference_models:$PWD/stream_vision
export PYTHONDONTWRITEBYTECODE=1
```

Open an example's README for its command, expected output and prerequisites.
GPU, camera, model and Redis cases are separate from the CPU fixtures; the index
does not imply that every example runs in a plain Python environment.

| Example | What you can try | Requires |
| --- | --- | --- |
| [01 — Passive foundation](01-passive-foundation/README.md) | Custom blocks, nested crops, mosaics, gates and invalid definitions. | CPU fixtures |
| [02 — Sequential parity](02-sequential-parity/README.md) | Run comparable definitions through V1 and V2; inspect conditional and nested behavior. | CPU fixtures |
| [03 — Tensor-native media](03-tensor-native/README.md) | Image provenance, native predictions, filtering, geometry and serialization. | CPU fixtures |
| [04 — Finite sources](04-source-lifecycle/README.md) | Declared image/CSV sources, timestamps, output groups and drain. | CPU fixtures |
| [05 — Temporal operators](05-temporal-operators/README.md) | Alignment, windows, stationary crops, temporal selection and errors. | CPU fixtures |
| [06 — Model phases](06-model-phases/README.md) | ResNet-18 phase graphs, alternative implementations and nested execution. | Cached weights; CPU/MPS |
| [07 — Bounded pipeline](07-bounded-pipeline/README.md) | Overlapping runs, per-source order, overload policies and failures. | CPU fixtures; cached model for model cases |
| [08 — Live detection](08-live-detection/README.md) | Camera detection with separate box/label painters and a display. | Camera and model |
| [09 — Thor multistream](09-thor-multistream/README.md) | The original host-fed detection benchmark with native Jetson decoding. | Thor and RTSP sources |
| [10 — Physical batching](10-thor-batching/README.md) | One model call for a batch; parity and throughput probes. | CPU fake check; Thor for real runs |
| [11 — Shared drawing preparation](11-structural-performance/README.md) | Compare separate painters with batched drawing preparation. | CPU parity; GPU benchmarks |
| [12 — Model loop versus workflow](12-throughput-gap/README.md) | Measure the same held frames through model, blocks and engine. | GPU and cached model |
| [13 — Drawing kernels](13-visualization-kernels/README.md) | Separate/fused painters, image copies, in-place ownership and Triton. | CPU checks; CUDA benchmarks |
| [14 — Concurrent processes](14-concurrent-pipelines/README.md) | Compare one and two independent pipelines on one GPU. | GPU, model and streams |
| [15 — Post-processing kernels](15-postprocess-kernels/README.md) | Experimental batched and segmented NMS with explicit parity checks. | CUDA/Triton |
| [16 — Tiled recording](16-tiled-recording/README.md) | Record a stable annotated grid on separate compositor/writer threads. | GPU and live streams |
| [17 — Reactive workflows](17-reactive-workflows/README.md) | Events, sync/async handlers, managed state, overflow and failures. | CPU fixtures |
| [18 — State machines](18-state-machines/README.md) | Transitions, external signals, stale decisions and lifecycle events. | CPU fixtures |
| [19 — Redis state](19-redis-state/README.md) | The managed-state API with real Redis isolation and atomic operations. | Local Redis |
| [20 — Capture and replay](20-capture-replay/README.md) | Record once; change retrospective thresholds/windows without rerunning a model. | CPU fixtures |
| [21 — Live segmentation](21-live-segmentation/README.md) | Camera masks and the phone-mirror demonstration. | Camera and model |
| [22 — Webcam replay](22-webcam-replay/README.md) | Capture a take, then replay it with different analysis settings. | Camera/model for capture; saved take for replay |
| [23 — Demand and quality](23-demand-quality/README.md) | Output pruning, wants(), and compile-time quality negotiation. | CPU fixtures |
| [24 — Compact segmentation views](24-segmentation-views/README.md) | Small mask grids and explicit full-resolution materialization. | CPU fixtures; GPU for model benchmarks |
| [25 — Runtime controls](25-runtime-controls/README.md) | Change thresholds and enable/disable steps without recompilation. | CPU fixtures |
| [26 — RF-DETR and tracking](26-l40s-tracking/README.md) | The original dGPU detection/tracking benchmark. | dGPU and cached TensorRT packages |
| [27 — Declared video sources](27-active-video/README.md) | Workflow-managed stream sets, batching, detection and drawing. | Thor and RTSP sources |
| [28 — Active segmentation](28-active-segmentation/README.md) | Dense/compact masks with workflow-managed sources and quality comparisons. | Thor, streams and segmentation model |
| [29 — Idle graph updates](29-live-recompilation/README.md) | Prepare and apply compatible changes while a session is idle. | CPU fixtures |
| [30 — Live graph updates](30-active-graph-update/README.md) | Change a running graph while retaining sources and compatible state. | CPU fixtures |
| [31 — Live mask branch](31-live-mask-update/README.md) | Attach a dense-mask branch to a compact preview without another model call. | Thor and segmentation model |
| [32 — Processing reset](32-processing-reset/README.md) | Rebuild downstream processing while sources remain connected. | CPU fixtures; optional Thor qualification |
| [33 — Interactive webcam updates](33-webcam-graph-update/README.md) | Toggle boxes/tails and switch detector models from the terminal. | Mac camera and cached models |
| [34 — Local alpha integration](34-local-alpha/README.md) | Packaged sources, detection, tracking and drawing with events, replay and updates. | CPU fixtures; separate Thor qualification |

| [35 — Thor alpha qualification](35-thor-alpha-qualification/README.md) | Packaged detection, optional tracking and drawing with managed video sources, measured windows and cleanup checks. | Thor, existing RTSP streams and a cached TensorRT package |

V2 is an opt-in Python capability alongside V1. These examples do not change
the server, SDK or stream-manager default engine selection.
