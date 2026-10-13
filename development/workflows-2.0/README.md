# Workflows 2.0 development examples

Runnable examples for the explicitly selected V2 engine. Its packages are
`roboflow_workflows.v2.engine`, `roboflow_workflows.v2.data` and
`roboflow_workflows.v2.blocks` (source: `workflows/roboflow_workflows/v2/`).

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
| [13 — Drawing kernels](13-visualization-kernels/README.md) | Separate/fused painters, image copies, in-place ownership and a staged box upload. | CPU checks; CUDA benchmarks |
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
| [36 — Threading](36-threading/README.md) | Compare batch size, pulses in flight and parallel branches; inspect operator emissions, live updates, drain and cancellation. | CPU fixtures |
| [37 — Implementation contributions](37-implementation-contributions/README.md) | Add a host-chosen implementation to an existing public block without changing the block or the workflow definition. | CPU fixtures |
| [38 — Native data](38-native-data/README.md) | Frame records, aligned prediction rows and native semantic results through compiled workflows and the wire. | CPU fixtures |
| [39 — Model cache](39-model-cache/README.md) | Select a detector by model and package id; load each distinct request once. | CPU fake loader |
| [40 — Mask providers](40-mask-providers/README.md) | Ask for binary instance masks the same way from score grids, dense masks or RLE; each answer is labelled reference or approximation. | CPU fixtures |
| [41 — Query foundation](41-query-foundation/README.md) | Validate existing UQL JSON once and evaluate it on plain V2 runtime values. | CPU fixtures |
| [42 — Regions and parent mapping](42-regions-and-parent-mapping/README.md) | Crop, slice, stitch and relabel; each result finds the detection row it was cut for. | CPU fixtures |
| [43 — Task models](43-task-models/README.md) | Six model families in one workflow, `run` versus `phases`, per-image class filters. | CPU fake models; optional cached model |
| [44 — Filter and transform](44-filter-and-transform/README.md) | Per-class confidence and UQL filter/transform chains on detections and masks. | CPU fixtures |
| [45 — Visualization gallery](45-visualization-gallery/README.md) | The ten annotation blocks, chained painters and `mask_quality`, written as PNGs. | CPU fake models |
| [46 — Merge and combine](46-merge-and-combine/README.md) | Combine two detectors, then merge rows into one box. | CPU fixtures |
| [47 — Query gates](47-query-gates/README.md) | The UQL gate `v2/continue_if@v1`, property definition and expression. | CPU fake models |
| [48 — Text and values](48-text-and-values/README.md) | String templates, CSV, first non-empty value and current time with an injected clock. | CPU fixtures |
| [49 — JSON fields](49-json-fields/README.md) | Parse JSON text into named step outputs. | CPU fixtures |
| [50 — Routing and throttling](50-routing-and-throttling/README.md) | Route by value, pass changed values and limit how often a branch runs. | CPU fixtures |
| [51 — Image history](51-image-history/README.md) | Retrieve earlier images or values while keeping camera histories separate. | CPU fixtures |
| [52 — Object cache](52-object-cache/README.md) | Share values across calls; inspect source namespaces, ordering and reset behavior. | CPU fixtures |
| [53 — Multi-model consensus](53-multi-model-consensus/README.md) | Keep detections that several models agree on, with explicit voting and merge rules. | CPU fixtures |
| [54 — Embedding change sampling](54-embedding-change-sampling/README.md) | Compare recent embedding directions and changes, including warmup and finite-window limits. | CPU fixtures |
| [55 — Passive image history](55-passive-image-history/README.md) | Feed repeated calls into a temporal window using the existing execution workers. | CPU fixtures |
| [56 — Detection changes](56-detection-changes/README.md) | Compare before/after detections and inspect removed, persisted and new objects. | CPU fixtures |
| [57 — Overlap analysis](57-overlap-analysis/README.md) | Inspect asymmetric box or mask-contour coverage in comparable coordinates. | CPU fixtures |
| [58 — Crop rollup](58-crop-rollup/README.md) | Map crop predictions into the parent frame, collect them and merge duplicates. | CPU fixtures |
| [59 — Tracking instances](59-tracking-instances/README.md) | ByteTrack per camera, with new and already-seen instances. | CPU fixtures |
| [60 — Compressed image history](60-compressed-image-history/README.md) | Keep the last frames per source as JPEG bytes. | CPU fixtures |
| [61 — Tracker variants](61-tracker-variants/README.md) | SORT, ByteTrack, OC-SORT and BoT-SORT on one scene, and which tracker to choose. | CPU fixtures |
| [62 — Track state and speed](62-track-state-and-speed/README.md) | Lock classes, stabilize boxes and measure speed per track. | CPU fixtures |
| [63 — Shape visualizations](63-shape-visualizations/README.md) | Circle, corner, dot, ellipse, triangle and halo painters chained on one image. | CPU fixtures |
| [64 — Image processing](64-image-processing/README.md) | Grayscale, blur, threshold, morphology and contrast, compared with V1. | CPU fixtures |
| [65 — Image measurements](65-image-measurements/README.md) | Distances, sizes, mask areas and colour counts with explicit calibration. | CPU fixtures |
| [66 — Custom Python with tensors](66-custom-python-tensors/README.md) | Run local dynamic blocks on V2 images and predictions without conversion. | CPU fixtures |
| [73 — Trace and heatmap](73-trace-and-heatmap/README.md) | Whole-frame source-bounded trace and heatmap drawings; resizing supported, cropping rejected before history mutation. | CPU fixtures |
| [74 — Zone drawings](74-zone-drawings/README.md) | Polygon-zone, line-counter and reference-path drawings with shared counter geometry and per-image count selectors. | CPU fixtures |
| [75 — Decorative drawings](75-decorative-drawings/README.md) | Background, recoloring and grid overlays through default V2 discovery. | CPU stand-ins; real services separately |
| [79 — Motion](79-motion/README.md) | Per-source motion and background models on generated CPU scenes. | CPU stand-ins; real services separately |
| [83 — Mask edge snap](83-mask-edge-snap/README.md) | Refine segmentation boundaries against image edges. | CPU fixtures |
| [84 — Local VLM prompt](84-local-vlm-prompt/README.md) | Caller-owned Qwen3.5 prompt models; CPU stand-ins and separate real weights. | CPU stand-ins; real services separately |
| [86 — Segment anything](86-segment-anything/README.md) | SAM2/SAM3 prompts, compact masks and selected classes through caller-owned models. | CPU fake models; real models separately |
| [88 — Remote chat](88-remote-chat/README.md) | Active OpenAI-compatible text/image chat through explicit injected transports. | CPU stand-ins; real services separately |
| [91 — OpenRouter vendors](91-openrouter-vendors/README.md) | Active Kimi, Meta and Zai outputs and native prediction composition. | CPU stand-ins; real services separately |
| [92 — Vendor SDK chat](92-vendor-sdk-chat/README.md) | Anthropic and Gemini with host-owned SDK clients; separate real-provider calls. | CPU stand-ins; real services separately |
| [93 — Compatible vendors](93-compatible-vendors/README.md) | Active Gemma and SpaceXAI wire formats through CPU fake transports. | CPU stand-ins; real services separately |
| [94 — Platform metadata and search](94-platform-metadata-search/README.md) | Search reference images, classify a query and tag its inference result. | CPU fake ports; explicit host ports for platform calls |
| [95 — Platform dataset upload](95-platform-dataset-upload/README.md) | Set asset attributes and submit bounded dataset uploads. | CPU fake ports; explicit host ports for platform calls |
| [96 — Model monitoring](96-model-monitoring/README.md) | Collect inference summaries and report through a caller-owned port. | CPU fake port; explicit host port for platform calls |
| [97 — Vision events](97-vision-events/README.md) | Submit vision events and collect event bundles through caller-owned ports. | CPU fake ports; explicit host ports for services and bundle storage |
| [104 — ONVIF PTZ](104-onvif-ptz/README.md) | Host-owned camera controllers, explicit cleanup and disabled-path checks. | CPU stand-ins; real services separately |
| [106 — Action recognition](106-action-recognition/README.md) | Classify bounded clips from two cameras with a caller-loaded local model. | CPU fake model; optional local model package |
| [107 — SAM video and interactive](107-sam-video-and-interactive/README.md) | Explicit video frame budgets, episode identities and interactive prompts; concept peak-memory and real-backend qualification remain open. | CPU fake models; real models separately |
| [109 — Florence-2](109-local-vlm-wave2/florence/README.md) | Run local Florence text and grounding tasks with explicit coordinate modes. | CPU stand-in; optional local model package |
| [109 — Moondream and Cosmos](109-local-vlm-wave2/moondream-cosmos/README.md) | Chain local object detection and reasoning with downstream consumers. | CPU stand-ins; optional local model packages |
| [109 — Unified Qwen](109-local-vlm-wave2/qwen-unified/README.md) | Choose native Qwen or an explicit OpenRouter backend. | CPU stand-in or fake transport; optional provider/model |
| [110 — Google Vision OCR](110-remote-ocr/google-vision/README.md) | Call direct Google Vision OCR with a host-owned transport and credential. | Fake transport; optional direct provider |

Examples 42–49 use the built-in `create_catalogue()`. It registers the V1
ports beside the older V2 prototypes: `v2/continue_if` (boolean) beside
`v2/continue_if@v1` (UQL), and `v2/box_visualization@v1` /
`v2/label_visualization@v1` beside `v2/bounding_box_visualization@v1` /
`v2/label_visualization@v2`.

V2 is an opt-in Python capability alongside V1. These examples do not change
the server, SDK or stream-manager default engine selection.
