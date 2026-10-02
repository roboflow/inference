# Parallel workflow stream startup (opt in)

`InferencePipeline.init_with_workflow(parallel_startup=True)` overlaps opening
video sources with workflow compilation. After validated manifests are available,
one worker prepares eligible known models while graph construction and block
initialization continue. This uses the actual model manager and actual sources:
metadata, downloads, backend construction and decoder preroll are not repeated.
There are at most two additional startup workers; models and sources each start
sequentially within their own worker. The default startup path is unchanged.

```python
from threading import Event
from inference import InferencePipeline

cancel_startup = Event()
pipeline = InferencePipeline.init_with_workflow(
    video_reference="rtsp://your-camera/live",
    workflow_specification=accepted_workflow,
    workflows_parameters={"model_id": "your-project/1"},
    api_key=accepted_api_key,
    on_prediction=on_prediction,
    parallel_startup=True,
    startup_model_limit=1,
    startup_cancel_event=cancel_startup,
    video_processing_mode="freshest",
    decoding_buffer_size=1,
    source_buffer_filling_strategy="DROP_OLDEST",
    source_buffer_consumption_strategy="EAGER",
)
try:
    print(pipeline.startup_phase_timings)
    pipeline.start(use_main_thread=False)
    # Your application owns the running pipeline here.
    pipeline.join()
finally:
    pipeline.terminate()
    pipeline.join()
```

Call `terminate()` and `join()` when finished, including if the returned pipeline
is never started. Sources already capture when initialization returns. For file
processing, use the normal batch/file settings instead of the live buffering
settings above: bounded WAIT buffering preserves every frame in order. Reconnect
uses the original producer factory after the old capture thread has stopped.

Only typed declared local execution dependencies are eligible. Access-only,
remote, third-party and block-owned models are excluded by their declarations.
Concrete IDs and IDs from ordinary fixed workflow parameters/defaults can load
early; the normal manifest validation and model-ID resolver apply. Image/video
metadata inputs, unresolved parameters, and step-derived IDs stay lazy. Full
runtime input validation still runs on real frames. No dummy workflow invocation
or synthetic model inference is performed. Manifest validation/reference
resolution must finish before dependency discovery; model preparation overlaps
subsequent graph construction and block initialization, including cache hits.

The limit defaults to one new model. Available fixed-cache slots further bound
admission; existing resident models are reused. `startup_model_limit=0` overlaps
source setup with compilation without preparing models. Raising the limit can
load models from branches that never execute, trigger earlier authorization or
allocation failures, and increase peak memory. Loading is serial, but decoder
memory and model initialization peaks coexist. Memory-pressure eviction can
still occur and emits the existing preload warning; evicted models load lazily
when needed. Do not also pass `workflows_dependencies_pre_init`.

A caller-supplied model manager must have exclusive use during startup; it remains
caller-owned on success and failure. Built-in workflow blocks receive a readiness
proxy for the existing ModelsProvider backed by that same manager. A constructor that accesses the manager waits for
model preparation, rather than racing it. Custom model managers/backends must
support construction on a worker thread, and custom blocks must use the supplied
manager to benefit from the barrier. Normal step concurrency is unchanged; this
option does not add concurrent inference calls to a model.

Set `startup_cancel_event` from another thread to request cancellation. Startup
checks it between model/source operations, at transfer, at start, and before
frame execution. Cancellation is cooperative: a native constructor/download may
continue until it returns. Initialization drains workers before releasing sources
and owned model references. Caller-owned models remain available to their owner.
`join()` drains workflow executors before removing owned models without deleting
cached artifacts. Backend constructors remain responsible for releasing resources
when they raise before registering a model.

`WORKFLOW_STARTUP_PHASE` status updates report `sources`, `models`, and
`compilation`, with relative monotonic start/end/duration seconds and outcome.
The same timings are available as `pipeline.startup_phase_timings` after successful
initialization. Reports contain no model IDs, URLs, credentials, or exception
messages. They measure startup phases, not first inference or first result.

Local tests use inert producers/models and real workflow compilation. The supplied
`development/benchmark_parallel_workflow_startup.py` measures synthetic scheduling
overlap with injected delays; it is not a GPU benchmark. CUDA device selection is
carried into startup workers and model initialization is synchronized before
handoff. Native GStreamer/NVDEC, TensorRT context/stream lifetime, cold/warm artifact
latency, multi-GPU placement and peak VRAM/OOM need hardware validation before
production adoption. No production performance claim follows from local tests.

The implementation lives in `streamvision` (coordinator and camera lifecycle)
and `roboflow-workflows` (compiler callback and typed dependency resolution).
The legacy Inference constructor forwards the options and supplies capacity and
owned-manager cleanup; standalone Streamvision treats its provider as caller-owned.
Keep the existing configuration, image codec, platform bindings and observer.
Install the matching locally built wheels for all changed packages when evaluating
this patch; copying the Inference tree alone does not include those changes.
A same-version wheel overlay must force reinstall and verify imported file hashes.

A job controller should set the cancellation event **before** waiting on its
lifecycle lock. The same event must reach startup and be checked before starting
the returned pipeline. Both feature-off and feature-on benchmarks must use the
same immutable image; historical image versions are context, not a control.
