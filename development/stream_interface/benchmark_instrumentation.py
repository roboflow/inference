"""Benchmark-only runtime instrumentation for `benchmark_pipeline_extraction.py`.

Nothing here is imported by production code. The module is installed only inside
processes the benchmark owns: the harness process (`--transport in-process`) and the
stream manager's pipeline child process (`--transport manager`, see
`InstrumentedInferencePipelineManager`). It hooks three existing seams of the legacy
runtime without changing the runtime itself:

* `status_update_handlers` of `InferencePipeline` (FRAME_CAPTURED / FRAME_CONSUMED /
  INFERENCE_COMPLETED) to record which frame ids reached each stage and when they
  were captured (perf_counter for same-process latency, wall clock for cross-process
  latency, since a file source's `frame_timestamp` is synthetic);
* the `on_prediction` sink, wrapped so sink entry is timed and the delivered ids are
  known before the manager's memory sink or the harness sink runs;
* `ModelManagerModelsProvider._infer` / `infer_from_request_sync` /
  `run_tensor_native_inference`, the boundary between workflow blocks and the legacy
  model manager, to time the actual model call and to observe what crosses that
  boundary (numpy arrays, CUDA tensors, ...). This is the model-call time; the whole
  workflow step trace is not;
* `VideoSourcesManager.retrieve_frames_from_sources` / `..._with_policy`, to record
  the exact FRAME_CONSUMED ids a single retrieval call raised, on its own thread,
  when that same call returns `None` (multiplexer batch dropped on stop). Only those
  observed ids are exempt from loss as a terminate-time discard.

Run as a module (`python -m development.stream_interface.benchmark_instrumentation`)
it starts the real stream manager (`manager_app.app.start`) with the pipeline process
class replaced by `InstrumentedInferencePipelineManager`, which installs the hooks in
its own `run()` so they exist under the spawn start method too. The TCP protocol and
the manager's behaviour are untouched; the probe is written to
`$STREAMS_BENCHMARK_PROBE_DIR/<pipeline_id>.json` when the pipeline process ends.
"""

import json
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from inference.core.interfaces.camera.source_reference_sanitizer import (  # noqa: E402
    redact_credentials_in_text,
)
from inference.core.interfaces.stream_manager.manager_app.inference_pipeline_manager import (  # noqa: E402
    InferencePipelineManager,
)

PROBE_DIR_ENV = "STREAMS_BENCHMARK_PROBE_DIR"
PROBE_SCHEMA_VERSION = 3  # 2: `drained`; 3: `terminate_discarded_ids`
MODEL_CALL_METHODS = (
    "_infer",
    "infer_from_request_sync",
    "run_tensor_native_inference",
)
MODEL_CALL_RANGE_NAME = "benchmark.model_call"

RETRIEVAL_METHODS = (
    "retrieve_frames_from_sources",
    "retrieve_frames_from_sources_with_policy",
)

_ACTIVE_PROBE: Optional["BoundaryProbe"] = None
_MODEL_CALL_TIMING_INSTALLED = False
_FRAME_RETRIEVAL_TRACKING_INSTALLED = False


class BoundaryProbe:
    """Thread-safe collector for one pipeline run inside the process that runs it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._captured: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._consumed_ns: Dict[Tuple[int, int], int] = {}
        self._workflow_ns: Dict[Tuple[int, int], int] = {}
        self.consumed_ids: Dict[int, Set[int]] = {}
        self.completed_ids: Dict[int, Set[int]] = {}
        # ids observed as FRAME_CONSUMED during a `VideoSourcesManager` retrieval
        # call that itself returned `None` (terminate-time discard): the multiplexer
        # already consumed these frames but dropped the whole batch on stop.
        self.terminate_discard_ids: Dict[int, Set[int]] = {}
        self._retrieval_local = threading.local()
        # [source_id, frame_id, captured_wall_ns, capture_to_sink_ns, workflow_ns]
        self.sink_entries: List[list] = []
        # [start_wall_ns, duration_ns, method_index, model_index, batch, thread_id]
        self.model_calls: List[list] = []
        self.model_ids: List[str] = []
        self.model_backends: Dict[str, str] = {}
        self.model_input_kinds: Counter = Counter()
        self.model_output_kinds: Counter = Counter()
        self.dropped_events: Counter = Counter()
        self.decoders: List[str] = []
        self._pipeline_ref: Any = None
        self.pending_at_termination: List[List[int]] = []
        # True only once terminate() + join() returned: every completed output has
        # then been dispatched to the sink. None: termination never finished.
        self.drained: Optional[bool] = None
        self.errors: List[str] = []
        self.cuda_trace: Optional["CudaCopyTrace"] = None
        # Outcome of `_maybe_init_cuda_on_child_main_thread`, see its docstring.
        self.cuda_child_main_thread_init: Optional[str] = None

    # -- status updates (FRAME_CAPTURED / FRAME_CONSUMED / INFERENCE_COMPLETED) ----

    def on_status_update(self, update: Any) -> None:
        perf_now, wall_now = time.perf_counter_ns(), time.time_ns()
        try:
            event, payload = update.event_type, update.payload or {}
            with self._lock:
                if event == "FRAME_CAPTURED":
                    key = (payload.get("source_id") or 0, payload["frame_id"])
                    self._captured[key] = (perf_now, wall_now)
                elif event == "FRAME_CONSUMED":
                    key = (payload.get("source_id") or 0, payload["frame_id"])
                    self._consumed_ns[key] = perf_now
                    self.consumed_ids.setdefault(key[0], set()).add(key[1])
                    if getattr(self._retrieval_local, "active", False):
                        self._retrieval_local.consumed.append(key)
                elif event == "INFERENCE_COMPLETED":
                    for source_id, frame_id in zip(
                        payload.get("sources_id", []), payload.get("frames_ids", [])
                    ):
                        key = (source_id or 0, frame_id)
                        self.completed_ids.setdefault(key[0], set()).add(frame_id)
                        consumed = self._consumed_ns.pop(key, None)
                        if consumed is not None:
                            self._workflow_ns[key] = perf_now - consumed
                elif event == "FRAME_DROPPED":
                    # Retire the capture timestamp: a dropped frame never reaches the
                    # sink, so `_captured` would otherwise keep growing for the rest
                    # of the run. Only the drop count is evidence anyone needs.
                    self._captured.pop(
                        (payload.get("source_id") or 0, payload.get("frame_id")), None
                    )
                    self.dropped_events[str(payload.get("source_id"))] += 1
        except Exception as error:  # noqa: BLE001 - never break the runtime
            self.errors.append(f"status update probe failed: {error!r}")

    # -- sink entry ------------------------------------------------------------------

    def wrap_sink(self, sink: Callable[..., Any]) -> Callable[..., Any]:
        def instrumented_sink(predictions: Any, video_frame: Any) -> Any:
            perf_now = time.perf_counter_ns()
            frames = video_frame if isinstance(video_frame, list) else [video_frame]
            try:
                with self._lock:
                    for frame in frames:
                        if frame is None:
                            continue
                        key = (frame.source_id or 0, frame.frame_id)
                        captured = self._captured.pop(key, None)
                        # None when the sink wins the race against
                        # INFERENCE_COMPLETED (the result is enqueued before that
                        # event is sent); export() joins it by frame identity.
                        self.sink_entries.append(
                            [
                                key[0],
                                key[1],
                                captured[1] if captured else None,
                                perf_now - captured[0] if captured else None,
                                self._workflow_ns.pop(key, None),
                            ]
                        )
            except Exception as error:  # noqa: BLE001 - never break the runtime
                self.errors.append(f"sink probe failed: {error!r}")
            return sink(predictions, video_frame)

        return instrumented_sink

    # -- model calls -----------------------------------------------------------------

    def record_model_call(
        self,
        method: str,
        model_id: str,
        model_manager: Any,
        inputs: Any,
        call: Callable[[], Any],
    ) -> Any:
        started_wall = time.time_ns()
        started = time.perf_counter_ns()
        trace = self.cuda_trace
        result = trace.run_model_call(call) if trace is not None else call()
        duration = time.perf_counter_ns() - started
        try:
            input_kind, batch = describe_model_boundary_value(inputs)
            output_kind, _ = describe_model_boundary_value(result)
            public_id = public_model_id(model_id)
            with self._lock:
                if public_id not in self.model_ids:
                    self.model_ids.append(public_id)
                    self.model_backends[public_id] = resolve_model_backend(
                        model_manager, model_id
                    )
                self.model_calls.append(
                    [
                        started_wall,
                        duration,
                        MODEL_CALL_METHODS.index(method),
                        self.model_ids.index(public_id),
                        batch,
                        threading.get_ident(),
                    ]
                )
                self.model_input_kinds[input_kind] += 1
                self.model_output_kinds[output_kind] += 1
        except Exception as error:  # noqa: BLE001 - never break the runtime
            self.errors.append(f"model call probe failed: {error!r}")
        return result

    # -- frame retrieval (terminate-time discard) -------------------------------------

    def record_retrieval(self, call: Callable[[], Optional[list]]) -> Optional[list]:
        """Wraps one `VideoSourcesManager` retrieval call to catch its own discard.

        Collects the FRAME_CONSUMED ids raised on this thread during this call only
        (not the whole run's history). If the call returns `None` (multiplexer
        dropped the batch on stop), those exact ids are the terminate-time discard;
        any other missing id is still classified as loss by `analyze_boundary_loss`.
        """
        local = self._retrieval_local
        local.consumed = []
        local.active = True
        try:
            result = call()
        finally:
            local.active = False
        if result is None and local.consumed:
            with self._lock:
                for source_id, frame_id in local.consumed:
                    self.terminate_discard_ids.setdefault(source_id, set()).add(
                        frame_id
                    )
        return result

    # -- termination / export --------------------------------------------------------

    def note_pending(self, buffer_sink: Any) -> None:
        """Records the (source_id, frame_id) pairs still held by the memory sink."""
        try:
            buffered = list(getattr(buffer_sink, "_buffer", []) or [])
        except Exception:  # noqa: BLE001 - diagnostics only
            buffered = []
        pending = []
        for _, frames in buffered:
            for frame in frames:
                if frame is not None:
                    pending.append([frame.source_id or 0, frame.frame_id])
        with self._lock:
            self.pending_at_termination = pending

    def note_drained(self, drained: bool) -> None:
        with self._lock:
            self.drained = drained

    def note_cuda_child_main_thread_init(self, status: str) -> None:
        with self._lock:
            self.cuda_child_main_thread_init = status

    def note_pipeline_reference(self, pipeline: Any) -> None:
        """Retains the pipeline so `note_decoders` can sample it once its video
        sources have actually initialised (`VideoSource._video` is set inside
        `start()`, not at construction time)."""
        self._pipeline_ref = pipeline

    def note_decoders(self, pipeline: Any = None) -> None:
        pipeline = (
            pipeline if pipeline is not None else getattr(self, "_pipeline_ref", None)
        )
        try:
            self.decoders = [
                type(getattr(source, "_video", None)).__name__
                for source in getattr(pipeline, "_video_sources", [])
            ]
        except Exception as error:  # noqa: BLE001 - diagnostics only
            self.errors.append(f"decoder probe failed: {error!r}")

    def export(self) -> dict:
        with self._lock:
            return {
                "schema_version": PROBE_SCHEMA_VERSION,
                "pid": os.getpid(),
                "consumed_ids": {
                    str(s): sorted(v) for s, v in self.consumed_ids.items()
                },
                "completed_ids": {
                    str(s): sorted(v) for s, v in self.completed_ids.items()
                },
                "terminate_discarded_ids": {
                    str(s): sorted(v) for s, v in self.terminate_discard_ids.items()
                },
                "sink_entries": [
                    (
                        [*entry[:4], self._workflow_ns[(entry[0], entry[1])]]
                        if entry[4] is None
                        and (entry[0], entry[1]) in self._workflow_ns
                        else list(entry)
                    )
                    for entry in self.sink_entries
                ],
                "model_calls": list(self.model_calls),
                "model_call_methods": list(MODEL_CALL_METHODS),
                "model_ids": list(self.model_ids),
                "model_backends": dict(self.model_backends),
                "model_input_kinds": dict(self.model_input_kinds),
                "model_output_kinds": dict(self.model_output_kinds),
                "dropped_events": dict(self.dropped_events),
                "decoders": list(self.decoders),
                "pending_at_termination": list(self.pending_at_termination),
                "drained": self.drained,
                "errors": list(self.errors),
                "cuda_child_main_thread_init": self.cuda_child_main_thread_init,
            }


def public_model_id(model_id: str) -> str:
    """Local package directories are host paths; results identify them by digest."""
    return "<local package>" if os.path.isdir(str(model_id)) else str(model_id)


def resolve_model_backend(model_manager: Any, model_id: str) -> str:
    """Class name of the loaded model behind the (decorated) model manager."""
    manager = model_manager
    for _ in range(8):
        inner = getattr(manager, "model_manager", None)
        if inner is None:
            break
        manager = inner
    try:
        model = getattr(manager, "_models", {}).get(model_id)
    except Exception:  # noqa: BLE001 - diagnostics only
        model = None
    if model is None:
        return "unknown"
    inner_model = getattr(model, "_model", None)
    if inner_model is not None:
        return f"{type(model).__name__}({type(inner_model).__name__})"
    return type(model).__name__


def describe_model_boundary_value(value: Any, depth: int = 0) -> Tuple[str, int]:
    """(kind, batch) of a value crossing the model boundary.

    Kinds: `ndarray`, `torch:cuda`, `torch:cpu`, `numpy_request` (request image dict
    holding a numpy array), `<type>` otherwise. Only the first element of a batch is
    inspected; the batch size is the list length.
    """
    if depth > 4 or value is None:
        return "none", 0
    module = type(value).__module__ or ""
    if module.startswith("torch"):
        return f"torch:{getattr(value.device, 'type', 'unknown')}", (
            int(value.shape[0]) if value.dim() == 4 else 1
        )
    if module.startswith("numpy"):
        return "ndarray", 1
    if isinstance(value, (list, tuple)):
        if not value:
            return "empty", 0
        kind, _ = describe_model_boundary_value(value[0], depth + 1)
        return kind, len(value)
    if isinstance(value, dict):
        if "type" in value and "value" in value:
            kind, _ = describe_model_boundary_value(value["value"], depth + 1)
            return f"{value['type']}:{kind}", 1
        return "dict", 1
    image = getattr(value, "image", None)
    if image is not None and hasattr(value, "model_dump"):
        # An inference request: `image` is the list of request images.
        return describe_model_boundary_value(image, depth + 1)
    for attribute in ("xyxy", "masks", "class_id", "embeddings", "mask"):
        inner = getattr(value, attribute, None)
        if inner is not None and type(inner).__module__.startswith(("torch", "numpy")):
            kind, _ = describe_model_boundary_value(inner, depth + 1)
            return f"{type(value).__name__}[{kind}]", 1
    return type(value).__name__, 1


def _model_call_inputs(method: str, args: tuple, kwargs: dict) -> Any:
    if method == "run_tensor_native_inference":
        return kwargs.get("images", kwargs.get("image"))
    request = kwargs.get("request", args[1] if len(args) > 1 else None)
    return getattr(request, "image", request)


def install_model_call_timing() -> None:
    """Wraps the legacy provider's model-call methods once per process.

    The wrappers consult the active probe on every call, so installing is idempotent
    and a probe can be swapped per run.
    """
    global _MODEL_CALL_TIMING_INSTALLED
    if _MODEL_CALL_TIMING_INSTALLED:
        return
    from inference.core.interfaces.workflows_models_provider import (
        ModelManagerModelsProvider,
    )

    for method in MODEL_CALL_METHODS:
        original = getattr(ModelManagerModelsProvider, method)

        def make_wrapper(name: str, function: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                probe = _ACTIVE_PROBE
                if probe is None:
                    return function(self, *args, **kwargs)
                model_id = kwargs.get("model_id", args[0] if args else "")
                return probe.record_model_call(
                    name,
                    model_id,
                    getattr(self, "_model_manager", None),
                    _model_call_inputs(name, args, kwargs),
                    lambda: function(self, *args, **kwargs),
                )

            wrapper.__name__ = name
            wrapper.__wrapped__ = function  # type: ignore[attr-defined]
            return wrapper

        setattr(ModelManagerModelsProvider, method, make_wrapper(method, original))
    _MODEL_CALL_TIMING_INSTALLED = True


def install_frame_retrieval_tracking() -> None:
    """Wraps `VideoSourcesManager`'s two retrieval methods once per process.

    Mirrors `install_model_call_timing`: the wrappers consult the active probe on
    every call, so a probe can be swapped per run.
    """
    global _FRAME_RETRIEVAL_TRACKING_INSTALLED
    if _FRAME_RETRIEVAL_TRACKING_INSTALLED:
        return
    from inference.core.interfaces.camera.utils import VideoSourcesManager

    for method in RETRIEVAL_METHODS:
        original = getattr(VideoSourcesManager, method)

        def make_wrapper(function: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                probe = _ACTIVE_PROBE
                if probe is None:
                    return function(self, *args, **kwargs)
                return probe.record_retrieval(lambda: function(self, *args, **kwargs))

            return wrapper

        setattr(VideoSourcesManager, method, make_wrapper(original))
    _FRAME_RETRIEVAL_TRACKING_INSTALLED = True


def activate(probe: BoundaryProbe) -> None:
    global _ACTIVE_PROBE
    install_model_call_timing()
    install_frame_retrieval_tracking()
    _ACTIVE_PROBE = probe


def deactivate() -> None:
    global _ACTIVE_PROBE
    _ACTIVE_PROBE = None


# ---------------------------------------------------------------------------------
# Optional CUDA copy trace (diagnostic run, never a timed baseline)
# ---------------------------------------------------------------------------------


CUDA_TRACE_SAMPLE = 32
CUDA_TRACE_TOP_ISSUERS = 20
_CPU_OP_CATEGORIES = ("cpu_op", "user_annotation")
_RUNTIME_CATEGORIES = ("cuda_runtime", "cuda_driver")
_DEVICE_COPY_CATEGORIES = ("gpu_memcpy", "gpu_memset")
_GPU_ACTIVITY_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
_CPU_COPY_OPS = ("aten::to", "aten::_to_copy", "aten::copy_", "aten::item")
_HOST_DEVICE_KINDS = ("HtoD", "DtoH")


class CudaCopyTrace:
    """Bounded torch.profiler trace of device copies, run in the model-call thread.

    torch.profiler records CPU ops (and `record_function` ranges) only on the thread
    that starts it, so it is started inside the first model call after arming, by
    the thread issuing that call, and stopped by that same thread at the end of the
    first model call after `seconds`. Every device copy is linked to the CUDA
    runtime/driver call that launched it (correlation id), and that launch to its
    issuing CPU op (External id) or thread. A copy counts as inside a model call
    only when its launch happened on the profiled thread within a
    `benchmark.model_call` range: never by time overlap across threads.
    """

    def __init__(self, seconds: float):
        import torch
        from torch.profiler import ProfilerActivity, profile, record_function

        activities = [ProfilerActivity.CPU]
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            activities.append(ProfilerActivity.CUDA)
        self._profile = profile(activities=activities)
        self._record_function = record_function
        self._seconds = seconds
        self._lock = threading.Lock()
        # armed -> running -> stopped; armed -> not_started; -> failed
        self.state = "armed"
        self._cancelled = False
        self._owner: Optional[int] = None
        self._deadline = 0.0
        self.profiled_calls = 0
        self.other_thread_calls = 0
        self._other_threads: Set[int] = set()
        self.errors: List[str] = []
        self.started = threading.Event()
        self.stopped = threading.Event()

    def run_model_call(self, call: Callable[[], Any]) -> Any:
        thread = threading.get_native_id()
        with self._lock:
            if self.state == "armed":
                try:
                    self._profile.__enter__()
                except Exception as error:  # noqa: BLE001 - diagnostic only
                    self.errors.append(f"profiler start failed: {error!r}")
                    self.state = "failed"
                    self.stopped.set()
                else:
                    self.state, self._owner = "running", thread
                    self._deadline = time.perf_counter() + self._seconds
                self.started.set()
            profiling = self.state == "running" and self._owner == thread
            if self.state == "running" and not profiling:
                self.other_thread_calls += 1
                self._other_threads.add(thread)
        if not profiling:
            return call()
        try:
            with self._record_function(MODEL_CALL_RANGE_NAME):
                return call()
        finally:
            with self._lock:
                self.profiled_calls += 1
                if self._cancelled or time.perf_counter() >= self._deadline:
                    try:
                        self._profile.__exit__(None, None, None)
                        self.state = "stopped"
                    except Exception as error:  # noqa: BLE001 - diagnostic only
                        self.errors.append(f"profiler stop failed: {error!r}")
                        self.state = "failed"
                    self.stopped.set()

    def cancel(self) -> None:
        """Called by the harness: an armed trace never starts; a running one is
        stopped by its owner thread at the end of its next model call."""
        with self._lock:
            self._cancelled = True
            if self.state == "armed":
                self.state = "not_started"
                self.started.set()
                self.stopped.set()

    def summarize(self) -> dict:
        with self._lock:
            state = self.state
            meta = dict(
                owner_tid=self._owner,
                profiled_calls=self.profiled_calls,
                other_thread_calls=self.other_thread_calls,
                other_tids=set(self._other_threads),
                cuda_traced=self.cuda,
                state=state,
                errors=list(self.errors),
            )
        events: List[dict] = []
        if state == "stopped":
            # Read only after the owner thread stopped the profiler.
            handle, path = tempfile.mkstemp(suffix=".json")
            os.close(handle)
            try:
                self._profile.export_chrome_trace(path)
                with open(path) as f:
                    events = json.load(f).get("traceEvents", [])
            except Exception as error:  # noqa: BLE001 - diagnostic only
                meta["errors"].append(f"trace export failed: {error!r}")
            finally:
                os.unlink(path)
        return summarize_cuda_trace(events, **meta)


def _copy_kind(name: str) -> str:
    if "memset" in name.lower():
        return "Memset"
    for kind in ("HtoD", "DtoH", "DtoD", "HtoH", "PtoP"):
        if kind in name:
            return kind
    return "other"


def _inside(ranges: List[Tuple[float, float]], ts: float) -> bool:
    return any(start <= ts <= end for start, end in ranges)


def summarize_cuda_trace(
    events: List[dict],
    owner_tid: Optional[int],
    profiled_calls: int,
    other_thread_calls: int,
    other_tids: Set[int],
    cuda_traced: bool,
    state: str,
    errors: List[str],
) -> dict:
    """Bounded summary of a chrome trace (torch.profiler export) of one window.

    Attribution of each device copy, from launch evidence only:
    `model_call` / `outside_model_call` when its launch is linked (correlation id)
    to a thread whose model calls were all profiled; `unattributed` otherwise (no
    launch record, or launched by a model-calling thread that was not profiled).
    `status` is `complete` only when that evidence covers every copy and every model
    call in the window, and positive model-range and GPU-activity coverage exist.
    `host_transfer_coverage_status` is the same requirement restricted to copies
    that can reach host memory (everything except DtoD/Memset, which are known by
    kind alone to stay on-device): it can be `complete` while `status` stays
    `unproven` on unattributed DtoD/Memset copies alone. `model_call_host_device_copies`
    is populated once `host_transfer_coverage_status` is `complete`, not full `status`.
    Even then copies are not linked to model inputs/outputs: this never proves the
    absence of input host round-trips (`input_host_round_trip` stays unknown).
    """
    owner = str(owner_tid) if owner_tid is not None else None
    others = {str(t) for t in other_tids}
    ranges: Dict[str, List[Tuple[float, float]]] = {}
    ops_by_external: Dict[Any, dict] = {}
    launches: Dict[Any, dict] = {}
    copies: List[dict] = []
    gpu_activity = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        category, args = event.get("cat", ""), event.get("args") or {}
        if category in _GPU_ACTIVITY_CATEGORIES:
            gpu_activity += 1
        if category in _CPU_OP_CATEGORIES:
            if "External id" in args:
                ops_by_external[args["External id"]] = event
            if event.get("name") == MODEL_CALL_RANGE_NAME:
                ranges.setdefault(str(event.get("tid")), []).append(
                    (event["ts"], event["ts"] + event.get("dur", 0))
                )
        elif category in _RUNTIME_CATEGORIES and "correlation" in args:
            launches[args["correlation"]] = event
        elif category in _DEVICE_COPY_CATEGORIES:
            copies.append(event)
    # CPU ops and runtime launches share OS thread ids only if the owner's ranges
    # carry the id the owner thread reported itself; otherwise only External ids
    # link launches to threads.
    tids_verified = owner is not None and owner in ranges

    def attribute(launch: Optional[dict]) -> Tuple[str, str]:
        if launch is None:
            return "unattributed", "<no launch record>"
        op = ops_by_external.get((launch.get("args") or {}).get("External id"))
        issuer = op.get("name") if op is not None else launch.get("name", "?")
        if op is not None:
            tid = str(op.get("tid"))
        elif tids_verified:
            tid = str(launch.get("tid"))
        else:
            return "unattributed", issuer
        if tid == owner:
            inside = _inside(ranges.get(owner, []), launch["ts"])
            return ("model_call" if inside else "outside_model_call"), issuer
        # `others` only lists threads whose model call was itself profiled (started
        # after the trace was armed). A thread not in `others` may still have a model
        # call in flight when the trace armed - `record_model_call` snapshots the
        # trace once, at entry, so that call was never registered here. Without a
        # call registry there is no way to tell "no model call" from "unregistered
        # model call" apart, so this copy stays unattributed rather than asserting
        # `outside_model_call`.
        return "unattributed", issuer

    device_copies: Dict[str, Dict[str, Dict[str, int]]] = {}
    issuers: Counter = Counter()
    samples = []
    for copy in copies:
        args = copy.get("args") or {}
        kind = _copy_kind(copy.get("name", ""))
        attribution, issuer = attribute(launches.get(args.get("correlation")))
        size = int(args.get("bytes") or 0)
        entry = device_copies.setdefault(kind, {}).setdefault(
            attribution, {"count": 0, "bytes": 0}
        )
        entry["count"] += 1
        entry["bytes"] += size
        issuers[f"{kind} {attribution} <- {issuer}"] += 1
        if len(samples) < CUDA_TRACE_SAMPLE:
            samples.append([kind, size, attribution, issuer])
    cpu_copy_ops: Dict[str, Dict[str, int]] = {}
    if owner is not None:
        for op in ops_by_external.values():
            if op.get("name") in _CPU_COPY_OPS and str(op.get("tid")) == owner:
                inside = _inside(ranges.get(owner, []), op["ts"])
                entry = cpu_copy_ops.setdefault(
                    op["name"], {"inside_model_call": 0, "outside_model_call": 0}
                )
                entry["inside_model_call" if inside else "outside_model_call"] += 1
    model_call_ranges = sum(len(r) for r in ranges.values())
    # DtoD/Memset never touch host memory, so a missing launch record for one of
    # them cannot hide a host transfer; every other kind (including "other", whose
    # direction is itself unknown) can, so it still blocks host-transfer coverage.
    non_host_kinds = {"DtoD", "Memset"}
    unattributed = sum(
        by_attribution.get("unattributed", {}).get("count", 0)
        for by_attribution in device_copies.values()
    )
    host_unattributed = sum(
        by_attribution.get("unattributed", {}).get("count", 0)
        for kind, by_attribution in device_copies.items()
        if kind not in non_host_kinds
    )
    base_reasons = list(errors)
    if state != "stopped":
        base_reasons.append(f"trace {state}: no profiled window")
    if not cuda_traced:
        base_reasons.append("CUDA activity not traced (CPU ops only)")
    if not model_call_ranges:
        base_reasons.append("no model-call range recorded in the profiled thread")
    elif model_call_ranges != profiled_calls:
        base_reasons.append(
            f"{model_call_ranges} model-call ranges for {profiled_calls} profiled calls"
        )
    if not gpu_activity:
        base_reasons.append("no GPU activity recorded")
    if other_thread_calls:
        base_reasons.append(
            f"{other_thread_calls} model calls on {len(others)} unprofiled threads"
        )
    if owner is not None and not tids_verified:
        base_reasons.append(
            "launch thread IDs not verified: owner thread range coverage missing"
        )
    reasons = list(base_reasons)
    if unattributed:
        reasons.append(f"{unattributed} device copies without launch attribution")
    complete = not reasons
    host_reasons = list(base_reasons)
    if host_unattributed:
        host_reasons.append(
            f"{host_unattributed} host-capable (non DtoD/Memset) device copies "
            "without launch attribution"
        )
    host_transfer_complete = not host_reasons
    model_call_host_device = [
        device_copies.get(kind, {}).get("model_call", {"count": 0, "bytes": 0})
        for kind in _HOST_DEVICE_KINDS
    ]
    return {
        "status": "complete" if complete else "unproven",
        "reasons": reasons,
        # Independent, narrower claim: every HtoD/DtoH (and any unknown-direction)
        # copy is attributed, even if some known-safe DtoD/Memset copies are not.
        # This is what backs `model_call_host_device_copies` below; it does not
        # imply full device-copy attribution (`status` above) and never proves
        # `input_host_round_trip`.
        "host_transfer_coverage_status": (
            "complete" if host_transfer_complete else "unproven"
        ),
        "host_transfer_coverage_reasons": host_reasons,
        "trace_state": state,
        "cuda_activity_traced": cuda_traced,
        "model_calls_profiled": profiled_calls,
        "model_call_ranges": model_call_ranges,
        "model_calls_other_threads": other_thread_calls,
        "gpu_activity_events": gpu_activity,
        "events_traced": len(events),
        "launch_thread_ids_verified": tids_verified,
        # {kind: {model_call|outside_model_call|unattributed: {count, bytes}}}
        "device_copies": device_copies,
        # Copies launched inside model calls; comparable between runs only when
        # both have complete host-transfer coverage. Not linked to model
        # inputs/outputs.
        "model_call_host_device_copies": (
            {
                "count": sum(e["count"] for e in model_call_host_device),
                "bytes": sum(e["bytes"] for e in model_call_host_device),
                "per_model_call": sum(e["count"] for e in model_call_host_device)
                / profiled_calls,
            }
            if host_transfer_complete
            else None
        ),
        "copy_issuers": dict(issuers.most_common(CUDA_TRACE_TOP_ISSUERS)),
        "cpu_copy_ops": cpu_copy_ops,
        # [kind, bytes, attribution, issuing op or launch call]
        "copy_samples": samples,
        "input_host_round_trip": "unknown",
        "note": (
            "Diagnostic only. Copies are attributed by launch correlation and the "
            "launching thread, not by time overlap. Neither this trace nor the "
            "model input/output kinds link a copy to a model input or output, so "
            "it never proves that inputs made no host round-trip; `unproven` "
            "means even the attribution is incomplete."
        ),
    }


def _sanitize_nested(value: Any) -> Any:
    """Recursively redacts credentials from every string (values and keys) in a
    saved structure, e.g. an error message echoing a source reference."""
    if isinstance(value, str):
        return redact_credentials_in_text(value)
    if isinstance(value, dict):
        return {
            (redact_credentials_in_text(k) if isinstance(k, str) else k): (
                _sanitize_nested(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_nested(v) for v in value]
    return value


# ---------------------------------------------------------------------------------
# Manager child process: hooks installed inside run(), so they survive spawn
# ---------------------------------------------------------------------------------


def _install_pipeline_hooks(probe: BoundaryProbe) -> None:
    # The class the manager actually builds pipelines with: the legacy wrapper
    # at the baseline, the host-neutral pipeline since WP-A03.
    from inference.core.interfaces.stream_manager.manager_app.inference_pipeline_manager import (
        InferencePipeline,
    )

    original = InferencePipeline.init_with_workflow

    def instrumented_init_with_workflow(cls: Any, *args: Any, **kwargs: Any) -> Any:
        handlers = list(kwargs.get("status_update_handlers") or [])
        kwargs["status_update_handlers"] = [probe.on_status_update] + handlers
        if kwargs.get("on_prediction") is not None:
            kwargs["on_prediction"] = probe.wrap_sink(kwargs["on_prediction"])
        pipeline = original(*args, **kwargs)
        # Decoders are sampled later (after `start()`/drain): at this point
        # `pipeline.start()` has not run yet, so every `VideoSource._video` is
        # still `None`.
        probe.note_pipeline_reference(pipeline)
        return pipeline

    InferencePipeline.init_with_workflow = classmethod(instrumented_init_with_workflow)


def _maybe_init_cuda_on_child_main_thread() -> str:
    """Best-effort CUDA primary-context init on this pipeline child's main thread.

    Benchmark-only mitigation for NVML mis-attributing the pipeline's GPU memory
    on some drivers; see `results/README.md`, "GPU memory, manager transport" for
    the experimental detail. `torch.cuda.init()` only creates the primary
    context -- it allocates no model tensors, but the context itself does use
    GPU resources. Runs before `super().run()` can start any worker thread.
    """
    if not sys.platform.startswith("linux"):
        return "skipped:not-linux"
    try:
        import torch

        if not torch.cuda.is_available():
            return "skipped:cuda-unavailable"
        torch.cuda.init()
    except Exception as error:  # noqa: BLE001 - benchmark instrumentation only
        return f"error:{error!r}"
    return "initialized"


class InstrumentedInferencePipelineManager(InferencePipelineManager):
    """The production pipeline process plus the probe; commands are unchanged.

    Defined at module level so multiprocessing can pickle it by reference for the
    spawned child, which then imports this module and runs the hooks itself.
    """

    def run(self) -> None:
        probe = BoundaryProbe()
        self._benchmark_probe = probe
        # Before anything else in this child: see `_maybe_init_cuda_on_child_main_thread`.
        probe.note_cuda_child_main_thread_init(_maybe_init_cuda_on_child_main_thread())
        activate(probe)
        _install_pipeline_hooks(probe)
        try:
            super().run()
        finally:
            self._write_probe(probe)

    def _execute_termination(self) -> None:
        drained = False
        try:
            super()._execute_termination()  # terminate() + join()
            drained = True
        finally:
            self._benchmark_probe.note_pending(self._buffer_sink)
            self._benchmark_probe.note_drained(drained)
            # Sources have started (or, on the drain path, already run and been
            # released without clearing `_video`) by now, so the retained pipeline
            # reference reports the actual producer class instead of `NoneType`.
            self._benchmark_probe.note_decoders()

    def _write_probe(self, probe: BoundaryProbe) -> None:
        directory = os.environ.get(PROBE_DIR_ENV)
        if not directory:
            return
        path = os.path.join(directory, f"{self._pipeline_id}.json")
        try:
            # Sanitize before writing: this scratch file can outlive a crashed run
            # and is read back into the parent result verbatim.
            with open(path + ".tmp", "w") as f:
                json.dump(_sanitize_nested(probe.export()), f)
            os.replace(path + ".tmp", path)
        except OSError:
            pass


def main() -> None:
    from inference.core.interfaces.stream_manager.manager_app import app

    app.InferencePipelineManager = InstrumentedInferencePipelineManager
    app.start()


if __name__ == "__main__":
    main()
