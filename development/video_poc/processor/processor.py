"""Video POC processor: a warm worker that runs Roboflow Workflows on video sources.

Modes:
  --job-file job.json          run a single job from disk (standalone testing, no platform)
  --api-url ... --api-key ...  poll the platform for job assignments (videoJobs/claim contract)

While a job runs, the processor exposes:
  GET /status                worker + job state, timing/fps stats, and the image outputs
                             it is redacting from events (imageOutputs) so clients can
                             attach to any of them after the job has started
  GET /events                SSE stream of per-frame workflow outputs (images redacted to refs)
  GET /preview.mjpeg?output= MJPEG stream of one image output (defaults to the job's
                             designated output, switchable per-request)
  GET /results/<jobId>/...   persisted batch-job results: video.mp4 (annotated, Range
                             support for scrubbing), frames.jsonl (one line per frame,
                             aligned with the mp4 by index), meta.json (fps, frames);
                             local fallback — results also upload to GCS on completion
  GET /metrics               Prometheus text: video_processor_busy gauge (warm-pool
                             autoscaling signal: replicas = sum(busy) + MIN_IDLE)

Design notes (mirrors the video strategy deck):
  - events and pixels are split at the source; base64 image blobs never ride the events channel
  - the pipeline uses InferencePipeline's live-stream defaults (ADAPTIVE_DROP_OLDEST) so
    latency stays bounded at ~one inference time regardless of ingest rate
"""

import argparse
import base64
import copy
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import requests

import cv2

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.workflows.core_steps.common.serializers import (
    serialize_wildcard_kind,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

POLL_INTERVAL_S = 2.0
EVENT_BUFFER_SIZE = 50
MJPEG_MAX_FPS = 12

# Used by stream-mode jobs on file sources: the file is replayed at native speed
# into the local relay and consumed as RTSP, so the pipeline sees a real stream.
RTSP_SIM_BASE = os.getenv("VIDEO_PROC_RTSP_BASE", "rtsp://127.0.0.1:8554")
FFMPEG_BIN = os.getenv(
    "FFMPEG_BIN",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "ffmpeg"),
)
if not os.path.exists(FFMPEG_BIN):
    FFMPEG_BIN = "ffmpeg"

# Batch-job results (annotated mp4 + per-frame JSONL) are kept here so the UI can
# scrub them after the job completes. Survives until the OS clears temp storage;
# the production shape is object storage.
RESULTS_ROOT = os.getenv(
    "VIDEO_PROC_RESULTS_DIR", os.path.join(tempfile.gettempdir(), "rf-video-poc-results")
)
RESULT_FILES = {
    "video.mp4": "video/mp4",
    "frames.jsonl": "application/x-ndjson",
    "meta.json": "application/json",
}
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def utcnow_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with getattr(self, "lock", threading.Lock()):
            self.job_received_at = None
            self.pipeline_started_at = None
            self.first_result_at = None
            self.frames = 0
            self.ema_fps = None
            self.last_latency_ms = None
            self.ema_latency_ms = None
            self._last_frame_time = None

    def on_job(self):
        self.reset()
        self.job_received_at = time.time()

    def on_pipeline_start(self):
        self.pipeline_started_at = time.time()

    def on_result(self, video_frame: VideoFrame):
        now = time.time()
        with self.lock:
            if self.first_result_at is None:
                self.first_result_at = now
            self.frames += 1
            if self._last_frame_time is not None:
                dt = now - self._last_frame_time
                if dt > 0:
                    inst = 1.0 / dt
                    self.ema_fps = inst if self.ema_fps is None else 0.9 * self.ema_fps + 0.1 * inst
            self._last_frame_time = now
            latency_ms = (datetime.now() - video_frame.frame_timestamp).total_seconds() * 1000.0
            self.last_latency_ms = latency_ms
            self.ema_latency_ms = (
                latency_ms if self.ema_latency_ms is None else 0.9 * self.ema_latency_ms + 0.1 * latency_ms
            )

    def snapshot(self) -> dict:
        with self.lock:
            out = {
                "frames": self.frames,
                "fps": round(self.ema_fps, 2) if self.ema_fps else None,
                "decodeToResultLatencyMs": round(self.ema_latency_ms, 1) if self.ema_latency_ms else None,
            }
            if self.job_received_at and self.pipeline_started_at:
                out["pipelineStartS"] = round(self.pipeline_started_at - self.job_received_at, 2)
            if self.job_received_at and self.first_result_at:
                out["timeToFirstResultS"] = round(self.first_result_at - self.job_received_at, 2)
            return out


class EventBus:
    """Fan-out for SSE subscribers with a small replay buffer."""

    def __init__(self):
        self.lock = threading.Lock()
        self.buffer = []
        self.subscribers = []

    def publish(self, event: dict):
        data = json.dumps(event, default=str)
        with self.lock:
            self.buffer.append(data)
            if len(self.buffer) > EVENT_BUFFER_SIZE:
                self.buffer.pop(0)
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(data)
            except Exception:
                pass

    def subscribe(self):
        import queue

        q = queue.Queue(maxsize=200)
        with self.lock:
            for item in self.buffer[-10:]:
                q.put_nowait(item)
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


class FrameStore:
    """Latest annotated JPEG per image output, for MJPEG preview.

    Keeping the latest frame for every image output (not just the designated one)
    is what makes late attachment possible: a viewer can pick any output after the
    job started. The frames are already JPEG-encoded by the pipeline's serializer,
    so the only extra cost per output is a base64 decode.
    """

    def __init__(self):
        self.cond = threading.Condition()
        self.jpegs = {}
        self.seqs = {}

    def set(self, output: str, jpeg_bytes: bytes):
        with self.cond:
            self.jpegs[output] = jpeg_bytes
            self.seqs[output] = self.seqs.get(output, 0) + 1
            self.cond.notify_all()

    def wait_next(self, output: str, last_seq, timeout=2.0):
        with self.cond:
            if self.seqs.get(output, 0) == last_seq:
                self.cond.wait(timeout=timeout)
            return self.jpegs.get(output), self.seqs.get(output, 0)

    def latest(self, output: str):
        with self.cond:
            return self.jpegs.get(output)

    def outputs(self):
        with self.cond:
            return sorted(self.jpegs)

    def clear(self):
        with self.cond:
            self.jpegs = {}
            self.seqs = {}
            self.cond.notify_all()


class JobRecorder:
    """Persists a batch job's results for post-hoc review with scrubbing.

    Writes the designated image output as an H.264 mp4 (via ffmpeg image2pipe, at
    the source's declared fps) and one JSON line per processed frame. In batch mode
    every frame is processed in order, so mp4 frame k, JSONL line k, and playhead
    time k/fps all refer to the same source frame — that's what lets the UI align
    the JSON with the video while scrubbing.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.dir = os.path.join(RESULTS_ROOT, job_id)
        os.makedirs(self.dir, exist_ok=True)
        self.events_file = open(os.path.join(self.dir, "frames.jsonl"), "w")
        self.ffmpeg = None
        self.fps = None
        self.frames = 0
        self.video_frames = 0
        self.video_failed = False
        self.lock = threading.Lock()

    def add(self, jpeg_bytes, fps, event: dict):
        with self.lock:
            if self.events_file is None:
                return
            # image outputs can be conditional (e.g. visualization only when
            # detections exist), so the mp4 can have fewer frames than the JSONL.
            # Each event records which encoded video frame it belongs to (or
            # null) so the review UI aligns the playhead correctly either way.
            has_video_frame = jpeg_bytes is not None and not self.video_failed
            event["videoFrame"] = self.video_frames if has_video_frame else None
            self.events_file.write(json.dumps(event, default=str) + "\n")
            self.frames += 1
            if not has_video_frame:
                return
            if self.ffmpeg is None:
                self.fps = float(fps) if fps and fps > 0 else 30.0
                try:
                    self.ffmpeg = subprocess.Popen(
                        [
                            FFMPEG_BIN, "-y", "-f", "image2pipe",
                            "-framerate", str(self.fps), "-i", "-",
                            "-c:v", "libx264", "-preset", "veryfast",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            os.path.join(self.dir, "video.mp4"),
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError as exc:
                    print(f"[processor] result recording disabled: {exc}", file=sys.stderr)
                    self.video_failed = True
                    return
            try:
                self.ffmpeg.stdin.write(jpeg_bytes)
                self.video_frames += 1
            except Exception:
                self.video_failed = True

    def finalize(self):
        with self.lock:
            events_file, self.events_file = self.events_file, None
            if events_file is None:
                return
            events_file.close()
            if self.ffmpeg is not None:
                try:
                    self.ffmpeg.stdin.close()
                    self.ffmpeg.wait(timeout=60)
                except Exception:
                    self.video_failed = True
                    # don't leak a wedged encoder (it also holds video.mp4 open)
                    try:
                        self.ffmpeg.kill()
                        self.ffmpeg.wait(timeout=5)
                    except Exception:
                        pass
            meta = {
                "jobId": self.job_id,
                "fps": self.fps,
                "frames": self.frames,
                "videoFrames": self.video_frames,
                "hasVideo": self.ffmpeg is not None and not self.video_failed,
            }
            with open(os.path.join(self.dir, "meta.json"), "w") as f:
                json.dump(meta, f)


# Which transport publishes the annotated output to the relay:
#   rtsp — ffmpeg subprocess (proven, encode fully isolated from the pipeline)
#   whip — in-process aiortc/WebRTC (real per-frame timestamps, PLI-driven
#          keyframes, no RTSP layer — lower latency; watchdog-guarded)
# Per-job override: claim payload's publisherTransport. The watchdog can
# hot-swap whip→rtsp mid-job if inference latency degrades.
PUBLISHER_TRANSPORT = os.getenv("VIDEO_PROC_PUBLISHER", "rtsp").strip().lower()
WHIP_SIM_BASE = os.getenv("VIDEO_PROC_WHIP_BASE", "http://127.0.0.1:8889")


class OutputPublisher:
    """Publishes one annotated image output to the relay as H.264 RTSP — but
    ONLY while the platform says someone is watching (the watch TTL rides the
    status-poll response). One publish serves every WHEP viewer through the
    relay, and nothing is encoded while nobody watches.

    Latency-shaped: frames are pushed the moment the pipeline produces them
    (event-driven, native fps — no resampling tick), ffmpeg stamps them with
    wallclock timestamps and encodes VFR, and a keyframe is forced every second
    so WHEP viewers join fast. zerolatency/ultrafast keeps encoder buffering
    at zero frames.
    """

    transport = "rtsp"

    def __init__(self, frames: FrameStore, publish_url: str):
        self.frames = frames
        self.publish_url = publish_url
        self.output = None
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def running(self):
        return self._proc is not None and self._proc.poll() is None

    def ensure(self, output: str):
        """Publish `output`, (re)starting if the output changed or ffmpeg died."""
        if self.running() and output == self.output:
            return
        self.stop()
        self.output = output
        self._stop.clear()
        try:
            proc = subprocess.Popen(
                [
                    FFMPEG_BIN,
                    # PTS from arrival time: frames are written as the pipeline
                    # produces them, so wallclock IS the correct timestamp and
                    # the stream carries no resampling delay
                    "-use_wallclock_as_timestamps", "1",
                    "-f", "image2pipe", "-i", "-",
                    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                    "-profile:v", "baseline", "-pix_fmt", "yuv420p",
                    # VFR passthrough (deprecated alias of -fps_mode vfr; kept
                    # for the ffmpeg 4.4 in the ubuntu-based processor image)
                    "-vsync", "vfr",
                    # keyframe every second of stream time regardless of fps:
                    # WHEP viewers can't render until an IDR arrives, so this
                    # bounds join latency at ~1s (was up to 2s)
                    "-force_key_frames", "expr:gte(t,n_forced)",
                    "-f", "rtsp", "-rtsp_transport", "tcp", self.publish_url,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            print(f"[processor] output publisher failed to start: {exc}", file=sys.stderr)
            return
        self._proc = proc
        self._thread = threading.Thread(target=self._pump, args=(proc, output), daemon=True)
        self._thread.start()
        print(f"[processor] viewer attached — publishing '{output}' to the relay")

    def _pump(self, proc, output):
        # event-driven: block until the pipeline produces a frame, ship it
        # immediately. No tick, no duplicate frames, no resampling staleness.
        seq = -1
        while not self._stop.is_set() and proc.poll() is None:
            jpeg, seq = self.frames.wait_next(output, seq, timeout=0.5)
            if jpeg is None:
                continue
            try:
                proc.stdin.write(jpeg)
                proc.stdin.flush()
            except Exception:
                break

    def stop(self):
        self._stop.set()
        thread, self._thread = self._thread, None
        proc, self._proc = self._proc, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        if proc is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            print("[processor] no viewers — stopped publishing to the relay")


class AiortcWhipPublisher:
    """OutputPublisher-compatible transport that encodes in-process (aiortc/PyAV)
    and publishes to the relay over WHIP (WebRTC ingest).

    Why it exists: real per-frame RTP timestamps (frames are stamped with their
    actual arrival time, no CFR faking), keyframes on viewer demand (PLI from
    the relay) instead of a fixed GOP, and no RTSP layer — measurably lower
    glass-to-glass latency than the ffmpeg/RTSP transport.

    Feeds from the worker's raw-frame store (ndarrays straight from the
    workflow — the sink owns serialization, so no JPEG ever exists on this
    path and encode input is generation-loss-free).

    Risk containment lives OUTSIDE this class: the worker's watchdog hot-swaps
    back to the ffmpeg/RTSP transport mid-job if inference latency degrades
    while this publisher runs (in-process encode shares CPU/GIL with the
    pipeline; the subprocess transport does not).
    """

    transport = "whip"

    def __init__(self, frames: FrameStore, whip_url: str):
        self.frames = frames
        self.whip_url = whip_url
        self.output = None
        self._thread = None
        self._stopped = threading.Event()
        self._failed = False

    def running(self):
        return self._thread is not None and self._thread.is_alive() and not self._failed

    def ensure(self, output: str):
        if self.running() and output == self.output:
            return
        self.stop()
        self.output = output
        self._stopped.clear()
        self._failed = False
        self._thread = threading.Thread(target=self._run, args=(output,), daemon=True)
        self._thread.start()

    def _run(self, output):
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._publish(loop, output))
        except Exception as exc:
            print(f"[processor] whip publisher error: {exc}", file=sys.stderr)
            self._failed = True
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    async def _publish(self, loop, output):
        import asyncio
        import fractions

        from aiortc import RTCPeerConnection, RTCSessionDescription
        from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
        from av import VideoFrame

        frames = self.frames  # raw ndarrays per output (no JPEG round trip)
        stopped = self._stopped

        class LatestFrameTrack(MediaStreamTrack):
            kind = "video"

            def __init__(self):
                super().__init__()
                self._seq = -1
                self._t0 = None

            async def recv(self):
                # event-driven: block (off the event loop) until the pipeline
                # produces a new frame, then stamp its actual arrival time
                while True:
                    if stopped.is_set():
                        raise MediaStreamError
                    image, seq = await loop.run_in_executor(
                        None, frames.wait_next, output, self._seq, 0.25
                    )
                    if image is not None and seq != self._seq:
                        self._seq = seq
                        break
                # yuv420 conversion requires even dimensions
                h, w = image.shape[:2]
                if h % 2 or w % 2:
                    image = image[: h - (h % 2), : w - (w % 2)]
                frame = VideoFrame.from_ndarray(image, format="bgr24")
                now = time.monotonic()
                if self._t0 is None:
                    self._t0 = now
                frame.pts = int((now - self._t0) * 90000)
                frame.time_base = fractions.Fraction(1, 90000)
                return frame

        pc = RTCPeerConnection()
        pc.addTrack(LatestFrameTrack())
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)

        # WHIP is just "POST the offer SDP, apply the answer"
        resp = await loop.run_in_executor(
            None,
            lambda: requests.post(
                self.whip_url,
                data=pc.localDescription.sdp,
                headers={"Content-Type": "application/sdp"},
                timeout=10,
            ),
        )
        resp.raise_for_status()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=resp.text, type="answer"))
        print(f"[processor] viewer attached — publishing '{output}' via WHIP (in-process)")

        try:
            while not stopped.is_set() and pc.connectionState not in ("failed", "closed"):
                await asyncio.sleep(0.5)
        finally:
            await pc.close()
            print("[processor] whip publisher stopped")

    def stop(self):
        self._stopped.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)


K8S_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"


class PodSelf:
    """Self-management for ready-pool mode: the worker detaches its own pod from
    the ready pool on claim (label change — the ReplicaSet instantly creates a
    replacement, which IS the pool refill) and deletes its own pod when the job
    ends (workers are single-use; the only pods that ever terminate chose to).
    Raw API calls with the mounted service-account credentials — no client lib.
    Inactive outside a cluster, so local dev behaves exactly as before.
    """

    def __init__(self):
        self.pod_name = os.getenv("POD_NAME")
        pool_mode = os.getenv("PROCESSOR_POOL_MODE", "").strip().lower()
        self.enabled = (
            pool_mode in ("1", "true", "yes", "on")
            and bool(self.pod_name)
            and os.path.isfile(os.path.join(K8S_SA_DIR, "token"))
        )
        if not self.enabled:
            return
        with open(os.path.join(K8S_SA_DIR, "namespace")) as f:
            namespace = f.read().strip()
        self._ca = os.path.join(K8S_SA_DIR, "ca.crt")
        self._pod_url = (
            f"https://kubernetes.default.svc/api/v1/namespaces/{namespace}/pods/{self.pod_name}"
        )

    def _auth(self):
        # bound service-account tokens rotate on disk and expire (~1h): a worker
        # that idled in the pool for hours must not present a startup-time token
        with open(os.path.join(K8S_SA_DIR, "token")) as f:
            return {"Authorization": f"Bearer {f.read().strip()}"}

    def detach_from_pool(self, job_id):
        if not self.enabled:
            return
        try:
            resp = requests.patch(
                self._pod_url,
                json={
                    "metadata": {
                        "labels": {"pool": "working"},
                        "annotations": {"roboflow.com/job-id": str(job_id)},
                    }
                },
                headers={
                    **self._auth(),
                    "Content-Type": "application/strategic-merge-patch+json",
                },
                verify=self._ca,
                timeout=10,
            )
            resp.raise_for_status()
            print(f"[processor] detached from ready pool (job {job_id}); pool will refill")
        except Exception as exc:
            print(f"[processor] pool detach failed (continuing): {exc}", file=sys.stderr)

    def is_detached(self):
        """True when this pod is already labeled pool=working — meaning this
        process is a crash-restart inside a spent pod (kubelet restarts the
        container in place), not a fresh pool member."""
        if not self.enabled:
            return False
        try:
            resp = requests.get(self._pod_url, headers=self._auth(), verify=self._ca, timeout=10)
            resp.raise_for_status()
            labels = resp.json().get("metadata", {}).get("labels", {})
            return labels.get("pool") == "working"
        except Exception:
            return False

    def self_delete(self):
        if not self.enabled:
            return False
        try:
            resp = requests.delete(
                self._pod_url,
                headers=self._auth(),
                verify=self._ca,
                timeout=10,
            )
            resp.raise_for_status()
            print("[processor] job over — retiring this pod")
            return True
        except Exception as exc:
            print(f"[processor] self-delete failed; staying alive: {exc}", file=sys.stderr)
            return False


class Worker:
    def __init__(self, args):
        self.args = args
        self.processor_id = args.processor_id or f"proc-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self.pod = PodSelf()
        self.retiring = False
        # What browsers should use to reach this worker. Priority: explicit URL,
        # then gateway base + this pod's IP (ready-pool mode: pod names are
        # random, so the gateway routes /ip-a-b-c-d/ segments), then localhost.
        gateway_base = os.getenv("GATEWAY_PUBLIC_BASE")
        pod_ip = os.getenv("POD_IP")
        if args.public_url:
            self.public_url = args.public_url
        elif gateway_base and pod_ip:
            self.public_url = f"{gateway_base.rstrip('/')}/ip-{pod_ip.replace('.', '-')}"
        else:
            # "localhost", NOT 127.0.0.1: browsers on an https app page allow
            # insecure subresources from hostname loopbacks but refuse to load
            # <img> streams from IP-literal hosts ("not upgraded to HTTPS
            # because its URL's host is an IP address" → MJPEG never paints)
            self.public_url = f"http://localhost:{args.port}"
        # serializes claims between the poll loop and the Pub/Sub wake-up so a
        # worker can never start two jobs
        self.claim_lock = threading.Lock()
        # run_job holds this end to end; stop_job takes it — so cancel can never
        # interleave with a still-starting job (which would leak a running
        # pipeline with no job attached and wedge the poll loop)
        self.lifecycle_lock = threading.Lock()
        # pool mode is single-use: once this pod has taken a job it never claims
        # again, closing every claim-after-detach race in one place
        self.had_job = False
        # set while stop_job tears a job down, so the pipeline-end watcher knows
        # not to upload partial results or report a cancelled job as completed
        self.cancelling = False
        self._pubsub_client = None
        self.stats = Stats()
        self.events = EventBus()
        self.frames = FrameStore()
        # raw ndarrays per image output (same store semantics): feeds the
        # in-process publisher without any JPEG round trip
        self.raw_frames = FrameStore()
        self.pipeline = None
        self.sim_process = None
        self.recorder = None
        self.publisher = None
        # watchdog state for the in-process (whip) publisher transport
        self._publisher_degraded = False
        self._publish_baseline_ms = None
        self._latency_strikes = 0
        self.job = None
        self.state = "idle"
        self.image_output = None
        self._stop_requested = False
        self.lock = threading.Lock()

    # ---------- sink ----------

    def on_prediction(self, predictions, video_frame):
        if isinstance(predictions, list):  # multi-source mode; POC runs one source
            predictions, video_frame = predictions[0], video_frame[0]
        if predictions is None or video_frame is None:
            return
        self.stats.on_result(video_frame)

        # We own serialization (serialize_results=False): image outputs stay raw
        # ndarrays (straight into the publisher's raw store — no JPEG round
        # trip, no generation loss) and everything else goes through inference's
        # own serialize_wildcard_kind, so event content is identical to what
        # the pipeline's serializer produced.
        outputs = {}
        designated_jpeg = None
        for key, value in predictions.items():
            if isinstance(value, WorkflowImageData):
                if self.image_output is None:
                    self.image_output = key
                try:
                    image = value.numpy_image
                    self.raw_frames.set(key, image)
                    ok, buf = cv2.imencode(".jpg", image)
                    if ok:
                        jpeg = buf.tobytes()
                        self.frames.set(key, jpeg)
                        if key == self.image_output:
                            designated_jpeg = jpeg
                except Exception:
                    pass
                outputs[key] = {"type": "image_ref", "output": key}
            elif isinstance(value, list) and value and all(isinstance(v, WorkflowImageData) for v in value):
                outputs[key] = [{"type": "image_ref", "output": key, "index": i} for i in range(len(value))]
            else:
                try:
                    outputs[key] = serialize_wildcard_kind(value=value)
                except Exception:
                    outputs[key] = str(value)

        event = {
            "frameId": video_frame.frame_id,
            "timestamp": utcnow_iso(),
            "latencyMs": self.stats.last_latency_ms and round(self.stats.last_latency_ms, 1),
            "outputs": outputs,
        }
        recorder = self.recorder
        if recorder is not None:
            recorder.add(designated_jpeg, video_frame.fps, event)
        self.events.publish(event)

    # ---------- job lifecycle ----------

    def run_job(self, job: dict):
        with self.lifecycle_lock:
            self._run_job_locked(job)

    def _run_job_locked(self, job: dict):
        with self.lock:
            self.job = job
            self.state = "starting"
            self.image_output = job.get("imageOutput")
        self.frames.clear()
        self.raw_frames.clear()
        self.stats.on_job()

        source_url = job["sourceUrl"]
        mode = job.get("mode") or ("batch" if source_url.startswith("http") else "stream")
        print(f"[processor] starting job {job.get('id')} ({mode}) on {source_url}")

        video_reference = source_url
        pipeline_kwargs = {}
        # OpenCV FFmpeg capture options are read when the capture opens (inside
        # init_with_workflow); one job per worker makes per-mode env safe. Both
        # knobs are overridable per job (claim payload: captureOptions,
        # captureBufferSize) — use cases differ: monitoring wants freshness,
        # some sources may need multi-threaded decode to keep up (4K), and a
        # non-sliced source stream may prefer thread_type tuning over threads;1.
        if mode == "stream":
            # threads;1 is the big one: H.264 frame-threaded decoding holds
            # (threads-1) frames INSIDE the decoder — a constant ~15-frame
            # (~500ms at 30fps) standing delay ahead of frame_timestamp,
            # invisible to every latency stat we print. The connector encodes
            # zerolatency (sliced) streams, so single-thread 720p30 decode is
            # cheap. nobuffer/low_delay trim demuxer-side buffering.
            default_capture_options = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|threads;1"
            )
            # cap the capture-side frame queue at 1: latest-frame semantics
            # must start at the decoder (CAP_PROP_BUFFERSIZE; harmlessly
            # ignored by backends that don't support it)
            default_buffersize = 1
        else:
            # batch wants decode throughput, not glass-to-glass latency
            default_capture_options = None
            default_buffersize = None
        capture_options = job.get("captureOptions", default_capture_options)
        if capture_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = capture_options
        else:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        buffersize = job.get("captureBufferSize", default_buffersize)
        if buffersize:
            pipeline_kwargs["video_source_properties"] = {"buffersize": float(buffersize)}
        if mode == "batch":
            # Record results so the UI can scrub them after the job completes.
            self.recorder = JobRecorder(str(job.get("id", "local")))
            # Batch means the WHOLE file, once: download it to a local path first.
            # A URL fails VideoSource's is_file check (os.path.exists), which has
            # two effects — stream buffer strategies (frames silently dropped) and,
            # worse, stream reconnection: at EOF the pipeline "reconnects" to the
            # URL and starts over, looping forever and never completing the job.
            # A local file gets true file semantics: every frame, natural end.
            if source_url.startswith("http"):
                try:
                    video_reference = self._download_source(source_url, job.get("id", "local"))
                except Exception as exc:
                    print(f"[processor] source download failed: {exc}", file=sys.stderr)
                    self._finalize_recorder()
                    self._cleanup_download()
                    with self.lock:
                        self.state = "error"
                        self.job = {**job, "error": f"source download failed: {exc}"}
                    # transient or not, same recovery as the sibling failure
                    # paths: retire, let the reaper requeue (attempts-capped)
                    self.retire_if_pool_mode()
                    return
            # Explicit strategies as a belt-and-braces: every frame, in order.
            pipeline_kwargs = {
                "source_buffer_filling_strategy": BufferFillingStrategy.WAIT,
                "source_buffer_consumption_strategy": BufferConsumptionStrategy.LAZY,
            }
        elif source_url.startswith("http"):
            # Stream mode on a file: replay it at native speed through the local
            # relay and consume RTSP, so the pipeline sees a genuine live stream
            # (real-time pacing, drops under load) — a recording standing in for
            # a camera that will be hooked up later. The platform hands us a
            # credentialed publish URL; the env base is the local-dev fallback.
            video_reference = job.get("simPublishUrl") or f"{RTSP_SIM_BASE}/sim-{job.get('id', 'local')}"
            try:
                # -stream_loop -1: a test source should keep behaving like a camera
                # until the job is stopped, not end when the recording runs out
                self.sim_process = subprocess.Popen(
                    [
                        FFMPEG_BIN, "-re", "-stream_loop", "-1", "-i", source_url,
                        "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp",
                        video_reference,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(2.0)  # let the publisher register with the relay
            except OSError as exc:
                print(f"[processor] could not start ffmpeg replay: {exc}", file=sys.stderr)
                with self.lock:
                    self.state = "error"
                    self.job = {**job, "error": f"ffmpeg replay failed: {exc}"}
                # deliberately NOT reported as terminal: failures here are often
                # transient (relay stream not up yet) — retiring lets the reaper
                # requeue the job, and the attempts cap handles poison jobs
                self.retire_if_pool_mode()
                return

        try:
            pipeline = InferencePipeline.init_with_workflow(
                video_reference=video_reference,
                workflow_specification=job.get("workflowSpecification"),
                workspace_name=job.get("workspaceName"),
                workflow_id=job.get("workflowId"),
                # the job carries its workspace's key so model access and usage
                # follow the job, not this worker's identity key
                api_key=job.get("apiKey") or self.args.api_key or os.getenv("ROBOFLOW_API_KEY"),
                on_prediction=self.on_prediction,
                # we serialize ourselves in on_prediction (images stay raw for
                # the publisher; the rest goes through inference's serializer)
                serialize_results=False,
                max_fps=job.get("maxFps"),
                **pipeline_kwargs,
            )
            self.pipeline = pipeline
            self.stats.on_pipeline_start()
            pipeline.start(use_main_thread=False)
            with self.lock:
                self.state = "running"
            # watch BOTH modes: a batch pipeline ending means the file is done
            # (finalize + upload + completed); a stream pipeline ending on its
            # own means the stream died and the job must be re-placed
            threading.Thread(
                target=self._watch_pipeline_end,
                args=(job.get("id"), pipeline, mode),
                daemon=True,
            ).start()
        except Exception as exc:
            print(f"[processor] job failed to start: {exc}", file=sys.stderr)
            pipeline, self.pipeline = self.pipeline, None
            if pipeline is not None:
                try:
                    pipeline.terminate()
                    pipeline.join()
                except Exception:
                    pass
            self._stop_sim_process()
            self._finalize_recorder()
            self._cleanup_download()
            with self.lock:
                self.state = "error"
                self.job = {**job, "error": str(exc)}
            # see the replay-failure comment above: retire → reaper requeues
            self.retire_if_pool_mode()
            return

    def _watch_pipeline_end(self, job_id, pipeline, mode):
        try:
            pipeline.join()
        except Exception:
            pass
        if self.cancelling:
            # stop_job is tearing this job down and owns cleanup; partial results
            # from a cancelled job must not be uploaded or marked complete
            return
        with self.lock:
            current = (
                self.job is not None
                and self.job.get("id") == job_id
                and self.state == "running"
            )
        if not current:
            return
        if mode == "batch":
            self._finalize_recorder()
            self._cleanup_download()
            self._upload_results(job_id)
            with self.lock:
                if self.job and self.job.get("id") == job_id and self.state == "running":
                    self.state = "completed"
                    print(f"[processor] job {job_id} completed; results are scrubbable")
        else:
            # a live pipeline ending on its own means the stream died (camera
            # offline, sim replay crashed). Free/retire the worker; the platform
            # reaper requeues the job once heartbeats stop carrying it.
            print(
                f"[processor] stream pipeline for job {job_id} ended unexpectedly; releasing job",
                file=sys.stderr,
            )
            self.stop_job()
            self.retire_if_pool_mode()

    def _upload_results(self, job_id):
        """Move finished batch results to durable storage via platform-signed URLs.

        The worker's disk is ephemeral: once this succeeds, nothing about the
        finished job depends on this process existing. Failure keeps the local
        copies servable via /results/ as a fallback.
        """
        if not self.args.api_url:
            return
        rdir = os.path.join(RESULTS_ROOT, str(job_id))
        files = [f for f in RESULT_FILES if os.path.isfile(os.path.join(rdir, f))]
        # a failed encode leaves an unplayable mp4 (no moov atom) — don't ship it
        meta_path = os.path.join(rdir, "meta.json")
        if "video.mp4" in files and os.path.isfile(meta_path):
            try:
                with open(meta_path) as f:
                    if not json.load(f).get("hasVideo"):
                        files.remove("video.mp4")
            except Exception:
                pass
        if not files:
            return
        try:
            resp = self.api(
                "POST",
                f"/video-jobs/{job_id}/results/upload-urls",
                json={"processorId": self.processor_id},
            )
            uploads = resp.get("uploads", {})
            uploaded = []
            for name in files:
                url = uploads.get(name)
                if not url:
                    continue
                with open(os.path.join(rdir, name), "rb") as fh:
                    # v4 signed PUT with no bound content type: send no
                    # Content-Type header (requests omits it for file bodies)
                    r = requests.put(url, data=fh, timeout=300)
                    r.raise_for_status()
                uploaded.append(name)
            if uploaded:
                self.api(
                    "POST",
                    f"/video-jobs/{job_id}/results/complete",
                    json={"files": uploaded, "processorId": self.processor_id},
                )
                print(f"[processor] uploaded results for {job_id}: {', '.join(uploaded)}")
        except Exception as exc:
            print(f"[processor] results upload failed (kept locally): {exc}", file=sys.stderr)

    def _finalize_recorder(self):
        recorder, self.recorder = self.recorder, None
        if recorder is not None:
            recorder.finalize()

    def _stop_sim_process(self):
        sim, self.sim_process = self.sim_process, None
        if sim is not None:
            try:
                sim.terminate()
                sim.wait(timeout=5)
            except Exception:
                try:
                    sim.kill()
                except Exception:
                    pass

    def _download_source(self, source_url: str, job_id) -> str:
        """Fetch a batch job's file to local disk so it gets true file semantics."""
        suffix = os.path.splitext(urlparse(source_url).path)[1] or ".mp4"
        fd, path = tempfile.mkstemp(prefix=f"rfv-job-{job_id}-", suffix=suffix)
        # track immediately so a mid-download failure still gets cleaned up
        self._downloaded_path = path
        started = time.time()
        with os.fdopen(fd, "wb") as out:
            with requests.get(source_url, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    out.write(chunk)
        size_mb = os.path.getsize(path) / 1e6
        print(f"[processor] downloaded source ({size_mb:.1f} MB in {time.time() - started:.1f}s) -> {path}")
        self._downloaded_path = path
        return path

    def _cleanup_download(self):
        path, self._downloaded_path = getattr(self, "_downloaded_path", None), None
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass

    def stop_job(self):
        # cancelling BEFORE taking the lifecycle lock: the pipeline-end watcher
        # must see it the instant our terminate() unblocks its join()
        self.cancelling = True
        # taken AFTER any in-flight start finishes — cancelling mid-start would
        # leak a running pipeline with no job attached
        with self.lifecycle_lock:
            publisher, self.publisher = self.publisher, None
            if publisher is not None:
                publisher.stop()
            pipeline, self.pipeline = self.pipeline, None
            if pipeline is not None:
                print("[processor] terminating pipeline")
                try:
                    pipeline.terminate()
                    pipeline.join()
                except Exception as exc:
                    print(f"[processor] error during terminate: {exc}", file=sys.stderr)
            self._stop_sim_process()
            self._finalize_recorder()
            self._cleanup_download()
            with self.lock:
                self.job = None
                self.state = "idle"

    def status(self) -> dict:
        with self.lock:
            job = self.job
            # public endpoint: never expose claim-payload internals — the job
            # dict carries the workspace api key and credentialed/signed URLs
            public_job = (
                {
                    "id": job.get("id"),
                    "mode": job.get("mode"),
                    "imageOutput": job.get("imageOutput"),
                    "error": job.get("error"),
                }
                if job
                else None
            )
            return {
                "processorId": self.processor_id,
                "state": self.state,
                "job": public_job,
                "stats": self.stats.snapshot(),
                # image outputs redacted from /events; each is watchable at
                # /preview.mjpeg?output=<name>
                "imageOutputs": self.frames.outputs(),
                "defaultImageOutput": self.image_output,
            }

    # ---------- platform polling ----------

    def api(self, method, path, **kwargs):
        url = f"{self.args.api_url.rstrip('/')}{path}"
        params = kwargs.pop("params", {})
        headers = kwargs.pop("headers", {})
        # Managed-pool (fleet) mode: authenticate with the service secret and
        # claim jobs across all workspaces. Without it, the worker key keeps
        # today's self-hosted behavior: workspace API key, workspace-scoped.
        fleet_secret = os.getenv("VIDEO_PROC_SERVICE_SECRET")
        if fleet_secret:
            headers["x-video-proc-service-access-token"] = fleet_secret
        else:
            params["api_key"] = self.args.api_key
        resp = requests.request(method, url, params=params, headers=headers, timeout=10, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def try_claim(self):
        """Claim (at most) one job if idle and eligible. Serialized so the poll
        loop and a Pub/Sub wake-up can't both start work on this worker.

        Returns the claimed job dict, None when the platform had no job, or
        False when this worker was not eligible to claim (busy/retiring/spent).
        The job pipeline is started on its OWN thread so the caller (usually the
        poll loop) keeps heartbeating while the source downloads and the model
        loads — otherwise any start longer than the reaper window would get the
        job requeued and double-processed."""
        with self.claim_lock:
            if self.retiring or self.state not in ("idle", "error"):
                return False
            if self.pod.enabled and self.had_job:
                # single-use invariant: a detached pod never takes a second job,
                # even if its retirement is still in flight or failed
                return False
            resp = self.api(
                "POST",
                "/video-jobs/claim",
                json={
                    "processorId": self.processor_id,
                    "processorUrl": self.public_url,
                },
            )
            job = resp.get("job")
            if not job:
                return None
            self.had_job = True
            self.cancelling = False
            self._publisher_degraded = False
            self._publish_baseline_ms = None
            self._latency_strikes = 0
            # leave the ready pool BEFORE the (slow) pipeline start so the
            # replacement worker is already warming while we work
            self.pod.detach_from_pool(job.get("id"))
            # claim the local state synchronously (no second claim can slip in),
            # then start the pipeline off-thread
            with self.lock:
                self.job = job
                self.state = "starting"
                self.image_output = job.get("imageOutput")
            threading.Thread(target=self.run_job, args=(job,), daemon=True).start()
            return job

    def _select_transport(self, job):
        if self._publisher_degraded:
            return "rtsp"
        transport = (job.get("publisherTransport") or PUBLISHER_TRANSPORT).lower()
        if transport == "whip":
            try:
                import aiortc  # noqa: F401
            except ImportError:
                print(
                    "[processor] aiortc not installed; using rtsp publisher",
                    file=sys.stderr,
                )
                return "rtsp"
        return transport if transport in ("whip", "rtsp") else "rtsp"

    def _make_publisher(self, job, transport):
        job_id = job.get("id", "local")
        if transport == "whip":
            whip_url = job.get("outWhipUrl") or f"{WHIP_SIM_BASE}/out-{job_id}/whip"
            # raw frames: the whip transport encodes straight from ndarrays
            return AiortcWhipPublisher(self.raw_frames, whip_url)
        publish_url = job.get("outPublishUrl") or f"{RTSP_SIM_BASE}/out-{job_id}"
        return OutputPublisher(self.frames, publish_url)

    def _handle_watch(self, watch, job):
        """React to the watch signal riding the status-poll response: publish
        the annotated output to the relay while someone is watching, stop when
        the TTL lapses. The viewer can switch outputs mid-stream, and the
        transport itself can hot-swap (config change or watchdog) — viewers
        just see a ~1s reconnect through the same relay stream."""
        if self.state != "running":
            return
        if watch.get("requested"):
            output = watch.get("output") or self.image_output
            if output is None:
                return  # no image output produced (JSON-only workflow)
            transport = self._select_transport(job)
            if self.publisher is not None and self.publisher.transport != transport:
                old, self.publisher = self.publisher, None
                old.stop()
            if self.publisher is None:
                # baseline BEFORE encoding starts: the watchdog compares
                # against unwatched inference latency
                self._publish_baseline_ms = self.stats.snapshot().get("decodeToResultLatencyMs")
                self._latency_strikes = 0
                self.publisher = self._make_publisher(job, transport)
            self.publisher.ensure(output)
            self._publisher_watchdog()
        elif self.publisher is not None:
            publisher, self.publisher = self.publisher, None
            publisher.stop()

    def _publisher_watchdog(self):
        """The invariant: customer inference latency beats preview transport.
        If the in-process (whip) publisher measurably degrades decode→result
        latency, hot-swap to the subprocess rtsp transport mid-job — viewers
        reconnect within ~1s, the pipeline never notices."""
        if self.publisher is None or self.publisher.transport != "whip":
            return
        baseline = self._publish_baseline_ms
        current = self.stats.snapshot().get("decodeToResultLatencyMs")
        if not baseline or not current:
            return
        if current > max(baseline * 1.8, baseline + 40):
            self._latency_strikes += 1
        else:
            self._latency_strikes = 0
        if self._latency_strikes >= 3:
            print(
                f"[processor] whip publisher degrading inference latency "
                f"({baseline:.0f}ms → {current:.0f}ms); hot-swapping to rtsp",
                file=sys.stderr,
            )
            self._publisher_degraded = True
            old, self.publisher = self.publisher, None
            old.stop()
            # the next watch tick recreates the publisher on rtsp

    def retire_if_pool_mode(self):
        """Single-use workers: once a job ends (completed, cancelled, or failed),
        the worker deletes its own pod instead of returning to the pool — a
        fresh ready worker already replaced it at claim time. If the delete
        fails (local dev, degraded RBAC) we fall back to long-lived behavior.
        The job itself is safe either way: an abrupt death here just means the
        platform requeues it via the heartbeat reaper."""
        if not self.pod.enabled or self.retiring:
            return
        self.retiring = True
        if not self.pod.self_delete():
            self.retiring = False

    def start_pubsub_listener(self):
        """Optional Pub/Sub wake-ups: messages are notifications, not work items —
        the claim endpoint stays the transactional source of truth. Busy workers
        nack so the backlog stays visible to autoscaling and redelivery reaches
        an idle worker; polling continues as the fallback."""
        subscription = self.args.pubsub_subscription
        if not subscription or not self.args.api_url:
            return
        try:
            from google.cloud import pubsub_v1
        except ImportError:
            print(
                "[processor] google-cloud-pubsub not installed; polling only",
                file=sys.stderr,
            )
            return

        def on_message(message):
            try:
                claimed = self.try_claim()
            except Exception:
                message.nack()
                return
            if claimed is False:
                # not eligible (busy/retiring/spent) — let redelivery reach an
                # idle worker and keep the backlog visible to autoscaling
                message.nack()
            else:
                # claimed a job, or the queue was already drained — consume it
                message.ack()

        self._pubsub_client = pubsub_v1.SubscriberClient()
        self._pubsub_client.subscribe(
            subscription,
            callback=on_message,
            flow_control=pubsub_v1.types.FlowControl(max_messages=1),
        )
        print(f"[processor] pubsub wake-ups on {subscription}")

    def poll_loop(self):
        print(f"[processor] {self.processor_id} polling {self.args.api_url} for jobs")
        while not self._stop_requested:
            try:
                if self.state in ("idle", "error"):
                    self.try_claim()
                else:
                    with self.lock:
                        job = self.job
                        state = self.state
                    if job is None:
                        # mid-transition (a concurrent stop); check again shortly
                        time.sleep(POLL_INTERVAL_S)
                        continue
                    resp = self.api(
                        "POST",
                        f"/video-jobs/{job['id']}/status",
                        json={
                            "state": state,
                            "stats": self.stats.snapshot(),
                            "processorId": self.processor_id,
                        },
                    )
                    if state == "completed":
                        print("[processor] completed job reported")
                        self.stop_job()
                        self.retire_if_pool_mode()
                    elif resp.get("cancel"):
                        print("[processor] job cancelled by platform")
                        self.stop_job()
                        self.retire_if_pool_mode()
                    else:
                        self._handle_watch(resp.get("watch") or {}, job)
            except requests.RequestException as exc:
                print(f"[processor] poll error: {exc}", file=sys.stderr)
            except Exception as exc:
                # the poll loop is the worker's heartbeat and cancel channel —
                # it must survive anything
                print(f"[processor] poll loop error: {exc}", file=sys.stderr)
            time.sleep(POLL_INTERVAL_S)

    def run(self):
        if self.args.job_file:
            with open(self.args.job_file) as f:
                job = json.load(f)
            job.setdefault("id", "local-job")
            self.run_job(job)
        else:
            if self.pod.is_detached():
                # crash-restarted container inside a spent (detached) pod: it
                # must not claim work — its old job is being requeued by the
                # reaper, and a hidden worker outside the pool breaks the
                # single-use invariant. Retire the pod instead.
                print("[processor] restarted inside a detached pod — retiring", file=sys.stderr)
                self.retiring = True
                self.pod.self_delete()
                return
            self.start_pubsub_listener()
            threading.Thread(target=self.poll_loop, daemon=True).start()


# ---------- HTTP ----------


def make_handler(worker: Worker):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/status":
                body = json.dumps(worker.status(), default=str).encode()
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/events":
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = worker.events.subscribe()
                try:
                    while True:
                        try:
                            data = q.get(timeout=15)
                            self.wfile.write(f"data: {data}\n\n".encode())
                        except Exception:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    worker.events.unsubscribe(q)
            elif path == "/preview.mjpeg":
                query = parse_qs(urlparse(self.path).query)
                requested_output = (query.get("output") or [None])[0]
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                seq = -1
                min_interval = 1.0 / MJPEG_MAX_FPS
                last_sent = 0.0
                try:
                    while True:
                        # resolved per-iteration so a stream opened before the first
                        # result attaches to the default output once it exists
                        output = requested_output or worker.image_output
                        if output is None:
                            time.sleep(0.2)
                            continue
                        jpeg, seq = worker.frames.wait_next(output, seq)
                        if jpeg is None:
                            continue
                        wait = min_interval - (time.time() - last_sent)
                        if wait > 0:
                            time.sleep(wait)
                        last_sent = time.time()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif path == "/metrics":
                # busy gauge drives the warm-pool autoscaler:
                # desired replicas = sum(busy) + MIN_IDLE
                busy = 1 if worker.state in ("starting", "running") else 0
                body = (
                    "# HELP video_processor_busy 1 while a job is assigned to this worker\n"
                    "# TYPE video_processor_busy gauge\n"
                    f"video_processor_busy {busy}\n"
                ).encode()
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path.startswith("/results/"):
                # /results/<jobId>/(video.mp4|frames.jsonl|meta.json) — persisted
                # batch-job results. Range support is what makes <video> scrubbing
                # work: the browser seeks by requesting byte ranges of the mp4.
                parts = path.split("/")
                if (
                    len(parts) == 4
                    and JOB_ID_RE.match(parts[2])
                    and parts[3] in RESULT_FILES
                ):
                    fpath = os.path.join(RESULTS_ROOT, parts[2], parts[3])
                    if os.path.isfile(fpath):
                        self._serve_file(fpath, RESULT_FILES[parts[3]])
                        return
                self._not_found()
            else:
                self._not_found()

        def _not_found(self):
            self.send_response(404)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _serve_file(self, fpath, content_type):
            size = os.path.getsize(fpath)
            start, end, status = 0, size - 1, 200
            range_header = self.headers.get("Range", "")
            match = re.match(r"bytes=(\d*)-(\d*)$", range_header)
            if match and (match.group(1) or match.group(2)):
                if match.group(1):
                    start = int(match.group(1))
                    if match.group(2):
                        end = min(int(match.group(2)), size - 1)
                else:  # suffix range: last N bytes
                    start = max(0, size - int(match.group(2)))
                if start > end or start >= size:
                    self.send_response(416)
                    self._cors()
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206
            length = end - start + 1
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(fpath, "rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Roboflow video POC processor")
    parser.add_argument("--api-url", default=os.getenv("RF_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"))
    parser.add_argument("--job-file", help="run a single local job definition (skips platform polling)")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--processor-id", default=None)
    parser.add_argument(
        "--public-url",
        default=os.getenv("PROCESSOR_PUBLIC_URL"),
        help="externally reachable base URL reported as processorUrl (e.g. an ingress path)",
    )
    parser.add_argument(
        "--pubsub-subscription",
        default=os.getenv("PROCESSOR_PUBSUB_SUBSCRIPTION"),
        help="optional GCP Pub/Sub subscription for instant job wake-ups (polling remains the fallback)",
    )
    args = parser.parse_args()

    if not args.job_file and not args.api_url:
        parser.error("either --job-file or --api-url is required")

    worker = Worker(args)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(worker))

    def shutdown(*_):
        print("\n[processor] shutting down")
        worker._stop_requested = True
        worker.stop_job()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    worker.run()
    print(f"[processor] http on :{args.port}  (/status /events /preview.mjpeg)")
    server.serve_forever()
    print("[processor] bye")


if __name__ == "__main__":
    main()
