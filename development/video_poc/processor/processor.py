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

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

POLL_INTERVAL_S = 2.0
EVENT_BUFFER_SIZE = 50
MJPEG_MAX_FPS = 12

# Used by stream-mode jobs on file sources: the file is replayed at native speed
# into the local relay and consumed as RTSP, so the pipeline sees a real stream.
RTSP_SIM_BASE = os.getenv("VIDEO_POC_RTSP_BASE", "rtsp://127.0.0.1:8554")
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
    "VIDEO_POC_RESULTS_DIR", os.path.join(tempfile.gettempdir(), "rf-video-poc-results")
)
RESULT_FILES = {
    "video.mp4": "video/mp4",
    "frames.jsonl": "application/x-ndjson",
    "meta.json": "application/json",
}
JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def utcnow_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def is_serialized_image(value) -> bool:
    return isinstance(value, dict) and value.get("type") == "base64" and "value" in value


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
        self.video_failed = False
        self.lock = threading.Lock()

    def add(self, jpeg_bytes, fps, event: dict):
        with self.lock:
            if self.events_file is None:
                return
            self.events_file.write(json.dumps(event, default=str) + "\n")
            self.frames += 1
            if jpeg_bytes is None or self.video_failed:
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
            meta = {
                "jobId": self.job_id,
                "fps": self.fps,
                "frames": self.frames,
                "hasVideo": self.ffmpeg is not None and not self.video_failed,
            }
            with open(os.path.join(self.dir, "meta.json"), "w") as f:
                json.dump(meta, f)


class Worker:
    def __init__(self, args):
        self.args = args
        self.processor_id = args.processor_id or f"proc-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        # what browsers should use to reach this worker (an ingress/gateway URL
        # in a cluster); falls back to localhost for single-machine dev
        self.public_url = args.public_url or f"http://127.0.0.1:{args.port}"
        # serializes claims between the poll loop and the Pub/Sub wake-up so a
        # worker can never start two jobs
        self.claim_lock = threading.Lock()
        self._pubsub_client = None
        self.stats = Stats()
        self.events = EventBus()
        self.frames = FrameStore()
        self.pipeline = None
        self.sim_process = None
        self.recorder = None
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

        outputs = {}
        designated_jpeg = None
        for key, value in predictions.items():
            if is_serialized_image(value):
                if self.image_output is None:
                    self.image_output = key
                try:
                    jpeg = base64.b64decode(value["value"])
                    self.frames.set(key, jpeg)
                    if key == self.image_output:
                        designated_jpeg = jpeg
                except Exception:
                    pass
                outputs[key] = {"type": "image_ref", "output": key}
            elif isinstance(value, list) and value and all(is_serialized_image(v) for v in value):
                outputs[key] = [{"type": "image_ref", "output": key, "index": i} for i in range(len(value))]
            else:
                outputs[key] = value

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
        with self.lock:
            self.job = job
            self.state = "starting"
            self.image_output = job.get("imageOutput")
        self.frames.clear()
        self.stats.on_job()

        source_url = job["sourceUrl"]
        mode = job.get("mode") or ("batch" if source_url.startswith("http") else "stream")
        print(f"[processor] starting job {job.get('id')} ({mode}) on {source_url}")

        video_reference = source_url
        pipeline_kwargs = {}
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
                    with self.lock:
                        self.state = "error"
                        self.job = {**job, "error": f"source download failed: {exc}"}
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
                return

        try:
            self.pipeline = InferencePipeline.init_with_workflow(
                video_reference=video_reference,
                workflow_specification=job.get("workflowSpecification"),
                workspace_name=job.get("workspaceName"),
                workflow_id=job.get("workflowId"),
                # the job carries its workspace's key so model access and usage
                # follow the job, not this worker's identity key
                api_key=job.get("apiKey") or self.args.api_key or os.getenv("ROBOFLOW_API_KEY"),
                on_prediction=self.on_prediction,
                serialize_results=True,
                max_fps=job.get("maxFps"),
                **pipeline_kwargs,
            )
            self.stats.on_pipeline_start()
            self.pipeline.start(use_main_thread=False)
            with self.lock:
                self.state = "running"
            if mode == "batch":
                # a batch job ends when the file does; detect it so results become
                # scrubbable and the worker frees up for the next job
                threading.Thread(
                    target=self._watch_pipeline_end,
                    args=(job.get("id"), self.pipeline),
                    daemon=True,
                ).start()
        except Exception as exc:
            print(f"[processor] job failed to start: {exc}", file=sys.stderr)
            self._stop_sim_process()
            self._finalize_recorder()
            with self.lock:
                self.state = "error"
                self.job = {**job, "error": str(exc)}
            return

    def _watch_pipeline_end(self, job_id, pipeline):
        try:
            pipeline.join()
        except Exception:
            pass
        self._finalize_recorder()
        self._cleanup_download()
        self._upload_results(job_id)
        with self.lock:
            if self.job and self.job.get("id") == job_id and self.state == "running":
                self.state = "completed"
                print(f"[processor] job {job_id} completed; results are scrubbable")

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
        if not files:
            return
        try:
            resp = self.api("POST", f"/video-jobs/{job_id}/results/upload-urls", json={})
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
                self.api("POST", f"/video-jobs/{job_id}/results/complete", json={"files": uploaded})
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
            return {
                "processorId": self.processor_id,
                "state": self.state,
                "job": self.job,
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
        params["api_key"] = self.args.api_key
        resp = requests.request(method, url, params=params, timeout=10, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def try_claim(self):
        """Claim (at most) one job if idle. Serialized so the poll loop and a
        Pub/Sub wake-up can't both start work on this worker."""
        with self.claim_lock:
            if self.state not in ("idle", "error"):
                return None
            resp = self.api(
                "POST",
                "/video-jobs/claim",
                json={
                    "processorId": self.processor_id,
                    "processorUrl": self.public_url,
                },
            )
            job = resp.get("job")
            if job:
                self.run_job(job)
            return job

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
            if self.state not in ("idle", "error"):
                message.nack()
                return
            try:
                self.try_claim()
                message.ack()
            except Exception:
                message.nack()

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
                    state = self.state
                    resp = self.api(
                        "POST",
                        f"/video-jobs/{self.job['id']}/status",
                        json={"state": state, "stats": self.stats.snapshot()},
                    )
                    if state == "completed":
                        # results stay on disk and servable; the worker frees up
                        print("[processor] completed job reported; back to idle")
                        self.stop_job()
                    elif resp.get("cancel"):
                        print("[processor] job cancelled by platform")
                        self.stop_job()
            except requests.RequestException as exc:
                print(f"[processor] poll error: {exc}", file=sys.stderr)
            time.sleep(POLL_INTERVAL_S)

    def run(self):
        if self.args.job_file:
            with open(self.args.job_file) as f:
                job = json.load(f)
            job.setdefault("id", "local-job")
            self.run_job(job)
        else:
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
