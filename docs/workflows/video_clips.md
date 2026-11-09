# Video clips on Vision Events

A Workflow that runs on a video stream can attach short video clips to the Vision
Events it sends. Two blocks work together:

* **Video Clip Recorder** (`roboflow_core/video_clip_recorder@v1`) keeps a rolling
  H.264 buffer of the frames the Workflow processes, per video stream, and outputs an
  `anchor` (kind `video_clip_anchor`) for every frame.
* **Roboflow Vision Events** (`roboflow_core/roboflow_vision_events@v1`, numpy and
  tensor-native variants) takes 1 to 4 anchors in `video_clips`. When it sends an
  event, it captures a clip around the triggering frame from each recorder, posts the
  event with the clips as pending videos, and uploads each clip as soon as its
  post-roll has been recorded.

This page describes what a clip contains, how to wire the blocks, where they run,
what they cost and how to configure them.

## Wiring

```json
{
    "steps": [
        {
            "type": "roboflow_core/video_clip_recorder@v1",
            "name": "raw_recorder",
            "image": "$inputs.image",
            "label": "Raw"
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "boxes",
            "image": "$inputs.image",
            "predictions": "$steps.model.predictions"
        },
        {
            "type": "roboflow_core/video_clip_recorder@v1",
            "name": "annotated_recorder",
            "image": "$steps.boxes.image",
            "label": "Annotated"
        },
        {
            "type": "roboflow_core/roboflow_vision_events@v1",
            "name": "events",
            "event_type": "safety_alert",
            "video_clips": [
                "$steps.raw_recorder.anchor",
                "$steps.annotated_recorder.anchor"
            ]
        }
    ]
}
```

The model step and the rest of the sink's settings (use case, event data, images)
are left out. Guidelines:

* Wire the input image into a recorder for a raw clip, and a visualization output
  for an annotated clip. Use one recorder per clip.
* Keep recorders on the unconditional path and gate the sink instead. A recorder
  under flow control only records the frames on which its branch runs, so clips lose
  their pre-roll; the compiler warns about it.
* On Serverless Streaming (and any host with CUDA tensor frames), keep
  `copy_image: true` (the default) on a visualization that feeds a recorder. An
  in-place tensor visualization hands the recorder a host image, which forces the
  x264 path for that recorder: in the acceptance run on an L4 the job dropped from
  30 to 28.3 fps with 15 ms encodes per frame. With `copy_image: true` both
  recorders stay on NVENC.
* Keep `fire_and_forget` enabled on the sink (the default). With
  `fire_and_forget: false` the upload slot requests and the event POST run on the
  pipeline thread: in the Serverless Streaming validation each `prepare-upload` took
  0.28-0.38 s and the event POST 0.36-0.55 s, during which no frame is processed.
* The anchor never leaves the process. Binding it into a step that runs elsewhere
  (an inner workflow with remote dispatch, Custom Python running remotely, a
  Roboflow model step running remotely) or exposing it as a workflow output is a
  compile-time error. The check is conservative: an Expression-style block that
  receives an anchor is treated as passing it on, even if it only reads metadata.

## What a clip contains

A clip is approximate evidence with reported bounds.

* **Processed frames only.** The recorder sees the frames the Workflow processes, so
  a clip's frame density follows the processing rate, not the camera rate. A
  pipeline that processes 4 fps produces a 4 fps clip that plays in real time with
  long frame holds.
* **Window.** `[trigger - pre_roll_seconds, trigger + post_roll_seconds]` on the
  `frame_timestamp` axis. The end is inclusive, so a post-roll of 0 still contains
  the triggering frame.
* **Keyframe lead.** A clip starts on the keyframe before the window, up to one
  output second early (`playback_speed` source seconds). The lead is reported as
  `lead_seconds`.
* **Timing.** Frames keep their source timing, divided by `playback_speed`. The first
  frame has PTS 0. Duplicate or backward timestamps are repaired by one tick and
  counted. A frame is never moved forward into the window: a long gap stays a gap.
* **Frame rate cap.** With `clip_fps`, the recorder keeps the first frame of each
  `playback_speed / clip_fps` source-second slot and counts the rest as decimated.
  Without it, every processed frame is kept.
* **Size.** `max_width` downscales wider frames, keeping the aspect ratio.
* **Copies.** The recorder converts or copies the frame inside its step. A
  visualization that later draws in place (`copy_image: false`) on the same image
  never reaches a raw clip.
* **Encoder start-up (NVENC).** See [Encoder start-up](#encoder-start-up): frames
  that arrive before the GPU encoder is ready are not recorded and are counted as
  `startup_dropped`.
* **Stale frames.** A frame whose processing lag (the time from its capture
  timestamp to the recorder's step) exceeds `max_processing_lag_seconds` (default 5)
  never enters a clip; it is counted as `stale_dropped`. Frames read from video files
  are exempt. This is the stale rule, not an encoder limit, and it matters when a
  host processes its first frames late. On the Serverless Streaming sandbox the
  first frame was processed 5.1-5.3 s after it was grabbed under gVisor (2.1 s under
  runc) because of model warm-up. A trigger on one of those first frames anchors the
  window on the old frame: the frames that follow arrive 2.1 s (runc) or more than
  5 s (gVisor) later, so the clip is short (about 88 frames of a 10 s window under
  runc, 1 frame under gVisor), or the video ends as `clip_failed` ("No frames were
  buffered in the requested window") when the recorder sees the trigger frame
  itself later than the limit. Raise `max_processing_lag_seconds` on such hosts if
  early triggers need clips.
* **Restarts.** A frame number going backwards or a resolution change starts a new
  recording; a clip in progress ends there as partial (`reset` or
  `resolution_change`). The previous recording's memory is freed as soon as its
  last pending clip is finished.
* **Streams.** State is kept per `video_identifier`, for at most 16 streams per
  recorder; the least recently seen stream's recording is discarded beyond that.
* No audio.

Limits of the time axis: frames that arrive in a burst after a stall (for example
WebRTC catching up) carry compressed capture times, and file sources are stamped
from their declared fps.

## Settings and limits

| Recorder setting | Default | Range |
| --- | --- | --- |
| `label` | `Evidence` | 1 to 256 characters; the Vision Events UI truncates long captions, so short labels such as `Raw` read best |
| `pre_roll_seconds` | 5 | 0 to 60 (host-raisable to 300) |
| `post_roll_seconds` | 5 | 0 to 60 (host-raisable to 300) |
| `playback_speed` | 1 | 0.1 to 32 |
| `clip_fps` | `null` (source timing) | 1 to 120 |
| `max_width` | `null` (native) | even, at least 64 |
| `max_processing_lag_seconds` | 5 | |

Raise the roll caps with `WORKFLOWS_VIDEO_CLIP_MAX_PRE_ROLL_SECONDS` and
`WORKFLOWS_VIDEO_CLIP_MAX_POST_ROLL_SECONDS` (see [Environment variables](#environment-variables)).
The post-roll, encode and upload must fit the upload URL's 1 hour lifetime.

The platform keeps the first 4 videos of an event, so the sink attaches up to 4
clips in the order given and reports extra anchors as `rejected:too_many_videos`.
The sink's `video_label` is used only when a recorder's label is empty.

## Sink outputs and statuses

The sink's `videos` output has one entry per anchor, in request order:

```json
{
    "video_id": "...",
    "label": "Raw",
    "video_status": "pending",
    "requested_window": {"start": "...", "end": "..."},
    "available_from": "..."
}
```

`video_status` is the immediate status, never a finished upload:

| `video_status` | Meaning |
| --- | --- |
| `pending` | Captured and admitted; the event references the video and the clip uploads when its post-roll is recorded |
| `disabled`, `throttled` | The sink was disabled or in its cooldown; nothing was captured |
| `unavailable:<reason>` | No clip: the recorder's anchor was unavailable (`no_timestamp`, `no_encoder`, `sinks_disabled`, `unsupported_host`), the recorder was stopping (`closing`), the capture failed (`capture_failed`), the input was not an anchor (`no_anchor`, `invalid_anchor`), or the sink writes to the local Event Store (`unsupported_sink_mode`) |
| `rejected:<reason>` | Captured but not attached: more than 4 anchors (`too_many_videos`), a resource cap (`budget_exceeded`, `too_many_pending_captures`, `clip_too_large`, or `admission_error` if the check itself failed), the event POST failed (`event_failed`), or the uploader was stopping (`uploader_unavailable`) |
| `prepare_failed` | The upload slot request failed; the event is sent without that video |

With `fire_and_forget` enabled, `video_id` is empty and `prepare_failed` is not
visible in the output, because the upload slot is requested in the background.

## Completion record

Each clip's outcome is logged once, at INFO level, as a structured
`vision_event_clip_completed` record:

| Field | Content |
| --- | --- |
| `event_id`, `video_id`, `label` | Which video of which event |
| `terminal_status` | `put_accepted`, `upload_failed`, `expired`, `clip_failed`, `cancelled` or `shutdown_deadline` |
| `report` | The clip report below, when a clip exists |
| `timings` | ISO timestamps `captured_at`, `ready_at`, `uploaded_at` |
| `ingestion` | `null`, or with `confirm_ingestion: true` one of `confirmed`, `pending`, `failed` after polling for up to 5 minutes |
| `error` | Redacted error text for failed statuses |

`put_accepted` means the storage accepted the upload; the platform may still be
ingesting it. Transient upload errors (connection errors, 5xx, 408, 429) are retried
with backoff until `clip_upload_timeout_seconds` (default 55 minutes) after the slot
was requested. Signed URL query strings never appear in logs or errors.

The `report` holds `requested_window`, `actual_start`, `actual_end`, `lead_seconds`,
`frames`, `frame_density_fps`, `source_fps_declared`, `playback_speed`, `clip_fps`,
`width`, `height`, `backend` (`x264` or `nvenc`), `partial`, `loss`,
`wall_clock_jump_flagged`, `encode_ms_p50`/`p95`, `ring_cost_ms_p50`/`p95` (time
spent on the pipeline thread per frame) and `finalize_ms`.

`partial` is `null` or the reason a clip is shorter than requested: `stalled` (no
frames arrived for 10 s, and at least the remaining post-roll plus 10 s passed since
the capture; a source processed slower than real time is not stalled while frames
keep arriving), `reset` (the source restarted: its frame number went backwards),
`resolution_change`, `shutdown` (the pipeline stopped during the post-roll) or
`ring_evicted` (the window start was no longer buffered).

`loss` counts the frames missing from the window by cause: `decimated` (`clip_fps`
grid), `ring_drops` (buffer byte cap), `queue_drops` (encoder queue full, or dropped
at shutdown), `startup_dropped` (the NVENC encoder was still starting, see [Encoder
start-up](#encoder-start-up)), `stale_dropped` (processing lag), `inference_gaps`
(intervals more than twice the nominal frame interval) and `pts_clamped` (repaired
timestamps).

## Encoder backends

The recorder picks a backend per stream at its first frame:

* **NVENC**, when the frame is a CUDA tensor and PyNvVideoCodec imports (Serverless
  Streaming GPU runners). Inside the step the recorder makes one GPU-to-GPU copy of
  the frame into a pooled buffer on the pipeline's own CUDA stream and records one
  event: two CUDA calls per frame. The wait, the `max_width` resize, a copy into a
  packed RGBA buffer and the encode run on the recorder's own thread and CUDA
  stream, so no frame is copied to the CPU and only encoded packets reach host
  memory. NVENC converts RGB to YUV itself (decoded colours match a BT.709
  conversion: mean error 1 grey level on an L4 colour-bar test), so the recorder's
  thread makes about four CUDA and encoder calls per frame. Keeping both threads to
  a few calls matters under gVisor, where every CUDA driver call goes through the
  sandbox and every call that releases the GIL competes with the host's threads:
  converting to NV12 in torch (about forty calls per frame) raised the Serverless
  Streaming preview's p95 latency by about 25% with two recorders on 2 CPUs. On an RTX 4080 at 1080p the step costs
  0.09-0.14 ms per frame (p50; p95 at most 0.32 ms), down from 0.45-0.76 ms (p95 up
  to 1.2 ms) when the conversion ran in the step. Bitrate: 4 Mbit/s at 1080p30 by
  default, scaled by pixel rate.
* **x264**, otherwise. The frame is converted to I420 with OpenCV inside the step
  (about 1.2-1.5 ms at 1080p) and encoded with libx264 (`ultrafast`, `zerolatency`,
  one thread) on the recorder's own thread.

Both write an IDR frame every output second and mux clips to MP4 with
`+faststart`. Encoding, muxing and uploading never use the Workflow's shared thread
pools. Clips are muxed one at a time per process, with a short pause every 32
packets, so clips that finish together do not compete with the pipeline at once;
a clip held in memory is uploaded as one buffer.

### Encoder start-up

NVENC needs process-wide preparation once (the PyNvVideoCodec import, the CUDA
context, the conversion kernels and the encoder driver) and then an encoder session
per stream. Neither runs on the pipeline thread:

* **Process warm-up.** A recorder constructed on a CUDA host (torch already imported
  with CUDA image tensors, or a CUDA context already initialised, and PyNvVideoCodec
  installed) starts the warm-up on a background thread when the Workflow is
  compiled. It takes 0.7-0.8 s on an RTX 4080, so it normally finishes during the
  host's model load, before the first frame. CPU hosts and hosts without
  PyNvVideoCodec skip it and never import torch for it. Frames that arrive before it
  finishes are not recorded and are counted as `startup_dropped`; the stream does
  not fall back to x264 while NVENC is starting. If the warm-up finds NVENC
  unavailable, or has not finished after 60 s, the stream records with x264 from
  then on.
* **Session open.** Each stream's first frame opens its encoder session on the
  recorder's thread. One session takes 0.43-0.55 s on an RTX 4080 (0.35-0.4 s on an
  L4); when two streams start together, the second session is ready 0.6-1.0 s after
  the first frame. Until its session is open, a stream's GPU frame pool may grow
  from 8 to 32 frames (190 MiB at 1080p, never more than 192 MiB), so those frames
  are kept. Once the backlog is encoded the extra frames are released and their
  GPU memory is handed back to the driver (one `torch.cuda.empty_cache()` per
  stream start-up), so steady state holds the 8-frame pool. Frames refused in that
  window are also counted as `startup_dropped`.

Measured on an RTX 4080 with 1080p CUDA frames, 300 frames per run, three runs per
row, the warm-up started 3 s before the first frame (a stand-in for the model load)
unless noted:

| Streams and rate | Start-up loss per stream | Frames lost after start-up |
| --- | --- | --- |
| 1 at 30 fps or 60 fps | 0 | 0 |
| 2 at 30 fps | 0 | 0 |
| 2 at 60 fps | 5-19 (the second session opens 0.6-1.0 s after the first frame) | 0-1 |
| 2 at 30 fps, frames from the moment the Workflow is compiled | 16-19 (the warm-up, about 0.6 s) | 0 |

Before this start-up handling the same runs lost 8-46 frames per stream in the first
second (8-21 at 30 fps), and on an L4 the first frame held the pipeline thread for
1.4 s (4.2 s in a cold process). A host that runs two 60 fps streams per process and
needs their first second can raise `WORKFLOWS_VIDEO_CLIP_NVENC_POOL_FRAMES`, which
also raises the start-up pool (48 frames is about 300 MB of GPU memory per stream at
1080p, held all the time).

## Memory

Measured in validation with the x264 backend at 1080p (CRF 23):

* The buffer costs 0.82-0.94 MB per second of window per stream. It normally holds
  the pre-roll plus about one second; while a capture is pending it holds the
  pre-roll, the post-roll and about 1.5 s. Overlapping captures share packets.
* Peaks per stream: 10 MB at 5 s + 5 s, 32 MB at 30 s + 5 s, 198 MB at 120 s + 120 s.
* Annotated clips are about 1.9 times the bytes of raw clips.
* Each x264 encoder adds about 60 MB of process memory.
* Each NVENC session used 64-142 MB of VRAM, plus its frame pool: 8 RGB frames (50 MB
  at 1080p) in steady state and up to 190 MiB at 1080p, capped at 192 MiB while the session opens.
  The start-up extra is returned to the driver once the backlog is encoded. On an L4
  with the Serverless Streaming processor at 1080p, two recorders raised whole-GPU
  memory from 516 MiB to 784 MiB in steady state, about 134 MiB per recorder (torch
  reserved 206 MiB in all); one recorder raised it to 669 MiB (153 MiB, which includes
  the once-per-process NVENC start-up). Before the start-up pool was returned, two
  recorders held 1042 MiB (about 263 MiB each, torch reserved 498 MiB).

Process-wide caps bound the total: `WORKFLOWS_VIDEO_CLIP_BUDGET_BYTES` covers buffered
packets, encoder queues, finished clips and in-flight uploads (not the NVENC GPU
frame pool). A capture that would exceed a cap is rejected with `rejected:<reason>`;
the event is still sent.

## Host support

| Host | Video clips |
| --- | --- |
| InferencePipeline and the stream manager (self-hosted, RFDM devices) | Supported; frame density equals the processing rate |
| Dedicated Deployments and Serverless WebRTC | Supported. Serverless WebRTC sessions with a time limit keep about 1 s for shutdown, so a clip still in its post-roll at the limit is lost |
| Serverless Streaming | Needs a cell image with a `roboflow-workflows` release that contains these blocks; until then the Workflow fails to compile like any unknown block. Cancelling a job currently kills the process, so a clip still in its post-roll or upload at cancel is lost |
| Batch video processing (`process_video_with_workflow`) | Clips follow `max_video_fps`. Validated locally; an end-to-end upload from hosted Batch Processing has not been tested |
| Vision Events sink in local Event Store mode | Event without video (`unavailable:unsupported_sink_mode`) |
| HTTP `/workflows/run`, Serverless Image API | Event without video (`unavailable:unsupported_host`) |
| Anchor bound into a remote step or a workflow output | Compile-time error |
| Older Inference or `roboflow-workflows` | The Workflow does not compile |

Hosts that build an `InferencePipeline` themselves and do not pass their
environment through to `roboflow-workflows` (for example a runner that sets
`WORKFLOWS_BLOCK_CLOSE_DEADLINE_SECONDS` but never installs it in the Workflows
configuration) set the block close deadline on the pipeline before `join()`:
`pipeline.set_block_close_deadline(seconds)`, for example the shutdown grace minus
the time the host needs after the pipeline ends. `None` restores the environment
default. It applies to pipelines built by `init_with_workflow`.

When a pipeline stops, the recorder stops taking frames, drops frames not yet
encoded (counted as `queue_drops`), finishes pending clips as `partial: shutdown`
and uploads them within the host's close deadline (`WORKFLOWS_BLOCK_CLOSE_DEADLINE_SECONDS`,
default 30 s). An upload still running at the deadline is not cut off: its outcome is
recorded when it returns. Clips of a fire-and-forget event whose background task
was cancelled before it ran are released, with a warning in the log.

Behind a secure gateway (`SECURE_GATEWAY`), the upload slot request, the clip upload
to its signed URL and the ingestion check all go through the gateway.

## Performance envelope

Bars: at most 5% loss of processed fps and at most 10% increase in p95 latency.
Measured during validation on a prototype with the same encoder settings, with a CPU
stand-in model at its natural rate and in-process software decode, unless noted. The
numbers depend on the device, the model and the content; reruns on this
implementation are pending.

| Device | Passes the bars | Measured but not passing |
| --- | --- | --- |
| x86, 8 vCPU (c3-standard-8) | 2x1080p30 (2 repeats) | 4x1080p30: -15% fps (CPU saturated); -5.3% with a GPU-bound model |
| Jetson AGX Thor | 4x1080p30 (-2.1 to -2.6%, 2 repeats) and 8x720p30 (-4.2 to -4.3%) | |
| Jetson AGX Orin | 4x1080p30 (-3.8 to -4.0%) | 8x720p30: about -5% (-5.4% over 30 s) |
| Jetson Orin NX, Orin Nano | Not tested | |
| GPU server (L4, RTX 4080), numpy frames | 720p targets (RTX 4080) | 8x1080p30 in a CPU-decode harness (the L4 baseline alone used 6.5 cores) |
| Serverless Streaming sandbox (L4 standing in for L40S, the cell's processor image, rfdetr-nano, raw and annotated recorders, NVENC, 2 repeats) | runc, 8 vCPU, two recorders: 30.0 fps, viewer p95 154.1 to 157.3 ms (+2.1%). gVisor (runsc-kvm-nvproxy) with 2 CPUs and one recorder: p95 157.6 to 162.2 ms (+2.9%). gVisor with 4 CPUs and two recorders: 156.8 to 162.1 ms (+3.4%). fps within 0.6% everywhere; about 0.15-0.29 ms per frame on the pipeline thread | gVisor with 2 CPUs and two recorders: p95 157.6 to 170.0 and 179.6 ms (+7.7% and +14.1%, so the 10% bar is met in one repeat of two), 30.0 fps; the extra latency sits mostly in the second after two clips finish together. The x264 path instead of NVENC: 21 fps |

Recording cost grows with processed fps and resolution. On constrained devices set
`max_width` or lower the model fps before adding streams. On a 2-CPU gVisor sandbox,
keep to one recorder per stream, or give the sandbox 4 CPUs, when the preview's
latency matters; host CPU was 0.50 cores with recorders off, 0.60 with one and 0.65
with two.

## Environment variables

All are optional. Invalid or non-positive values fall back to the default with a
warning.

| Variable | Default | Purpose |
| --- | --- | --- |
| `WORKFLOWS_VIDEO_CLIP_MAX_PRE_ROLL_SECONDS` | `60` | Largest accepted `pre_roll_seconds`; values above 300 are clamped to 300 |
| `WORKFLOWS_VIDEO_CLIP_MAX_POST_ROLL_SECONDS` | `60` | Largest accepted `post_roll_seconds`; values above 300 are clamped to 300 |
| `WORKFLOWS_VIDEO_CLIP_BUDGET_BYTES` | 1 GiB | Process-wide byte cap over buffered packets, encoder queues, finished clips and in-flight uploads |
| `WORKFLOWS_VIDEO_CLIP_MAX_PENDING_CAPTURES` | `16` | Admitted clips that have not reached a terminal status |
| `WORKFLOWS_VIDEO_CLIP_MAX_UPLOAD_CONCURRENCY` | `2` | Clip uploads running at once in the process |
| `WORKFLOWS_VIDEO_CLIP_MAX_CLIP_BYTES` | 512 MiB | Largest clip, estimated at capture and checked again after muxing |
| `WORKFLOWS_VIDEO_CLIP_INMEMORY_MAX_BYTES` | 64 MiB | Clips up to this size are held in memory; larger clips stay on disk until uploaded |
| `WORKFLOWS_VIDEO_CLIP_TEMP_DIR` | system temp directory | Directory clips are muxed into; each clip file is created readable by its owner only (mode `0600`) |
| `WORKFLOWS_VIDEO_CLIP_NVENC_POOL_FRAMES` | `8` | GPU frames queued per NVENC stream (1 to 256) once its encoder session is open; a full pool counts `queue_drops`. While the session opens the pool may grow to 32 frames (see [Encoder start-up](#encoder-start-up)) |
| `WORKFLOWS_VIDEO_CLIP_NVENC_BITRATE_KBPS` | scaled (4000 at 1080p30) | NVENC bitrate in kbit/s (100 to 200000) |
| `WORKFLOWS_BLOCK_CLOSE_DEADLINE_SECONDS` | `30` | Seconds a stopping InferencePipeline gives Workflow blocks to finish, for example to upload pending clips |

The byte, capture, concurrency and file settings are read once, when
`roboflow-workflows` is imported. The roll caps are read when a Workflow is
compiled, and the NVENC settings when a stream's encoder starts.
