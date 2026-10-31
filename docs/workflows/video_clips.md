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
* **Stale frames.** Frames processed more than `max_processing_lag_seconds` (default
  5) after their capture timestamp are not recorded, for example the first frames
  after a slow model load. Frames read from video files are exempt.
* **Restarts.** A frame number going backwards or a resolution change starts a new
  recording; a clip in progress ends there as partial.
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
frames for the remaining post-roll plus 10 s), `resolution_change`, `shutdown` (the
pipeline stopped during the post-roll) or `ring_evicted` (the window start was no
longer buffered).

`loss` counts the frames missing from the window by cause: `decimated` (`clip_fps`
grid), `ring_drops` (buffer byte cap), `queue_drops` (encoder queue full, or dropped
at shutdown), `stale_dropped` (processing lag), `inference_gaps` (intervals more than
twice the nominal frame interval) and `pts_clamped` (repaired timestamps).

## Encoder backends

The recorder picks a backend per stream at its first frame:

* **NVENC**, when the frame is a CUDA tensor and PyNvVideoCodec imports (Serverless
  Streaming GPU runners). The frame is converted to NV12 on the GPU and encoded by
  NVENC without a copy to the CPU. Only encoded packets reach host memory. Bitrate:
  4 Mbit/s at 1080p30 by default, scaled by pixel rate.
* **x264**, otherwise. The frame is converted to I420 with OpenCV inside the step
  (about 1.2-1.5 ms at 1080p) and encoded with libx264 (`ultrafast`, `zerolatency`,
  one thread) on the recorder's own thread.

Both write an IDR frame every output second and mux clips to MP4 with
`+faststart`. Encoding, muxing and uploading never use the Workflow's shared thread
pools.

## Memory

Measured in validation with the x264 backend at 1080p (CRF 23):

* The buffer costs 0.82-0.94 MB per second of window per stream. It normally holds
  the pre-roll plus about one second; while a capture is pending it holds the
  pre-roll, the post-roll and about 1.5 s. Overlapping captures share packets.
* Peaks per stream: 10 MB at 5 s + 5 s, 32 MB at 30 s + 5 s, 198 MB at 120 s + 120 s.
* Annotated clips are about 1.9 times the bytes of raw clips.
* Each x264 encoder adds about 60 MB of process memory.
* Each NVENC session used 64-142 MB of VRAM.

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

When a pipeline stops, the recorder stops taking frames, drops frames not yet
encoded (counted as `queue_drops`), finishes pending clips as `partial: shutdown`
and uploads them within the host's close deadline (`WORKFLOWS_BLOCK_CLOSE_DEADLINE_SECONDS`,
default 30 s).

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
| Serverless Streaming sandbox (L4 standing in for L40S, real processor, raw and annotated recorders, NVENC) | 1x1080p30 at 30.0 fps under runc and gVisor with 2 CPUs; +4 ms decode-to-result latency, about 0.9 ms per frame on the pipeline thread | The x264 path instead of NVENC: 21 fps |

Recording cost grows with processed fps and resolution. On constrained devices set
`max_width` or lower the model fps before adding streams.

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
| `WORKFLOWS_VIDEO_CLIP_TEMP_DIR` | system temp directory | Directory clips are muxed into |
| `WORKFLOWS_VIDEO_CLIP_NVENC_POOL_FRAMES` | `8` | GPU frames queued per NVENC stream (1 to 256); a full pool counts `queue_drops` |
| `WORKFLOWS_VIDEO_CLIP_NVENC_BITRATE_KBPS` | scaled (4000 at 1080p30) | NVENC bitrate in kbit/s (100 to 200000) |
| `WORKFLOWS_BLOCK_CLOSE_DEADLINE_SECONDS` | `30` | Seconds a stopping InferencePipeline gives Workflow blocks to finish, for example to upload pending clips |

The byte, capture, concurrency and file settings are read once, when
`roboflow-workflows` is imported. The roll caps are read when a Workflow is
compiled, and the NVENC settings when a stream's encoder starts.
