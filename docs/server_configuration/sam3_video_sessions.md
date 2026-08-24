# SAM3 Video Sessions

HTTP + SSE tracking sessions at `/sam3/video/sessions`. The worker downloads
`video_url`, tracks with `sam3video`, streams overlay events, and writes
artifact chunks through Roboflow-signed GCS uploads.

These routes are independent of WebRTC. Disable WebRTC without losing SAM3
sessions, or disable SAM3 without turning off WebRTC.

## Routes

| Method | Path | Auth |
|---|---|---|
| `POST` | `/sam3/video/sessions` | API key (query, Bearer, or JSON body) |
| `GET` | `/sam3/video/sessions/{session_id}` | Session owner API key |
| `GET` | `/sam3/video/sessions/{session_id}/events` | Session owner API key (SSE) |
| `POST` | `/sam3/video/sessions/{session_id}/end` | Session owner API key |
| `POST` | `/sam3/video/sessions/{session_id}/internal/events` | API key plus session `publish_token` |

Routes register only when `SAM3_VIDEO_SESSIONS_ENABLED` is true and the
process is not Lambda / GCP serverless (the Lambda authorizer cannot carry
these paths).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SAM3_VIDEO_SESSIONS_ENABLED` | `True` | Register the `/sam3/video/sessions` surface. Separate from `WEBRTC_WORKER_ENABLED`. |
| `SAM3_VIDEO_EVENTS_CALLBACK_BASE` | unset (`http://127.0.0.1:{PORT}/`) | Server-side callback base the worker POSTs events to. Never taken from the request `Host` header. Modal workers need a reachable public base here, or a Roboflow `events_callback_base` on the create request. |
| `SAM3_VIDEO_SESSION_SILENCE_TIMEOUT_SECONDS` | `600` | SSE ends with `error` and releases the quota slot if the worker publishes nothing for this long. |
| `SAM3_VIDEO_SESSION_MAX_RETAINED_EVENTS` | `32` | Cap on replayable events kept in the shared cache. |
| `SAM3_VIDEO_SESSION_EVENT_PAGE_SIZE` | `32` | Max events returned by one `list_events` poll. |

`events_callback_base` on the create request is optional. When set, the host
must be loopback or a Roboflow app / serverless host (`*.roboflow.com`,
`*.roboflow.one`). Non-loopback hosts must use `https`.
