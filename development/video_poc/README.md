# Video Sources POC

End-to-end proof of concept for video sources as a platform primitive. This file is the
**runbook**; for the full context — what we're proving, architecture, flows, data model,
known gaps — read [HANDOFF.md](HANDOFF.md) first.

```
connector (Go, laptop/LAN)  --RTSP push-->  mediamtx  --RTSP-->  processor (Python, warm worker)
        |                                       |                        |
        |  outbound HTTP poll (commands+ack)    |  WHEP (browser         |  MJPEG + SSE events
        v                                       |  live preview)         v  (browser, via job.processorUrl)
   Roboflow app  (Video Sources page; functions: token.js + deviceApi.js; Firestore)
```

Components in this folder (branch `hansent/video-poc`):

- `connector/` — Go agent. Discovers USB cameras (avfoundation/v4l2), RTSP URLs (flags), and
  video files (a folder); polls the platform for commands; pushes streams via ffmpeg. Stdlib only.
- `processor/processor.py` — warm worker. Polls the platform for jobs, runs
  `InferencePipeline.init_with_workflow` on the assigned source in one of two modes
  (`batch` = every frame as fast as possible; `stream` = real-time with drops, files
  replayed via `ffmpeg -re` through mediamtx), splits outputs into an SSE events stream
  (images redacted to refs) and per-output MJPEG previews (`/preview.mjpeg?output=`,
  available outputs advertised in `/status`).
- `mediamtx.yml`, `fetch-deps.sh`, `bin/` — local media plane (RTSP ingest :8554, WHEP :8889).

The platform half lives in the `roboflow` repo (branch `hansent/video-sources-poc`):
Video Sources page + `/query/video-sources*` routes (token.js) + connector/processor
endpoints (deviceApi.js) + Firestore collections `video_sources`, `video_connectors`, `video_jobs`.

## Running the demo (everything local)

Terminal 0 — deps (once):
```bash
cd development/video_poc && ./fetch-deps.sh
cd ../.. && uv venv development/video_poc/.venv --python 3.11
uv pip install --python development/video_poc/.venv/bin/python -e . onnxruntime requests
cd development/video_poc/connector && go build -o ../bin/rfv-connector .
```

Terminal 1 — the app (roboflow worktree). Environment quirks discovered on 2026-07-06:
node 24 required (`n exec 24.11.0`), firebase-tools 15 comes from the repo-root node_modules,
the API-key auth path needs a local Redis on :6379 (`brew install redis`; avoids the docker
compose stack, which also wants nginx), and `build:dev:once` skips docker entirely.

```bash
redis-server --port 6379 &
cd ~/code/wt/roboflow-video-poc/app
n exec 24.11.0 npm run build:dev:once                       # one-shot webpack build
source ../secrets.sh staging gac && \
  PATH="/usr/local/n/versions/node/24.11.0/bin:$PATH" NODE_OPTIONS=--preserve-symlinks \
  ../node_modules/.bin/firebase emulators:start --only hosting,functions
# app: http://localhost:5000  ·  functions: http://localhost:5001
```

Terminal 2 — media plane:
```bash
cd development/video_poc && ./bin/mediamtx mediamtx.yml
```

Terminal 3 — warm processor. Two env vars point model downloads at staging (matching the
staging API key): `PROJECT=roboflow-staging` for the classic inference paths, and
`ROBOFLOW_ENVIRONMENT=staging` for the new `inference_models` weights provider (it has its
own config and defaults to prod — without it model loads fail with UnauthorizedModelAccessError):
```bash
cd development/video_poc/processor && \
  PROJECT=roboflow-staging ROBOFLOW_ENVIRONMENT=staging ../.venv/bin/python processor.py \
  --api-url http://localhost:5001/roboflow-staging/us-central1/light-v2-device \
  --api-key $ROBOFLOW_API_KEY
```

Terminal 4 — connector (the thing a customer runs):
```bash
cd development/video_poc && ./bin/rfv-connector \
  --api-url http://localhost:5001/roboflow-staging/us-central1/light-v2-device \
  --api-key $ROBOFLOW_API_KEY \
  --files-dir ./videos          # plus e.g. --rtsp cam1=rtsp://... ; USB cams auto-discovered
```

The connector serves a local status/config UI at **http://127.0.0.1:8070** (`--ui-addr` to
change, empty to disable): platform connection status, per-source enable/disable toggles,
add-RTSP-camera form, and API-key entry — the key can be omitted from the CLI entirely and
set in the UI instead. Runtime changes persist to `connector.json` (`--config`).

Then open the app → **Video Sources** in the sidebar:
1. Connector's cameras/files appear as sources within ~2s of it starting.
2. Click a source → live preview (WHEP) or file playback.
3. "Start processing with workflow" → pick a workflow → annotated MJPEG + JSON event stream.
4. Stop processing → the platform reconciles and tells the connector to stop pushing.

## Standalone processor test (no platform)

```bash
cd development/video_poc
./bin/mediamtx mediamtx.yml &
./bin/ffmpeg -re -f lavfi -i "testsrc2=size=1280x720:rate=30" -c:v libx264 -preset veryfast \
  -tune zerolatency -pix_fmt yuv420p -g 30 -f rtsp rtsp://127.0.0.1:8554/test &
cd processor && ../.venv/bin/python processor.py --job-file test-job-blur.json
# http://127.0.0.1:8890/status | /events | /preview.mjpeg
```

Measured on M-series laptop (720p30, blur workflow, CPU): 30 fps sustained,
~40 ms decode→result latency, ~12 s job→first-result (engine init dominates — a
warm processor amortizes imports; model load is the remaining per-job cost).

## Design notes

- **Control channel**: outbound-only polling with commands in the healthcheck response and
  ack-by-id — the same pattern as `device-healthcheck-v2` / device-manager.
- **Stream lifecycle**: the platform reconciles desired state (active jobs + preview TTLs)
  against connector-reported streams on every healthcheck — start/stop commands are emitted
  from the diff. Registered ≠ streaming: video flows only while something needs it.
- **Events vs pixels**: the processor redacts image outputs from the JSON events (SSE) and
  serves any image output as MJPEG on demand (`?output=`); `/status` advertises what exists
  so viewers can attach to a job after it started.
- **Two processing modes**: `batch` (files as they are: every frame, in order, faster than
  real time is the goal) vs `stream` (real-time pacing, drops under load — cameras, or files
  standing in for cameras). Details and the `is_file` gotcha that motivated it: HANDOFF.md §5.
- **Media plane**: mediamtx is one static binary doing RTSP ingest + WHEP browser preview;
  the processor consumes plain RTSP from it. This is the seam where the real relay/cell
  architecture (see the video strategy deck) slots in.

An interactive architecture diagram with scenario walkthroughs (for demos) lives at
[architecture.html](architecture.html) — open it directly in a browser.
