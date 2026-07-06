# Video Sources POC — Handoff

This is the full context document for the video-sources proof of concept. It exists so
that anyone (human or agent) can pick up the work without access to prior conversations.
The [README](README.md) is the runbook (how to build and run everything locally); this
document is the *why and how*: what we're doing, what we're trying to prove, how it's
implemented, and how the pieces fit together.

The POC spans two repos:

| Repo | Branch | Draft PR | What lives there |
|---|---|---|---|
| `roboflow/inference` | `hansent/video-poc` | [#2616](https://github.com/roboflow/inference/pull/2616) | `development/video_poc/` — Go connector agent, Python processor (warm worker), local media plane (mediamtx), this doc |
| `roboflow/roboflow` | `hansent/video-sources-poc` | [#13264](https://github.com/roboflow/roboflow/pull/13264) | Video Sources page (React), `/query/video-sources*` routes, connector/processor API endpoints, Firestore DAOs |

There is also an internal video strategy deck that motivates all of this — the POC
deliberately implements the shapes recommended there:
https://rising-denim-n8yv.here.now/ (password-protected; ask Thomas for access).

---

## 1. What we're doing

Roboflow is making a bet on video. Today the platform is image-first: video support is
fragmented across a WebRTC preview path, a batch video API, and edge deployments, each
with its own way of getting pixels in. The bet is that **"video source" should be a
platform primitive**: a named, addressable object (an uploaded file, a pushed stream, a
camera behind a customer's firewall, an edge device) that a user registers once and can
then *do things to* — preview it, run a workflow on it live, process it as a batch job,
eventually record it.

This POC builds a thin end-to-end slice of that: sources get registered (uploaded or
auto-discovered by an on-prem agent), listed in the app, previewed in the browser, and
processed by a warm worker running Roboflow Workflows, with results streaming back to
the UI as JSON + annotated video.

## 2. What we're proving

Each of these was a specific open question; the POC answers all of them concretely:

1. **The firewall problem is solvable with an outbound-only agent.** Cameras (RTSP/USB)
   sit on networks we can't dial into. The connector agent makes only outbound
   connections: HTTP polling for control, RTSP push for media. No inbound ports, no
   VPN, no firewall changes. ✅ works
2. **Registered ≠ streaming.** A source is a *record*; video flows only while something
   needs it (an active job or a preview with a TTL). The platform reconciles desired
   state against connector-reported state on every healthcheck and issues start/stop
   commands from the diff. ✅ works
3. **A warm worker kills the cold-start problem.** The current hosted video path spins
   infrastructure per request (minutes). A pre-provisioned processor that polls for
   jobs starts producing results in seconds — measured ~12s job→first-result on a
   laptop, dominated by engine init that a long-lived worker amortizes; model load is
   the remaining per-job cost. ✅ works
4. **Events and pixels must be split at the source.** Workflow outputs contain both
   structured data and images. Base64 images never ride the JSON events channel — the
   processor redacts them to refs and serves pixels on a separate on-demand channel.
   ✅ works
5. **Files and streams are different processing modes, and both are needed** — see §5.
   ✅ works
6. **A viewer can attach to a running job dynamically** — discover what image outputs
   exist and watch any of them, without restarting the job. ✅ works (locally; see §8
   for the production-shape caveat)

## 3. The plan (where this fits)

The strategy is phased:

- **Phase 1 — sources + warm pool.** Video sources as records, file upload path, warm
  processing capacity. No new protocol decisions needed.
- **Phase 2 — connector agent + live monitoring GA.** Agent discovery/push, relay
  (ingest fan-out), continuous workflows on live streams.
- **Phase 3 — recording + scale.** Record-then-process, storage lifecycle, multi-cell
  scale-out.

The POC cuts across phases 1 and 2 to de-risk the end-to-end shape before any piece is
productionized.

## 4. System map

```
 customer network                          local "cloud"                        browser
┌──────────────────────┐      ┌──────────────────────────────────┐      ┌────────────────────┐
│  USB cams  RTSP cams │      │  mediamtx (media plane)          │      │  Video Sources page│
│      \      /        │      │   :8554 RTSP in/out              │      │                    │
│   connector (Go)  ───┼──────┼─▶ :8889 WHEP  ────────────────────┼──────▶ live preview      │
│   · discovers        │ RTSP │                                  │      │                    │
│   · pushes on demand │ push │  processor (Python, warm)        │      │                    │
│   · polls commands ──┼──┐   │   · polls for jobs               │      │                    │
│   · local UI :8070   │  │   │   · InferencePipeline + workflow │      │                    │
└──────────────────────┘  │   │   · :8890 /status /events        │      │                    │
                          │   │            /preview.mjpeg ────────┼──────▶ results (SSE+MJPEG)│
                          │   └────────────▲─────────────────────┘      └─────────┬──────────┘
                          │                │ poll/claim/status                    │ /query/* (session)
                          ▼                │ (API key)                            ▼
                    ┌─────────────────────────────────────────────────────────────────┐
                    │  Roboflow app (firebase functions + Firestore)                  │
                    │  token.js: UI routes   deviceApi.js: connector+processor routes │
                    │  collections: video_sources · video_connectors · video_jobs     │
                    └─────────────────────────────────────────────────────────────────┘
```

Everything runs locally in the POC (see README for the 4-terminal setup). In production
the media plane + processors become the "cell" (relay + warm pool colocated on GPU
infra) and the connector stays exactly as it is — that's the point of proving it with
outbound-only connections now.

## 5. The two processing modes (important)

A video file and a live stream are **fundamentally different jobs**, and the platform
treats them as such via `job.mode`:

- **`batch`** — *process the file as it actually is*: every frame, in order, as fast as
  inference can go. Output is complete and deterministic; "faster than real time" is
  the goal, not a bug. Implemented by passing explicit buffer strategies
  (`WAIT` filling + `LAZY` consumption) to `InferencePipeline`.
- **`stream`** — *real-time semantics*: the pipeline keeps up with the clock and drops
  frames when inference falls behind (`ADAPTIVE_DROP_OLDEST`), keeping latency bounded
  at ~one inference time. This is the only sane mode for cameras, and it's also
  offered for files ("simulate a live camera") so a recording can stand in for a
  camera that will be hooked up later — the processor replays the file at native speed
  through the local relay with `ffmpeg -re` and consumes it back as RTSP, so the
  pipeline sees a genuine live stream.

Mode selection: connector sources are always `stream`; uploaded files default to
`batch` with a UI radio to choose the simulation mode.

Note that **a file behind the connector is not a file to the platform**: the
connector can only replay it as looping real-time RTSP (`ffmpeg -re -stream_loop
-1`) — a test stand-in for a camera. It is labeled "Video File (test stream)" in
the UI. Only uploaded files (a URL the processor can read directly) support batch
processing; letting the connector *transfer* a file for batch is future work.

**Batch results are recorded and scrubbable.** During a batch job the processor
writes the designated image output to an H.264 mp4 (ffmpeg image2pipe at the
source's declared fps) and one JSON line per frame. Because batch processes every
frame in order, mp4 frame k, JSONL line k, and playhead time k/fps are the same
source frame — so the UI can serve a seekable annotated video with the JSON
result aligned to the playhead. When the file ends, the processor finalizes the
recording, reports `completed`, frees itself for the next job, and keeps serving
results at `/results/<jobId>/{video.mp4,frames.jsonl,meta.json}` (mp4 with HTTP
Range support — that is what makes browser scrubbing work). Results live in the
processor's temp dir in the POC; the production shape is object storage.

**Gotcha that motivated all this:** `VideoSource.discover_source_properties` in
inference classifies a source as a file only if `os.path.exists(ref)` is true. A signed
GCS URL fails that check, so without explicit strategies the pipeline treats an
uploaded file as a live stream: decode at network speed, drop frames under load. That's
why early tests showed output "faster than the input" *with silent frame drops* —
neither mode done properly. The explicit strategies fix it.

## 6. How each flow works

### Source registration
Two births for a source record:
- **User-created**: upload a video → `POST /query/video-sources` with the upload id.
- **Agent-discovered**: the connector healthchecks every ~2s with its source roster
  (USB via avfoundation/v4l2, RTSP from flags/UI, files from a watched folder). The
  platform upserts sources by `(connectorId, localId)`. Connector identity defaults to
  `conn-<hostname>` — same machine re-registers as itself; a different machine is a
  different connector with new sources.

Sources never expire on their own; a connector going away leaves its sources listed as
`offline` (status is computed from the connector's `lastSeen`, 15s window). The UI has
a per-row **Remove** (refused while a job is active; a source the connector still
reports will re-register on the next healthcheck — use the connector UI's disable list
for that case).

### Preview
- Uploaded file → signed GCS playback URL, plain `<video>`.
- Connector source → the app stamps `previewRequestedUntil = now + 5min` on the source;
  the reconciler tells the connector to start pushing RTSP to mediamtx; browser watches
  WHEP. The modal re-requests every 60s to keep the TTL warm; when it lapses, the
  stream is torn down. Video flows only while watched.

### Job lifecycle
```
UI: POST /query/video-sources/:id/jobs {workflowUrl, imageOutput?, mode?}
        → job doc {state: queued}
processor: POST /video-jobs/claim {processorId, processorUrl}   (every 2s while idle)
        → platform resolves source URL (signed URL | RTSP path) + workflow spec
          (Firestore workflow config), returns {job}, state → claimed
processor: starts InferencePipeline, state → running
processor: POST /video-jobs/:id/status {state, stats}           (every 2s while busy)
        → response may carry {cancel: true} — the platform's ONLY signal path to a
          running processor is piggybacked on this poll (this is by design, see §8)
UI: POST /query/video-jobs/:id/cancel → sets cancelRequested; if the processor has
    not reported for 15s it is presumed dead and the job is cancelled directly.
```

Orphan handling: `heartbeatAt` is written ONLY by the processor's own calls (claim +
status) — never by cancel, which would make a dead processor look alive. Jobs in
claimed/running whose heartbeat is >30s old are lazily reset to `error` on read
(`listJobs`), so a crashed/killed processor cannot leave the UI stuck on
"processing". A reaped job is terminal: a zombie processor posting status for it
gets `{cancel: true}` back instead of resurrecting it.

### Results path (events vs pixels)
The processor's per-frame sink:
- decodes and stores the latest JPEG for **every** serialized image output (cheap: the
  pipeline serializer already JPEG-encodes; storing all outputs is one b64decode each,
  and it's what makes late attachment work),
- publishes the frame's outputs to SSE subscribers with images **redacted to
  `{type: "image_ref", output}` markers**,
- `/status` advertises `imageOutputs` + `defaultImageOutput`.

The UI shows: the latest event as pretty-printed JSON updated in place (keys are
re-rendered from each event — do not assume they're stable across a job), an MJPEG
`<img>` on `/preview.mjpeg?output=<name>`, and a dropdown to switch outputs live, fed
by polling `/status`.

### Workflow output selection
Before a job starts, the app parses the chosen workflow server-side —
`resolveWorkflowSpecification` (Firestore) + serverless `POST
/workflows/describe_interface` (via `describeInterfaceBySpec` in the inference
adapter) — and the UI offers a dropdown of outputs whose kind includes `image`.
Free-text output names are gone: a mistyped name used to mean a silently empty preview.

## 7. Data model (Firestore, all POC-new collections)

- **`video_sources`**: `{workspace, kind: file|usb|rtsp, name, connectorId?, localId?,
  videoUploadId?, previewRequestedUntil?, created_at, lastSeen}`. Status
  (`connected|offline|ready`) is computed at read time, not stored.
- **`video_connectors`**: `{workspace, name, hostname, platform, lastSeen, streams[]}`
  plus a commands queue (`start_stream` / `stop_stream`, drained on delivery,
  ack-by-id) — same contract shape as device-manager healthchecks.
- **`video_jobs`**: `{workspace, sourceId, sourceName, workflowUrl|workflowSpecification,
  imageOutput?, mode: batch|stream, state: queued|claimed|running|completed|error|cancelled,
  cancelRequested, processorId?, processorUrl?, stats?, created_at, updated_at}`.

## 8. Known gaps and how they're meant to close

- **Browser→processor is direct HTTP** (`job.processorUrl`, localhost). Fine for the
  POC; wrong shape for production (NAT, TLS, authz). The designed replacement, worked
  out but not yet built: results-video goes out via the processor **publishing the
  annotated stream to the relay** (it becomes just another stream name, e.g.
  `job-<id>-<output>`, watched over WHEP like any source preview), and *wanting to
  watch* is signaled through the existing status-poll channel — the UI stamps a
  `watchRequestedUntil` TTL on the job, the processor sees it within 2s and starts
  publishing, and stops when the TTL lapses. Identical pattern to source preview TTLs;
  requires no new connection into the processor. Result video should not stream when
  nobody is watching.
- **Batch results are processor-local**: the annotated mp4 + JSONL live in the
  processor's temp dir and die with the machine. Production shape: upload to object
  storage on finalize and serve from there (also unblocks reviewing results after
  the processor is reassigned).
- **Processor HTTP has no auth** and CORS `*` (localhost POC only).
- **Single-workspace claim**: processors claim jobs only for the workspace of their API
  key. Real warm pools need cross-tenant scheduling, leases/heartbeats on claims, and
  model-affinity placement.
- **No recording** (phase 3 by design).
- **No metering** (the intended model: stream-hours + GPU-seconds).
- **`imageOutput` list outputs** (arrays of images) are redacted from events but not
  stored/previewable.

## 9. Where to pick up

1. Read the README for the local runbook (node 24 / redis / staging env quirks are
   documented there — they are real and will bite you).
2. The task list that produced this: dev tooling ✅, backend API ✅, connector ✅,
   processor ✅, frontend ✅, e2e demo verified through "workflow on uploaded file with
   live annotated preview" ✅.
3. Natural next moves, in rough order of value: batch results to object storage
   (§8); the relay-published results stream + `watchRequestedUntil` signaling (§8);
   connector-source e2e polish (USB/RTSP path got less testing than files); claim
   leases.
