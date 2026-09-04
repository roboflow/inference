# Video Sources POC — Handoff

This is the full context document for the video-sources proof of concept. It exists so
that anyone (human or agent) can pick up the work without access to prior conversations.
The [README](README.md) is the runbook (how to build and run everything locally); this
document is the *why and how*: what we're doing, what we're trying to prove, how it's
implemented, and how the pieces fit together. The next scaling phase is specified in
[MULTI_CELL_SCALING_RFC.md](MULTI_CELL_SCALING_RFC.md); material decisions made there
must also be reflected here.

The system now spans four components:

| Component | Original work | What lives there |
|---|---|---|
| `roboflow/inference` | Draft [#2616](https://github.com/roboflow/inference/pull/2616) | `development/video_poc/` — Python processor, local MediaMTX config, harnesses, and architecture docs |
| `roboflow/roboflow` | [#13264](https://github.com/roboflow/roboflow/pull/13264) plus production hardening follow-ups | Video Sources UI, `/query/video-sources*` routes, connector/processor APIs, Firestore control plane, relay auth, and result URLs |
| `roboflow/roboflow-infra` | [#2290](https://github.com/roboflow/roboflow-infra/pull/2290) plus environment follow-ups | Terraform and `helm/roboflow-video-proc/`: MediaMTX, ready GPU/CPU pools, gateway, Pub/Sub subscriptions, secrets, and monitoring |
| `roboflow/rf-video-connector` | Extracted from the original POC | Released Go connector binary with bundled ffmpeg, discovery, local UI, and the outbound command/media clients |

**Deployment status (2026-08-10):** staging and the first feature-flagged production
cell (`crusoe-use1` on Crusoe US East) are live. The production cell runs one
MediaMTX relay and ready GPU/CPU pools configured for up to four concurrent jobs per
worker. Production deployment mechanics now live in the roboflow-infra chart/stack;
[DEPLOY_PLAN_STAGING.md](DEPLOY_PLAN_STAGING.md) is retained for the rationale and
history of the first deployment. Draft infra
[#2443](https://github.com/roboflow/roboflow-infra/pull/2443) adds internal
MediaMTX metrics scraping and the first relay dashboard; it is applied on
staging. Production rollout remains separately approved and controlled.

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
  scale-out, and workload-aware processor allocation. The multi-cell/capacity work is
  now tracked in [MULTI_CELL_SCALING_RFC.md](MULTI_CELL_SCALING_RFC.md).

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

### The media plane (mediamtx)

The relay is [mediamtx](https://github.com/bluenviron/mediamtx) — one static binary,
fetched by `fetch-deps.sh` (latest release, **unpinned** — pin before anyone else
depends on this) and run as a separate process: `./bin/mediamtx mediamtx.yml`. It is
*almost* vanilla; every deviation in `mediamtx.yml` is deliberate:

| Setting | Value | Why |
|---|---|---|
| `rtspTransports` | `[tcp]` | force TCP: predictable through localhost/firewalls, no UDP port-range headaches; both the connector (push) and processor (read) use it |
| `webrtc` / `webrtcAddress` | `:8889` | WHEP endpoint the browser preview reads (`/<stream>/whep`) |
| `webrtcLocalUDPAddress`, `webrtcAdditionalHosts` | `:8189`, `[127.0.0.1, localhost]` | makes browser WebRTC work on a laptop with no STUN/public IP |
| `hls`, `rtmp`, `srt` | `no` | narrow the surface to exactly what the POC uses: RTSP in, RTSP+WHEP out |
| `api` | `127.0.0.1:9997` | debugging only (`curl localhost:9997/v3/paths/list` shows active streams); nothing in the code depends on it |
| `paths: all_others` | catch-all | any stream name can be published/read. In THIS repo's local-dev config there is **no auth** (fine on a laptop); the staging chart's ConfigMap adds `authMethod: http` pointing at the platform's `/video-relay/auth` hook, which validates per-stream keys on every connection — that's where the deck's ingest-URL + stream-key design landed |

Stream naming conventions (both sides must agree; defined in
`videoSourcesService.js` and `processor.py`):
- `src-<sourceId>` — connector-pushed source streams (platform tells the connector
  the full ingest URL in the `start_stream` command)
- `sim-<jobId>` — the processor's own file replay for simulate-a-camera jobs

The deployed chart pins MediaMTX, validates per-stream credentials through the
platform auth hook, and colocates the relay with the warm pools as a cell. Remaining
production-scale work includes secure WAN ingest/consume options and MediaMTX/cell
capacity metrics; aggregate ingress/egress and reader fan-out are cell sizing inputs.

## 5. The two processing modes (important)

A video file and a live stream are **fundamentally different jobs**, and the platform
treats them as such via `job.mode`:

- **`batch`** — *process the file as it actually is*: every frame, in order, as fast as
  inference can go. Output is complete and deterministic; "faster than real time" is
  the goal, not a bug. Implemented by **downloading the file to processor-local disk
  first** (so `VideoSource` gets true file semantics), plus explicit buffer strategies
  (`WAIT` filling + `LAZY` consumption) as belt-and-braces.
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
GCS URL fails that check, with two consequences: stream buffer strategies (decode at
network speed, drop frames under load — early tests showed output "faster than the
input" with silent drops), and worse, **stream reconnection at EOF** — the pipeline
treats end-of-file as a dropped stream, reconnects to the URL, and replays the file
forever, so the job never completes. Downloading to a local path fixes both at the
root; the explicit strategies stay as a guard.

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
claimed/running whose heartbeat is >30s old are lazily **requeued** on read
(`listJobs`): processor assignment cleared, `attempts` bumped, a fresh Pub/Sub
wake-up published — so a crashed/killed/evicted processor is a seconds-long blip,
not a stuck UI. After 3 lost processors the job goes to terminal `error` (poison
cap). A zombie processor posting status for a requeued or terminal job gets
`{cancel: true}` back instead of resurrecting it.

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

### Consuming results (the four cases)

1. **Live JSON** — the SSE `/events` stream described above. Locally the browser
   connects to the processor directly; in the cluster it goes through the
   processor-gateway (`https://video-processors…/{worker}/events` — nginx with
   `proxy_buffering off`, which is load-bearing: SSE dies silently behind a
   buffering proxy). The worker reports its gateway URL as `processorUrl` at claim
   time, so consumers never notice the indirection. **Known limitation:** this
   subscribes to a *worker*, not a *job* — if the job is re-placed, the consumer
   must re-read the job doc and reconnect. Fine for the interactive UI; wrong for
   programmatic consumers. Production contract: `GET /video-jobs/{id}/events`, a
   platform-authenticated **job-addressed** stream. Two implementations, in order
   of effort: (a) a smart proxy in the cell that resolves jobId→current worker and
   re-attaches upstream on worker death (do NOT put this on Firebase functions —
   long-lived streaming responses fight the platform); (b) real fan-out: the
   processor *publishes* events to a stream keyed by jobId (Redis stream / pub-sub)
   and an edge service serves N subscribers with last-event-id replay — the events
   channel getting its own "mediamtx". Bandwidth is a non-issue for JSON
   (~1–2KB/frame, orders of magnitude under the video) — it only becomes one if
   images sneak into the JSON, which the redaction rule exists to prevent.
2. **Notifications on events** — Workflows sinks inside the pipeline. Nothing new.
3. **Warehouse/storage sinks** — same: Workflows blocks.
4. **Process now, download later** — batch already works this way (JSONL + mp4 to
   GCS on completion, signed-URL retrieval, frame-aligned for scrubbing). For
   continuous streams the same shape extends: roll the event stream into
   hour-partitioned JSONL in object storage with lifecycle policies, later aligned
   with recorded video segments (phase 3) so "everything from 2–4pm" is one call.

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
  imageOutput?, mode: batch|stream, streamKey (relay credential for sim-<jobId>,
  never sent to browsers), processorAccessToken (processor HTTP credential,
  returned only by an authorized job-access route),
  state: queued|claimed|running|completed|error|cancelled,
  attempts (requeue counter, capped at 3), cancelRequested, processorId?,
  processorUrl?, heartbeatAt?, stats?, resultsFiles?/resultsUploadedAt? (GCS),
  error?, created_at, updated_at}`. The claim payload derives from this doc plus
  `sourceUrl` (signed GCS URL or credentialed RTSP consume URL), `apiKey` (the
  job workspace's inference key), and `simPublishUrl`.

The current schema does not persist a source or job cell. Proposed `homeCell`,
`relayShard`, `executionCell`, placement-generation, and workload-profile fields
are specified in [MULTI_CELL_SCALING_RFC.md](MULTI_CELL_SCALING_RFC.md); they are
not implemented yet.

## 8. Known gaps and how they're meant to close

- **Browser→processor addressing is solved (gateway), but coupled to workers.**
  Locally `job.processorUrl` is localhost; in the cluster it's the processor-gateway
  path to a specific worker. The remaining production work on the JSON/event side is
  *job-addressing* (`/video-jobs/{id}/events` — see §6 "Consuming results"); processor
  HTTP auth is closed below.
  For the video half this is **CLOSED**: the processor **publishes the annotated
  stream to the relay** (`out-<jobId>`, credentialed with the job's stream key,
  watched over WHEP like any source preview), and *wanting to watch* is signaled
  through the existing status-poll channel — `POST /video-jobs/{id}/watch` stamps
  a 60s `watchRequestedUntil` TTL (+ desired output; the UI renews every 30s),
  the processor sees it within 2s, publishes (in-process aiortc→WHIP by
  default, event-driven at native fps; ffmpeg-RTSP as fallback/hot-swap),
  restarts on output switches, and stops when the TTL lapses.
  Identical pattern to source preview TTLs; no new connection into the
  processor; result video never streams unwatched. The processor's MJPEG
  endpoint remains for debugging only. (Historical local-dev gotcha it also
  killed: https app pages refuse `<img>` streams from IP-literal insecure
  hosts, so the MJPEG preview silently never painted on
  `https://localapp.roboflow.one`.)
- ~~Batch results are processor-local~~ **CLOSED**: on completion the processor
  uploads mp4/JSONL/meta to GCS via platform-signed PUT URLs and the review UI
  reads platform-signed GET URLs; processor-local files remain only as fallback.
- ~~Relay is unauthenticated~~ **CLOSED**: per-stream keys (per source, per job)
  minted by the platform, embedded in every issued URL, validated per connection
  by mediamtx's external-auth hook → `POST /video-relay/auth`. No shared secrets
  are pushed to connectors.
- ~~Managed-pool workers hold a workspace key / claim is a privilege step-up~~
  **CLOSED for the fleet**: pool workers authenticate to claim/status/results
  with a service secret (`x-video-proc-service-access-token`, same pattern as
  batch-processing's), claim **across all workspaces**, and hold no tenant
  credentials at rest — the claim payload still carries the JOB's workspace key
  so per-tenant execution is unchanged; ownership on status/results is the
  processor identity. Terraform mints the secret (k8s + Secret Manager;
  functions bind it as a runtime secret). Self-hosted processors keep
  workspace-key auth and its workspace scoping — per-job scoped tokens remain
  the eventual hardening there. The fleet secret is the crown jewel: never
  user-facing.
- ~~Processor HTTP endpoints are unauthenticated~~ **CLOSED**: the platform mints a
  high-entropy `processorAccessToken`, returns it only through authorized job-access
  routes and claim payloads, and redacts it from general job serializers. The worker
  removes it from the retained workflow payload and checks it per job on `/events`,
  `/events/poll`, `/status`, `/preview.mjpeg`, and `/results`. Fetch clients should
  send `Authorization: Bearer <token>`; native `<img>`/`<video>` consumers may use
  `?access_token=<token>` (responses set `Referrer-Policy: no-referrer`). `/metrics`
  and bare `/status` remain unauthenticated for scraping/readiness, but bare
  `/status` returns only aggregate counts in managed mode. Managed workers default
  token enforcement on when `VIDEO_PROC_SERVICE_SECRET` is present; explicit
  `REQUIRE_JOB_ACCESS_TOKEN=true|false` overrides this for rollout/local testing.
  The job-addressed platform events endpoint (§6) remains the preferred long-term
  programmatic surface because the platform authenticates at its own front door.
- **Nothing behind platform Traefik can stream** (staging deploy finding,
  2026-07-08): `crusoe/addons/traefik.tf` attaches the `buffering` middleware
  (`body-size-limit`, a 100MB request cap) to the whole websecure entrypoint,
  and Traefik's buffering holds RESPONSES until completion too — an infinite
  SSE/MJPEG body is withheld forever (verified: gateway nginx sent 110KB, the
  client got 0). Same pattern exists in the GKE Traefik configs. POC
  workaround SHIPPED: the UI consumes worker events via cursor-based
  long-polling (`GET /events/poll?cursor=N` → finite `{cursor, events[]}`
  responses pass any proxy); the SSE endpoint remains for direct/local
  consumers. **Chosen platform fix (do when hardening): replace the
  `buffering` middleware with a streaming request-limit Traefik plugin** —
  reject on Content-Length when present, else abort via a counting body
  reader past the cap; request protection is identical, the response path is
  untouched, and the per-request 20MB buffer memory goes away. Precedent for
  the packaging exists (`github.com/roboflow/traefik-req-logger` local
  plugin); rollout is a one-line swap of the entrypoint middleware list.
  WebSockets are unaffected either way (Upgrade hijacks the connection).
- ~~Orphan reaping goes to `error`~~ **CLOSED**: the reaper requeues (attempts-capped
  at 3) and re-publishes the Pub/Sub wake-up, so crashes / node reclaims / evictions
  re-place the job on a fresh worker in seconds — the prerequisite the ready-pool
  model (§9) needed, now implemented alongside it.
- ~~Output preview lags the source by ~600ms~~ **CLOSED — the standing
  latency was ffmpeg's h264 decoder reorder buffer, and no cv2 option can fix
  it.** Decomposed with `processor/latency_harness.py` (pixel-clock stream: 32
  bars encode wallclock ms; `publish` / `probe` / `probe-ffmpeg` modes, plus
  `latency_harness_whep.py` for the WebRTC leg). Measured on the same relay
  stream: cv2.VideoCapture 586ms; every documented low-latency capture option
  (nobuffer, low_delay, max_delay 0, probesize 32, threads 1) still ≥585ms;
  ffmpeg CLI default 703ms; ffmpeg CLI with `-flags low_delay` 81-121ms; WHEP
  40ms. Root cause: the decoder holds a DPB-sized (~16-frame ≈ 530ms at 30fps)
  frame-reorder buffer unless `AV_CODEC_FLAG_LOW_DELAY` is set on the *codec*
  context — `OPENCV_FFMPEG_CAPTURE_OPTIONS` only reaches the *format* context,
  so the flag is unreachable from cv2. Fix: stream-mode jobs ingest through
  `LowLatencyRtspProducer` (PyAV — already a dependency via aiortc — sets
  low_delay + single-threaded decode) plugged into `VideoSource`'s
  producer-factory path; no inference-repo changes needed. Verified
  glass-to-glass source→annotated-output: **~20-50ms** (was ~600). Job-level
  `captureOptions` become libavformat open options for stream mode; batch keeps
  the cv2 path (throughput over latency).
- **Claims and media URLs are cell-unaware.** Fleet processors now claim across
  workspaces, but the Firestore claim query filters by state and CPU/GPU tier, not
  cell. Functions also construct ingest/consume/WHEP URLs from one global
  `VIDEO_PROC_*` configuration. A second cell is unsafe until source placement,
  job placement, URL resolution, and transactional claim filtering all agree on a
  persisted cell. See [MULTI_CELL_SCALING_RFC.md](MULTI_CELL_SCALING_RFC.md).
- **Multi-stream admission is count-only.** Production allows up to four jobs in one
  worker and biases claims toward partially filled workers so model loads can be
  shared. This works for the tested simple workflows, but there is no workflow-cost
  estimate, multidimensional resource budget, model-affinity scheduler, or workspace
  fairness. Relay and workflow capacity benchmarks must establish safe workload
  classes before raising concurrency or admitting heavy mixes.
- **Relay and cell capacity are not characterized.** Infra
  [#2443](https://github.com/roboflow/roboflow-infra/pull/2443) is applied on
  staging with the internal-only MediaMTX scrape and first relay dashboard (the
  approved PR still needs to be merged). The draft inference PR adds a
  reproducible relay harness, provisional workflow corpus, bounded aggregate
  processor metrics, and a staging-only service-API corpus runner.
  These are measurement tools, not certified limits. Public LB, relay-node,
  CNI/east-west, WHEP, and processor bandwidth still need independent capacity
  curves before choosing a relay shard size. Controlled pprof remains disabled.
- **Processor terminal metrics need a retirement scrape window.** Detached pool
  workers expose `video_processor_retiring=1` and remain alive for 35 seconds
  after their final job outcome so the default 15-second Prometheus interval can
  capture process-local `video_processor_jobs_finished_total` counters before
  pod self-deletion. `PROCESSOR_FINAL_METRICS_GRACE_S=0` restores immediate
  retirement; changing the scrape interval should preserve at least two scrape
  opportunities.
- **Staging measurement path is live (2026-08-12).** GPU and CPU pools run image
  `2e4a97ee5`; MediaMTX and both processor PodMonitor targets report `up=1`.
  Prometheus captured a cancelled stream-job terminal increment before the CPU
  worker retired, and the ready pool replaced that worker after its metrics grace
  window. [`benchmarks/run_api_workflow_corpus.py`](benchmarks/run_api_workflow_corpus.py)
  now drives list/start/poll/cancel without the UI, defaults to dry-run, refuses
  production hosts, and writes credential-free JSON reports. The first valid
  uploaded-file smoke (`cpu-blur`, CPU, concurrency 1, output disabled) ran the
  full 60-second window with zero retries, 1,083 API-reported frames, 4.03-second
  pipeline start, 6.54-second first result, and clean cancellation. A prior
  connector-local file attempt correctly failed with a structured redacted 404;
  prefer an uploaded `ready` fixture unless the connector file is revalidated.
  The original provisional `yolov8n-*` aliases also resolved as nonexistent
  workspace resources; the corpus now uses the public model IDs already proven
  in this staging workspace. The corrected single-detection GPU profile passed
  60-second runs at concurrency 1, 2, and 4. The 2- and 4-job runs packed every
  job onto one worker, had zero retries, cancelled cleanly, and at concurrency 4
  reported final per-job decode-to-result latency between 10.8 and 13.3 ms.
  A follow-up unbounded-file saturation series raised the staging-only admission
  ceiling to 24 and tested concurrency 5, 6, and 8 after the passing c4 point.
  All runs packed onto one L40S worker and cleaned up successfully, but every
  point above four failed at least throughput-retention and fairness gates. At
  c5, aggregate FPS reached 94.411 but the slowest stream retained only 82.8%
  of c1 throughput and cohort spread reached 13.4%; sampled latency p95 remained
  37.9 ms. C6 delivered 75.868 FPS with 10.6% spread and 88.4 ms worst sampled
  latency; c8 delivered 55.464 FPS with 41.5% spread and 104.3 ms latency. GPU
  utilization remained low: c5 averaged 7.89% (15% max), c6 7.05% (10% max),
  and c8 4.63% (8% max). This is evidence of a current worker/runtime scheduling
  or CPU-side knee, not an L40S compute knee. The 24-job value remains a
  benchmark admission ceiling, not certified capacity; further curves must use
  API-enforced source FPS and the new per-job counters/histogram.
  A model-for-model CPU curve used the exact same immutable
  `microsoft-coco-obj-det/8` workflow specification with output disabled. C1
  passed at 9.417 delivered FPS and 39.12 ms sampled latency p95; c2 passed at
  18.570 aggregate FPS, 2.9% spread, and 43.38 ms. C3 retained 97.6% or better
  of c1 throughput per stream and had only 3.5% spread at 27.403 aggregate FPS,
  but narrowly failed the strict 50 ms latency gate at 55.72 ms. C4 reached the
  CPU limit and failed clearly: 30.842 aggregate FPS, 37.8% spread, and 190.58
  ms latency. Prometheus container CPU averaged 1.287 cores at c1, 2.360 at c2,
  and 4.953 at c4, where it hit the 8-core limit. Therefore two streams is the
  current strict-SLO CPU packing point, three is a viable approximately-9-FPS
  throughput tier if the latency budget is relaxed slightly, and four is beyond
  the safe boundary. This is the requested simple YOLO comparison: the staging
  pretrained-model catalog identifies `microsoft-coco-obj-det/8` as YOLOv8 Nano
  640x640, and both tiers use the identical workflow/model. Do not replace that
  staging canonical ID with the generic `yolov8n-640` alias: the current alias
  table resolves it to the production-era `coco/3` resource, which is absent in
  staging. Keep environment-specific canonical IDs explicit in future matrices.
  The first API-enforced `maxFps=5` GPU matrix then ran two repetitions at
  c1/c4/c8/c12/c18. It delivered only about 2.46 FPS even at c1, so none of
  those runs certifies a 5-FPS target. C4 stayed at 9.706/9.838 aggregate FPS
  with 20 ms p95, c8 at 19.694/19.824 FPS with 35 ms, and c12 at
  31.632/32.094 FPS with 150 ms. C18 regressed to 29.412/22.534 FPS at
  250/500 ms, and one repetition lost a pipeline after its frame counter
  stalled for about 35 seconds. Fairness remained strong through successful
  runs, but the current operational latency knee is between c8 and c12 and c18
  is unstable. Treat c1/c4/c8, with conditional c12, as the legacy A/B set for
  a new-manager worker; investigate the roughly-half-rate `maxFps`
  under-delivery separately before certifying target-FPS capacity.
  The c1 counter deltas localized that under-delivery before model execution:
  the worker captured about 59.6 FPS, consumed about 31.2, explicitly dropped
  about 24.8, and inferred only 2.46 while frame-latency p95 stayed 20 ms. The
  uploaded `traffic.mp4` replay—not the connector—fed those runs. Inference
  v1.4 already contains the targeted demand-driven
  `VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE` fix for starvation in the legacy
  open-loop estimator. The POC branch is now based on v1.4 and adds an explicit
  `PROCESSOR_VIDEO_INGEST_MODE=pyav|gstreamer_cuda` A/B: PyAV remains the
  default; CUDA mode directly constructs the GStreamer/NVDEC tensor producer,
  refuses host-frame fallback, and reports producer/bridge identity. Both
  variants use `freshest`, `DROP_OLDEST`, and a one-frame decoding queue so the
  comparison does not confound decoder choice with buffering. Tensor-aware
  result serialization is selected at process start, and image outputs retain
  `WorkflowImageData` until an actual MJPEG/RTSP/WHIP/batch sink requires a host
  image. Run the c1 unbounded/5/10/15 gate twice per variant and require at
  least 90% target attainment before restarting any concurrency curve.
  The API harness now also supports credential-separated multi-workspace waves,
  delayed arrivals, same-worker fairness assertions, compact atomic recovery
  checkpoints, SIGINT/SIGTERM cleanup, and staging-only exact-run janitors. A
  matrix digest binds cleanup to the original routing configuration, so editing
  a workspace mapping cannot retarget an old job ID. See
  [`benchmarks/MULTI_WORKSPACE_FAIRNESS.md`](benchmarks/MULTI_WORKSPACE_FAIRNESS.md)
  and [`benchmarks/UNATTENDED_RECOVERY.md`](benchmarks/UNATTENDED_RECOVERY.md).
- **Process isolation is an experiment, not a current guarantee.** The worker
  has a default-off staging mode, `PROCESSOR_EXECUTION_DOMAIN_MODE=workspace_probe`,
  that starts one empty lifecycle child per active workspace while pipelines,
  models, frames, and credentials deliberately remain in the parent. It proves
  ownership, crash containment, and cleanup hooks only; reports must not call it
  tenant isolation. Domain failure stops all owned parent runs concurrently. If
  a wedged pipeline exceeds the bounded containment deadline, the experimental
  worker hard-exits so Kubernetes and the heartbeat reaper can requeue its held
  jobs instead of losing the sole monitor. Full blockers are documented in
  [`experiments/process_isolation/WORKER_INTEGRATION.md`](experiments/process_isolation/WORKER_INTEGRATION.md).
  A separate default-off staging topology,
  `PROCESSOR_JOB_EXECUTION_MODE=process`, now moves each live job's decoder,
  workflow, model, CUDA context, and direct MediaMTX publisher into a spawned OS
  process. Claims, heartbeats, cancellation, browser tokens, aggregate metrics,
  and durable failures remain supervisor-owned; frames/tensors never cross IPC.
  Use the controlled original/PyAV vs v1.4/PyAV vs v1.4/NVDEC D/E/F matrix in
  [`experiments/process_isolation/JOB_PROCESS_MATRIX.md`](experiments/process_isolation/JOB_PROCESS_MATRIX.md).
  Separately, draft inference
  [#2788](https://github.com/roboflow/inference/pull/2788) packages the new
  multi-process model manager benchmark. Its current routing key is
  `model_id:instance`, not workspace identity: tenants using the same model and
  empty instance can share one backend process, while distinct instances load
  separate processes. Treat raw MPS as a throughput/fairness experiment, never
  as an authorization or memory-isolation boundary. PR #2788 now also includes
  a 200 GiB Cloud Build path and a digest-only renderer for one isolated L40S
  Deployment in `video-proc-bench-mmp`; it refuses production repositories and
  does not replace the normal `video-proc` pool. The first local image build
  reached the pinned CUDA runtime but exhausted Docker Desktop's internal build
  storage, so the registry build remains pending staging GCP reauthentication.
  A read-only probe of the current staging image on `l40s-48gb.10x` confirmed
  driver `570.133.20`, Torch `2.6.0+cu124`, CUDA `12.4`, and both raw-MPS
  binaries. At probe time the live worker had only the Docker-default 64 MiB
  `/dev/shm`; applied staging infra PR #2454 later replaced it with a 2 GiB
  memory-backed volume. The node advertises `mig.capable=false`,
  `mps.capable=false`, GPU sharing
  strategy `none`, and one replica per device. Therefore Gate 4 can test raw MPS
  inside one exclusive-GPU pod after the 4 GiB shared-memory deployment, but it
  cannot test MIG or Kubernetes device-plugin MPS sharing on this L40S fleet.
  Staging Cloud Build `f0bd00b6-f0d1-410a-8ba6-bf9f8e8ccfd0` subsequently
  produced the dedicated benchmark image at
  `mmp-benchmark@sha256:6a6592f77e0eb1d3bfc8b82d7add6a7206e946a437a072f7f3a58cf693b1716d`
  from PR #2788 revision `5f02db12ebdda013ff92e6607f92f88c7f9582ec`.
  A separate multi-workspace MMP load pod should still start with 4 GiB.
  Draft inference
  [#2789](https://github.com/roboflow/inference/pull/2789) is the isolated
  video-worker integration: it merges the exact #2251 manager head, passes its
  compatibility adapter into workflow-backed `InferencePipeline` instances,
  and creates one bundled manager per workspace. Same-workspace jobs can share
  loaded model subprocesses and auto-batching, while different workspaces use
  separate loaded model processes and in-memory route caches. The parent still
  owns all pipelines, frames, credentials, and publishing, and every child can
  see the pod filesystem/download cache, so this is not a tenant security
  boundary. It includes a full merged GPU-image build, an image-level manager
  readiness smoke, and a digest-only bounded local-workflow Pod that never
  joins the staging ready pool. Cloud Build
  `984175ce-625d-42aa-9f93-ba691d1006b1` passed and produced worker digest
  `sha256:c6ad147dd30897874dc3a5dda4fc97345ab9f4220d15405139bf76fde415b1cd`.
  The standalone Crusoe staging smoke completed the YOLOv8 Nano workflow for
  all 538 frames with zero drops at 15.38 delivered FPS. Status reported one
  `mmp-bundled-subprocess` workspace domain; the worker remained PID 1 while
  the loaded model ran in PID 169 using 892 MiB VRAM. Decoded frames exceeded
  the original 12 MB shared-memory slot, so the minimum tested configuration
  is eight 32 MB slots (256 MB total) inside the 2 GiB `/dev/shm`. The test Pod,
  ConfigMap, and short-lived API-key Secret were removed after evidence
  collection. A same-workspace stream A/B then used that exact image, one L40S,
  real-time pod-local RTSP replay of the 4K fixture, YOLOv8 Nano, and 5 FPS per
  stream. Legacy delivered 18.053 FPS at c4 and 33.134 FPS at c8 using 2016 and
  3516 MiB. Bundled subprocess MMP used only 890 MiB but fell to 13.305 FPS at
  c4 and 11.363 FPS at c8. The bundled direct control retained model reuse
  without the subprocess boundary and delivered 17.992 FPS at c4 and 33.070
  FPS at c8 using 962 and 1078 MiB. This isolates the current regression to the
  subprocess path rather than manager/adapter lookup; decoded 4K shared-memory
  transport was the leading hypothesis. The 640p control confirmed it:
  subprocess MMP delivered 17.067 FPS versus legacy's 16.812 at c4 and 33.097
  versus 33.852 at c8, still using only 890 MiB. The ndarray path currently
  serializes with parent-side `np.save`, copies into SHM, then copies and loads
  the `.npy` payload again in the child. Replace that with typed buffer
  descriptors or pipeline-side preprocessing to a model-sized tensor; add
  slot, marshal/unmarshal, batch, and GPU telemetry. The target boundary is one
  process per processing job, fed by a shared bounded source-frame ring and
  using a workspace-scoped model service. Keep worker pods workspace-affine
  because same-pod processes and L40S MPS are not hard tenant boundaries.
  Direct mode is not a process-isolation or tenant-security result. Cross-
  workspace fairness, failure containment, and raw-MPS comparisons remain
  unmeasured.
- **Fault injection is bound to the actual video cell.** The dry-run controller
  accepts only kubeconfig context/cluster `ck8s-stg` at the exact Crusoe staging
  API server and refuses a context alias pointed elsewhere. Earlier draft
  examples incorrectly named the unrelated GKE platform staging cluster; they
  were corrected before any pod deletion. Processor evidence revalidates the
  exact service-reported job/pod immediately before the write and joins the
  old/new pod identities to verified frame progress in a complete report.
- **No recording** (phase 3 by design).
- ~~Connector camera identity is by enumeration index~~ **CLOSED**: macOS
  reshuffles avfoundation indices when devices come and go (lid close,
  Continuity camera, USB replug), which relabeled platform source records and
  made capture legs grab the wrong device or hang (observed twice in one day
  of local testing). Cameras are now keyed `usb:<name-slug>` and ffmpeg opens
  the device by its exact avfoundation NAME (identically-named duplicates fall
  back to index selection with ordinal-suffixed IDs). Existing `usb:<n>`
  source records re-register under the new IDs on next connector restart;
  Linux v4l2 already used stable `/dev/videoN` paths.
- **No metering** (the intended model: stream-hours + GPU-seconds).
- **`imageOutput` list outputs** (arrays of images) are redacted from events but not
  stored/previewable.
- **Soak and recovery certification is prepared, not yet measured.** The staging
  harness has an exact GPU/CPU 15-minute -> 1-hour -> 4-hour -> 12-hour ladder.
  Every point publishes output through the external 60-second watch lease,
  renewed every 20 seconds by the runner, and is promoted only by a hash-bound
  analyzer that recomputes report, counter, cleanup, watch, latency, restart,
  relay, Prometheus coverage, and host/relay/VRAM drift evidence, including
  every shorter predecessor. Processor-pod
  faults already join exact old/new workers to post-requeue frame progress;
  relay faults now join exact old/new relay ownership to stable processor
  identity plus post-replacement frame/output progress, explicitly as a polled
  upper bound rather than gapless-media proof. Per-job child crashes remain in
  the c2 process-containment evidence procedure rather than a generic PID-kill
  controller. No soak or live fault result should be inferred from this harness
  preparation.

## 9. Where the design is heading (updated 2026-08-10)

Decisions and direction that came out of team review — this is current intent, not
yet all implemented. The detailed scaling plan, benchmarks, rollout gates, and open
questions live in [MULTI_CELL_SCALING_RFC.md](MULTI_CELL_SCALING_RFC.md).

- **A live source gets a sticky home cell on first activation.** Registration remains
  metadata-only. The first preview or live job transactionally selects and persists a
  cell/relay shard; ordinary jobs inherit it so connector ingest, relay fan-out, and
  processing stay colocated. Workspace/connector policy biases or constrains the
  choice, enabling regional and dedicated cells. Idle sources may be reassigned;
  active migration waits for a generation-aware connector handshake and acceptable
  stateful-workflow recovery.

- **Scaling model: ready pool, not replica scaling.** "There are always N workers
  ready to accept jobs." A Deployment manages only *ready* workers; claiming a job
  **detaches** the worker (it relabels its own pod so the replica controller no
  longer owns it → the pool refills automatically), and a finished worker
  **deletes its own pod**. Rationale: any ordinary ReplicaSet/StatefulSet
  scale-down picks victims blindly and will kill a mid-job box; in the pool model
  the only workers that ever terminate chose to. Recovery for crashes/evictions =
  heartbeats + reap-to-requeue (§8). **IMPLEMENTED (2026-07-07)**: the chart ships
  the ready-pool Deployment (`PodSelf` in processor.py does the label-detach and
  self-delete; the reaper requeues with a 3-attempt cap).
  Costs: worker RBAC (patch/delete own pod), pod-IP-based gateway routing (random
  pod names), a janitor for leaked non-Running working pods, and awareness that
  long-lived monitoring pods outlive Deployment rollouts (they drain via
  requeue, which the relay makes cheap).

  **Documented successor: `pod-deletion-cost` annotations** (one plain
  Deployment, all pods managed; workers annotate their pod with current load,
  scale-in kills the emptiest). Wins vs label-detach: no orphans (no janitor,
  no rebirth guard), PodDisruptionBudgets actually protect busy workers during
  node drains, and cost-as-load gives the autoscaler a gradient that fits the
  multi-stream fill/drain future far better than a binary ready/working label.
  Why NOT yet: (a) rolling updates ignore deletion cost — every deploy would
  rotate every months-long stream, whereas the ready-pool's rollouts only ever
  touch idle pods; (b) cost is honored best-effort by spec; (c) the warm floor
  becomes autoscaler arithmetic again instead of instant ReplicaSet refill.
  **Decision trigger**: switch when stream re-placement is verifiably seamless
  — i.e. requeue + relay reattach (done) PLUS externalized workflow-block state
  (Pawel's EE work) make "deploy churns every stream" mean "a few seconds of
  blur," not lost trackers. At that point the simpler managed-Deployment model
  wins and the label-detach machinery can be retired.
- **Dedicated Deployments converge with cells.** A DD (existing product: an
  inference server on a customer-dedicated GPU) gets extended to also run one or a
  pool of **separate, lean processor processes** on the same box — signaled via
  the server/platform, NOT built into the inference HTTP server; the lean
  processor is what makes fast starts possible. Dedicated vs pooled is a
  *placement decision* (workspace-pinned claims), not an architecture fork.
- **Local vs remote execution is a per-block decision inside the workflow
  execution engine** — not a top-level "decode video, ship frames" split (shipping
  decoded frames explodes bandwidth; encoded video is orders of magnitude
  smaller). Direction agreed with Pawel: (a) measure workflow **bulkiness**
  (per-stream cost at a target fps) to enable multi-stream packing and stability
  guarantees; (b) **externalizable/cacheable state** for stateful blocks
  (trackers, counters) so a dead worker's streams *resume* elsewhere rather than
  restart — pairs with reap-to-requeue: infra re-places in seconds, workflow state
  makes it seamless; (c) selective externalization — pre/post-processing stays
  local to decode, only forward passes go remote, and only when needed (big VLMs;
  stream to/from the serving side rather than request-per-frame); (d) all of it
  lands as coherent changes to the EE's streaming mode.
- **Application-level stream control.** The connector control channel gets
  extended so the platform can signal *what to send*: target fps/resolution
  derived from the workflow's measured bulkiness (a 5fps workflow should not cost
  a 30fps uplink), camera substream switching, and per-source protocol choice —
  WebRTC push where network-level congestion adaptation matters (RTSP has none).
  The relay itself stays transcode-free by default (full-quality frames, no
  generation loss, no added latency); transcoding is possible if a use case
  demands it but is a last resort because it alters what models see.
- **Relationship to the rtsp-bridge spike** (github.com/roboflow/rtsp-bridge-poc):
  same transport, independently validated — camera → outbound ffmpeg (`-c copy`
  remux over RTSP/TCP) → mediamtx → colocated consumer. The connector is the
  managed layer on that shape: discovery, source records, control channel,
  on-demand streaming, per-stream keys. Transport findings transfer 1:1.

## 10. Where to pick up

1. Read the README for the local runbook (node 24 / redis / staging env quirks are
   documented there — they are real and will bite you).
2. Staging and the first feature-flagged production cell are live. The implemented
   base includes outbound connector ingest, relay auth, fleet auth, transactional
   claim, ready pools, orphan requeue, per-job processor tokens, multiple workflows
   per source, up to four jobs per worker, GCS batch results, and relay-published
   watched outputs.
3. Next: complete Phase 0 of
   [MULTI_CELL_SCALING_RFC.md](MULTI_CELL_SCALING_RFC.md): agree on SLOs; review,
   merge the already-applied staging MediaMTX observability chart from infra
   [#2443](https://github.com/roboflow/roboflow-infra/pull/2443); deploy and
   validate the aggregate processor telemetry overlay; then run the controlled
   FPS, multi-workspace, MPS/MMP, recovery, mixed-workload, and soak matrices.
   A production apply remains a separate approval step.
4. Then introduce cell-aware source/job contracts while only East is registered;
   prove claim isolation before deploying a second non-production cell.
5. Remaining adjacent hardening: job-addressed live events, connector-source polish,
   recording/metering, and externalizable state for seamless stateful recovery.
