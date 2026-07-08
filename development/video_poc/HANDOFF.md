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

Production deltas beyond auth: pin the version, TLS on ingest, colocation with the
warm pool as a "cell", and capacity metrics (aggregate ingress/egress is the cell
sizing input).

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
  never sent to browsers), state: queued|claimed|running|completed|error|cancelled,
  attempts (requeue counter, capped at 3), cancelRequested, processorId?,
  processorUrl?, heartbeatAt?, stats?, resultsFiles?/resultsUploadedAt? (GCS),
  error?, created_at, updated_at}`. The claim payload derives from this doc plus
  `sourceUrl` (signed GCS URL or credentialed RTSP consume URL), `apiKey` (the
  job workspace's inference key), and `simPublishUrl`.

## 8. Known gaps and how they're meant to close

- **Browser→processor addressing is solved (gateway), but coupled to workers.**
  Locally `job.processorUrl` is localhost; in the cluster it's the processor-gateway
  path to a specific worker. The remaining production work is *job-addressing*
  (`/video-jobs/{id}/events` — see §6 "Consuming results") and **auth** (below).
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
- **Processor HTTP has no auth** and CORS `*` — the gateway hostname is the only
  barrier in staging. This is the top prod-readiness item. Planned shape: the
  platform mints a per-job access token (same pattern as stream keys), returns it
  with the job doc to authorized UI/API callers, and the processor (or the gateway)
  checks it on `/events`, `/status`, `/preview.mjpeg`, `/results`. The job-addressed
  events endpoint (§6) subsumes most of this for programmatic consumers, since the
  platform authenticates at its own front door.
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
- **Single-workspace claim**: processors claim jobs only for the workspace of their API
  key. Real warm pools need cross-tenant scheduling, leases/heartbeats on claims, and
  model-affinity placement.
- **One stream per GPU**: multi-stream-per-GPU is required for monitoring economics.
  Prerequisite: measuring workflow "bulkiness" (per-stream cost of a graph at a
  target fps) so a scheduler can pack N streams per node with stability guarantees —
  see §9.
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

## 9. Where the design is heading (team alignment, 2026-07-07)

Decisions and direction that came out of team review — this is current intent, not
yet all in the PRs:

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
2. The task list that produced this: dev tooling ✅, backend API ✅, connector ✅,
   processor ✅, frontend ✅, e2e demo verified through "workflow on uploaded file with
   live annotated preview" ✅.
3. Prod-readiness order (per §8/§9, post-review): ~~reap-to-requeue~~ ✅;
   ~~ready-pool scaling swap in the chart~~ ✅; processor endpoint auth (per-job
   tokens); the job-addressed events endpoint; ~~relay-published results stream +
   `watchRequestedUntil`~~ ✅; multi-stream-per-GPU (needs bulkiness measurement);
   connector-source e2e polish (USB/RTSP got less testing than files).
