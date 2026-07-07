# Staging deployment plan — video POC on Crusoe

> **Status 2026-07-07**: the "day-2" items below are now IMPLEMENTED, not
> planned — per-stream keys + relay external auth, per-job API key in the claim
> payload, batch results to GCS (with processor-local fallback), Pub/Sub
> dispatch wake-ups (claim stays the transactional source of truth; claim is
> now an actual Firestore transaction), the `video_processor_busy` metric, and
> the KEDA warm-floor autoscaling. The Helm chart exists:
> `helm/roboflow-video-proc/` in roboflow-infra
> ([PR #2290](https://github.com/roboflow/roboflow-infra/pull/2290)) — its
> README lists the remaining MANUAL steps (image push, Pub/Sub topic/sub,
> secrets, DNS, functions env). What's left of this document that still needs
> doing is exactly that manual list plus the smoke tests.
>
> **Update (rename + all-Terraform)**: the project is now **video-proc** (not
> staying a POC) and the entire deployment — Pub/Sub, processor SA, namespace
> secrets, AND the helm release — is owned by the `crusoe/video-proc`
> Terraform stack in roboflow-infra (Spacelift, depends on crusoe/addons; the
> namespace itself lives in addons like every app namespace). deploy.sh is
> gone: merge to master + confirm the Spacelift run IS the deploy. Env vars
> renamed VIDEO_POC_* → VIDEO_PROC_*; functions env ships in the roboflow
> repo's .env.roboflow-staging (.env.local overrides for laptop dev). The only
> out-of-band steps left: seed VIDEO_PROC_WORKSPACE_API_KEY in Secret Manager
> once, build/push + pin the processor image, and DNS records after the first
> LB IP. Manual kubectl/gcloud sections below are superseded by the chart
> README + the stack.

Goal: mediamtx (relay) + video processors running on the Crusoe staging cluster
(`ck8s-stg`), talking to the staging Roboflow platform, demo-able end to end.
This plan assumes the POC as of PRs
[inference#2616](https://github.com/roboflow/inference/pull/2616) and
[roboflow#13264](https://github.com/roboflow/roboflow/pull/13264); read
[HANDOFF.md](HANDOFF.md) first for what the components are.

## 0. Decision: where does this deploy from?

Two candidates were evaluated (both ultimately target the **same cluster**,
`ck8s-stg` in Crusoe us-east1):

- **`async-serverless`** — skaffold + Helm + Cloud Build, namespace `async`.
  Has a ready GPU-worker pattern (consumer-inference: L40S selectors, GPU
  taints, secret-fetcher init container, KEDA). But: everything there is built
  around the HTTP→RabbitMQ→consumer request flow, which the video processor
  does not use; adding a service means wiring a Dockerfile into Cloud Build,
  skaffold modules, CI build checks, and the GitOps version-bump sed logic;
  and the repo has **zero LoadBalancer / non-HTTP precedent** for the relay.
- **`roboflow-infra`** — Terraform via Spacelift for infra, plain **Helm charts
  in `helm/` deployed by the `helm-deploy-crusoe.yaml` GitHub Action** (or a
  chart-local `*_deploy.sh`) for apps. The cluster addons it manages are
  exactly what we need already running: Traefik ingress + cert-manager
  (DNSimple DNS-01), the **Crusoe L4 LoadBalancer controller** (reacts to any
  `Service type: LoadBalancer`), registry pull secrets, GPU node pools.

**Decision: a new Helm chart `helm/roboflow-video-proc/` in `roboflow-infra`,
deployed to a new `video-proc` namespace on `ck8s-stg`. No new Terraform/
Spacelift stack** (that would mean a new cluster/VPC — overkill; the agent
model for a stack here is cluster-level isolation). Not async-serverless for
now because the processor is poll-driven, not queue-driven — none of that
repo's machinery helps, and its CI wiring costs a day we don't have. If the
processor later adopts queue-based dispatch and KEDA scaling, migrating it
into async-serverless as another skaffold module is straightforward and this
plan doesn't foreclose it.

Chart to copy as a starting shape: `helm/inference-internal-crusoe/`
(Deployment + Service + Traefik IngressRoute + cert-manager Certificate) or
`helm/events-relay/` (simpler).

## 1. Target topology

```
 laptop / customer network                Crusoe ck8s-stg (namespace video-proc)
┌────────────────────────┐      ┌───────────────────────────────────────────────┐
│ connector ──RTSP push──┼──────┼─▶ Crusoe L4 LB :8554 ──▶ mediamtx             │
│ browser ───WHEP────────┼──────┼─▶ Traefik :443 ───────▶ mediamtx :8889        │
│ browser ───status/SSE/─┼──────┼─▶ Traefik :443 ──▶ nginx gateway ──▶ processor-N :8890
│            mjpeg/results│      │                                               │
│                        │      │   processors (StatefulSet, L40S GPU)          │
│                        │      │     │ consume RTSP (cluster-internal svc DNS) │
│                        │      │     │ publish sim replays (internal)          │
└────────────────────────┘      │     ▼ outbound HTTPS                          │
                                └─────┼─────────────────────────────────────────┘
                                      │ claim/status poll          ▲ GCS signed URLs
                                      ▼                            │ (batch downloads)
                        staging Firebase functions (GCP)          GCS
```

Hostnames (staging convention is `*.crusoe.roboflow.one` — `.one` IS staging;
prod would be `.com`. Do not invent `video-staging.roboflow.com`):

| Host | Path to | Protocol | Wired via |
|---|---|---|---|
| `video-ingest.crusoe.roboflow.one` | mediamtx :8554 | RTSP/TCP | **new** `Service type: LoadBalancer` (annotation `service.beta.kubernetes.io/crusoe-load-balancer-scheme: external`, `externalTrafficPolicy: Local`) + DNSimple A record to LB IP |
| `video-relay.crusoe.roboflow.one` | mediamtx :8889 | HTTPS (WHEP) | Traefik IngressRoute + cert-manager Certificate |
| `video-processors.crusoe.roboflow.one` | nginx gateway → pod :8890 | HTTPS (JSON/SSE/MJPEG/mp4) | Traefik IngressRoute + Certificate |

Two separate hosts for relay vs processors on purpose: WHEP returns session
resource URLs in `Location` headers, so serving mediamtx at a path prefix
behind a rewriting proxy is asking for breakage — give it a root.

**WebRTC media path (do not miss this):** Traefik only carries the WHEP
*signaling* (HTTP). The media itself is RTP over ICE, which must reach
mediamtx directly. Plan: enable mediamtx **ICE TCP mux** (`webrtcLocalTCPAddress
:8189`), expose TCP 8189 on the same `mediamtx-ingest` LoadBalancer as RTSP,
and set `webrtcAdditionalHosts: [<LB IP or video-ingest hostname>]` so the
advertised ICE candidates point at the LB. TCP-mux WebRTC is slightly worse
than UDP but one port, one LB, and it works through the same untested-LB risk
we're already carrying. Test = WHEP playback actually rendering frames, not
just a 201 from the signaling request.

## 2. Kubernetes objects (the chart's contents)

1. **Namespace** `video-proc` (chart `--create-namespace`, or add to
   `crusoe/addons/namespaces.tf` if we want it Terraform-owned).
2. **mediamtx**: Deployment ×1 on the default CPU pool (untainted), upstream
   image `bluenviron/mediamtx:<pinned-version>` — pin it, do not use latest.
   ConfigMap = our `mediamtx.yml` plus staging deltas (below). Services:
   - ClusterIP `mediamtx` (8554, 8889) — what processors and the gateway use
   - LoadBalancer `mediamtx-ingest` (TCP 8554 only) — the new primitive
3. **processor**: **StatefulSet** ×2 (`video-processor-0/1` — stable names are
   what makes per-pod routing work) + headless Service. GPU scheduling copied
   from inference-internal: `nodeSelector: {gpu_type: L40S}`, toleration
   `gpu=true:NoSchedule`, `resources.limits: {nvidia.com/gpu: 1}`, optionally
   `schedulerName: gpu-binpack-scheduler`. Env below. Image: see §3.
   **UPDATE: replaced by the ready-pool model (Deployment + label-detach +
   self-delete) — IMPLEMENTED; see §8 and the chart. Gateway now routes by
   pod IP (`/ip-a-b-c-d/`, CIDR-pinned regex) since pool pods have random
   names.**
4. **processor-gateway**: nginx Deployment ×1 (CPU pool) routing
   `/{pod}/{rest}` → `http://{pod}.video-processor-headless.video-proc.svc.cluster.local:8890/{rest}`.
   Critical nginx settings: `proxy_buffering off` and `proxy_http_version 1.1`
   (SSE and MJPEG die behind buffering), long `proxy_read_timeout`, resolver
   set to cluster DNS with a variable proxy_pass (so nginx re-resolves pod DNS).
5. **Secrets**: workspace API key for processors from GCP Secret Manager
   (`roboflow-staging`), materialized to a k8s Secret by the deploy workflow —
   same pattern `helm-deploy-crusoe.yaml` already uses for inference-internal
   creds. Image pull: reuse `gcp-ar-pull-secret`.
6. **Deploy hook**: add a `video-proc` service option to
   `.github/workflows/helm-deploy-crusoe.yaml` (~15 lines), or ship
   `helm/roboflow-video-proc/deploy.sh` for day one and wire CI after.

## 3. Images

- **mediamtx**: upstream image + ConfigMap; nothing to build.
- **processor**: new `development/video_poc/processor/Dockerfile` in the
  inference repo:
  ```dockerfile
  # see processor/Dockerfile (canonical): inference GPU base + ffmpeg CLI
  # + google-cloud-pubsub/requests + processor.py, ENV FFMPEG_BIN=ffmpeg
  ```
  The base image ships the full `inference` package with CUDA onnxruntime; the
  runtime stage does NOT include the ffmpeg CLI (checked), hence the apt line
  (processor needs it for sim replays and batch result encoding).
  Build/push for tomorrow: local `docker build` +
  `docker push us-central1-docker.pkg.dev/roboflow-staging/<repo>/video-processor:<sha>`
  (CCR mirror picks it up per the existing registry mirror setup). Cloud Build
  wiring is day-2.

## 4. Control-protocol deltas (code changes, all small)

The full connection audit — what breaks when components leave one machine:

| # | Arrow | Today | Staging | Change needed |
|---|---|---|---|---|
| 1 | connector → platform | localhost emulator | staging functions URL | none (flag/env on connector invocation) |
| 2 | connector → relay ingest | `rtsp://127.0.0.1:8554` | `rtsp://video-ingest.crusoe.roboflow.one:8554` | none in code — platform sends full `ingestUrl` in `start_stream`; set functions env |
| 3 | browser → relay preview | `http://127.0.0.1:8889/...whep` | `https://video-relay.crusoe...` | none — `VIDEO_PROC_WHEP_BASE` env exists |
| 4 | browser → processor | `job.processorUrl` = `http://127.0.0.1:8890` | `https://video-processors.crusoe.../ip-a-b-c-d` | **done**: processor derives its public URL from `GATEWAY_PUBLIC_BASE` + `POD_IP` (pod-IP routing; ready-pool pods have random names) |
| 5 | processor → platform (claim/status) | localhost emulator | staging functions URL | none — `--api-url` flag exists; needs staging **functions deployed from the branch** (see §5) |
| 6 | processor → relay (consume `src-*`) | same base URL as #2 | cluster-internal `rtsp://mediamtx.video-proc.svc:8554` | **functions change**: split `VIDEO_PROC_RTSP_BASE` into `VIDEO_PROC_RTSP_INGEST_BASE` (public, for connector commands) and `VIDEO_PROC_RTSP_CONSUME_BASE` (internal, used in claim's `sourceUrl`). One function file; fallback = one var for both (works via public LB, wastes hairpin bandwidth) |
| 7 | processor → relay (publish `sim-*`) | `VIDEO_PROC_RTSP_BASE` env | internal svc DNS | none — env exists on processor |
| 8 | processor → GCS (batch download) | laptop → GCS | Crusoe → GCS egress | none for now (decision: stay on GCS; Crusoe object storage colocation is future work — batch download latency/egress cost is the tax) |

Security deltas (staging-appropriate, do not skip #9):

| # | Item | Plan |
|---|---|---|
| 9 | relay publish auth | mediamtx `authInternalUsers`: require user/pass to **publish**; platform embeds creds in ingest URLs (`rtsp://user:pass@video-ingest...`) — connector and processor sim-replay pass them through ffmpeg unchanged. Read (WHEP) stays open for the demo. This kills "anyone can push video into our cluster" |
| 10 | processor endpoints | open for the demo (CORS `*` already), gateway host is obscure. Day-2: shared bearer token checked by the processor, injected by the platform into `processorUrl` responses |
| 11 | mediamtx API :9997 | keep `127.0.0.1` bind — never exposed |

## 5. Platform side (roboflow repo)

The processors poll the platform, so the platform must be reachable **from
Crusoe** — the local emulator isn't. Primary: deploy the
`hansent/video-sources-poc` branch functions to the `roboflow-staging`
Firebase project (normal staging functions deploy; the POC routes ride along).
Fallback if branch-deploy to staging is contentious tomorrow: keep the
emulator local and expose it with a `cloudflared` tunnel; point processors and
connector at the tunnel URL. Ugly but unblocks the demo.

Functions env to set (staging): `VIDEO_PROC_RTSP_INGEST_BASE`,
`VIDEO_PROC_RTSP_CONSUME_BASE`, `VIDEO_PROC_WHEP_BASE` per the table above.

## 6. Execution order for tomorrow

Morning (infra, parallelizable):
1. Build + push processor image (§3). Smoke-test locally first:
   `docker run ... processor.py --job-file test-job-blur.json`.
2. Write chart (`helm/roboflow-video-proc/`), `helm install` into `ck8s-stg`
   namespace `video-proc` with GPU replicas=1 to start.
3. Apply the `mediamtx-ingest` LoadBalancer Service; **this is the highest-risk
   unknown** (Crusoe LB controller is private beta; no TCP LB exists in either
   repo today). Verify with a raw
   `ffmpeg -re -f lavfi -i testsrc2 ... -f rtsp rtsp://<LB-IP>:8554/test` from
   a laptop, and `curl :9997/v3/paths/list` from inside the pod.
   **Fallback if the L4 LB doesn't work**: run mediamtx on a GCP VM using the
   existing `scriptops/mediamtx-rtsp-config/` pattern (staging test VMs already
   do exactly this, port 8554 open). Video then crosses clouds — wrong for the
   colocation story, fine for a staging demo; call it out as temporary.
4. DNSimple records: A `video-ingest` → LB IP; `video-relay`,
   `video-processors` → Traefik (CNAME like the other `*.crusoe.roboflow.one`
   hosts); Certificates via cert-manager.

Afternoon (wiring):
5. Processor code change #4 (PROCESSOR_PUBLIC_URL) + functions change #6
   (ingest/consume split) — both small; commit to the existing PR branches.
6. Deploy branch functions to staging (or tunnel fallback), set the three env
   vars.
7. Bottom-up smoke tests, in order — each isolates one arrow:
   a. WHEP playback of the test stream in a browser via
      `https://video-relay.crusoe.roboflow.one/test/whep`.
   b. `curl https://video-processors.crusoe.roboflow.one/ip-<pod-ip-dashed>/status`
      (get a worker's exact URL from a job doc's `processorUrl`).
   c. Upload a file in the app → batch job → processor claims (watch pod
      logs) → MJPEG progress in the modal → completes → scrub the results.
   d. Connector on laptop with `--files-dir` → source appears → live preview
      (WHEP through Crusoe!) → stream-mode job → annotated output.
8. Demo checklist = 7c + 7d working from a clean browser session.

## 7. Known gaps this deployment adds (track, don't fix tomorrow)

- ~~Batch results live on processor-local disk~~ done: uploaded to GCS on
  completion via platform-signed URLs; local files are only the fallback.
  Next step: recorder uploads mp4+jsonl to GCS on finalize, platform stores
  the URLs on the job doc, review UI stops touching the processor entirely
  (also removes the biggest reason browsers talk to processors at all).
- Processor endpoint auth (#10) and per-stream relay credentials (real stream
  keys per source, not one shared publish user).
- GPU-per-processor is wasteful for a demo (idle GPU while no job); fine for
  staging, revisit worker/GPU packing later.
- KEDA/scale: replicas are static; queue-depth-based scaling is exactly where
  the async-serverless machinery becomes relevant.
- mediamtx version pinning + `fetch-deps.sh` alignment.
- GCS→Crusoe egress for batch downloads (revisit when Crusoe object storage
  is adopted; explicitly deferred).

## 8. Poll → queue migration path (design, not tomorrow)

A video job is a **lease, not a request** — monitoring jobs run for months, so
queue ack cannot mean "done"; it means "claimed". The queue therefore only
replaces dispatch; the heartbeat/lease/reaping system built in the POC stays
as the ownership mechanism, and the status poll stays as the control channel
(cancel, watch signaling).

- **Phase A (now)**: polling claim. Prerequisite fix regardless of queues:
  make claim a transactional compare-and-set — today two processors can race
  on the same queued job.
- **Phase B**: **GCP Pub/Sub** for dispatch — NOT RabbitMQ. Rationale: the
  functions already live in GCP and publish natively (no Firestore→queue
  dispatcher bridge needed at all), and — decisively — cells must not depend
  on infrastructure that only exists because async-serverless happens to be
  co-deployed on the same cluster. A cell is relay + processors; it may land
  on clusters that run nothing else. Pub/Sub keeps dispatch cell-agnostic:
  processors consume via StreamingPull over **outbound** HTTPS from anywhere,
  consistent with the outbound-only doctrine everywhere else in this design.
  - Topic `video-jobs`; one subscription per cell with an attribute filter
    (`cell="crusoe-use1"`) so jobs land where their source's relay lives.
  - Ack on claim (not on completion — a monitoring job runs for months);
    processor compare-and-sets Firestore before starting, discarding
    stale/duplicate deliveries (Pub/Sub is at-least-once). Orphan reaping
    additionally re-publishes. Poll mode remains for local dev.
  - Processors on Crusoe authenticate to Pub/Sub with a GCP service-account
    key from Secret Manager — same pattern the cluster already uses.
- **Phase C**: split classes. Batch = true backlog; monitoring = placement
  problem (scale on assigned-streams-per-worker; rebalancing = drain →
  re-publish → re-place, cheap because workers re-attach to relay streams
  locally, not across the customer's NAT).

**Scaling model — REVISED (2026-07-07): ready pool, not replica scaling.**
The first design here was a KEDA warm-floor query (`replicas = sum(busy) +
MIN_IDLE`) scaling the StatefulSet, with `pod-deletion-cost` guarding
scale-down. That guard doesn't exist for StatefulSets (deletion-cost is a
ReplicaSet feature), so scale-in could still kill a mid-job worker. The
revised model eliminates victim selection entirely:

- A Deployment manages only **ready** workers; `replicas = N` means "N workers
  ready to accept jobs" — the warm-pool guarantee is the replica count itself.
- On claim, the worker **detaches itself** (patches its own pod label so the
  ReplicaSet selector no longer matches). The ReplicaSet instantly creates a
  replacement — that's the refill, native reconciliation, no controller code.
- On finish, the worker **deletes its own pod**. The only workers that ever
  terminate chose to; nothing ever aims at a busy pod. Single-use workers also
  give clean GPU/memory hygiene per job.
- Recovery for crashes / node reclaim / spot eviction: heartbeats +
  **reap-to-requeue** (the reaper must re-queue + re-publish instead of
  erroring — this becomes mandatory, with an attempts cap). For monitoring
  jobs the relay makes re-placement a seconds-long blip.
- Costs: worker RBAC (patch/delete own pod, downward-API pod name), gateway
  routes by pod-IP DNS (`<ip-dashed>.<ns>.pod.cluster.local`) since names are
  random, a janitor for leaked non-Running detached pods, and long-lived
  monitoring pods outliving Deployment rollouts (they drain via requeue).
- Elasticity composes safely on top: KEDA (Pub/Sub backlog) may scale the
  *ready pool's* N — every pod it manages is idle by definition, so scale-down
  is always harmless. The `video_processor_busy` gauge stays for observability
  and pool-sizing dashboards.

**IMPLEMENTED (2026-07-07)** across all three PRs: the chart ships the
ready-pool Deployment + RBAC + pod-IP gateway + janitor CronJob (ScaledObject
removed); the processor label-detaches on claim and self-deletes on job end
(`PodSelf`, pool mode auto-off outside a cluster); the platform reaper
requeues orphaned jobs with a 3-attempt cap and refuses status posts from
zombie workers for requeued jobs.

Never queued: heartbeats, cancel/watch signals, results to the browser (see
HANDOFF §6 "Consuming results" for how the events/JSON channel is addressed in
the cluster and its job-addressed production shape).
