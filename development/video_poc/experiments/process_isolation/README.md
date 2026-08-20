# Video processor process-isolation and MPS experiment

This directory is intentionally separate from the deployed POC worker. It
contains the lifecycle prototype and the test plan needed before changing the
staging worker.

## Recommended architecture

Use three process layers inside one GPU-owning pod:

1. **Supervisor process** — owns platform claims, job access tokens, public
   HTTP authentication, heartbeats, aggregate metrics, and child lifecycles. It
   must never initialize CUDA.
2. **Workspace execution processes** — one spawned process per active
   workspace by default. Each process may run several `JobRun` pipelines for
   that workspace. `PROCESSOR_ISOLATION_MODE=job` remains an experimental
   stricter option. A Python/GIL failure or malformed workflow cannot cross a
   workspace boundary.
3. **Model Manager Process (MMP)** — the implementation on
   `origin/feat/new-model-manager`. Each workspace process uses its own
   `MMPClient`/`ModelManagerAdapter`; the pod-global MMP routes through ZMQ and
   shared-memory slots to one spawned backend per loaded model. Same-model
   requests can auto-batch across workspace processes without loading the model
   in every pipeline process.

The model backend remains a shared cross-workspace execution boundary. A
backend crash affects all jobs currently using that model, but not pipelines
using other models; the MMP already detects worker death and reloads the model.
A fatal GPU or MPS-server fault can still affect the entire pod. Process
isolation reduces that blast radius but does not eliminate it.

The first integrated staging capacity experiment uses one process per job. It
is deliberately stricter than the earlier workspace recommendation because it
cleanly answers whether shared-interpreter contention explains the observed
scaling ceiling. See [`JOB_PROCESS_MATRIX.md`](JOB_PROCESS_MATRIX.md). Workspace
grouping remains a possible production optimization after D/E/F establish the
per-process cost and throughput bound.

## Integration work on the worker

The current worker cannot simply wrap `JobRun` in `multiprocessing.Process`.
The parent and child responsibilities need an explicit protocol:

- Parent to child: `start`, `cancel`, output-watch selection, graceful stop.
- Child to parent: state, structured error/log tail, stats, completion, and
  process-health events.
- Output WebRTC/RTSP publishing stays in the child and goes directly to
  MediaMTX. No video frame needs to cross the parent boundary for normal UI.
- Debug MJPEG can initially be disabled in process mode, then proxied from a
  child-only loopback endpoint. Do not copy raw frames through a Python queue.
- The parent immediately marks every job in a dead workspace process as
  failing and frees its claims. It must not wait for heartbeat reaping.
- Use only the `spawn` multiprocessing method. Forking after CUDA, PyAV, gRPC,
  or threaded ZMQ initialization is unsafe.
- Start the MMP before accepting jobs, publish its ZMQ address and SHM geometry
  to child environments, and create the `ModelManagerAdapter` inside each child.
- Do not call MMP `unload`/`clear` when one workspace exits; model lifetime is
  pod-global and should remain idle-time/VRAM-eviction managed.

The MMP integration needs two correctness additions before cross-workspace
traffic is certified:

- attach a non-secret tenant/workspace identity to submissions and metrics;
- enforce per-tenant inflight limits or fair queuing. The current MMP routes
  submissions immediately and model workers greedily batch arrival order, so a
  high-FPS producer can dominate a shared model queue.

The adapter's `stat_model_while_checking_auth` verifies platform-model access
inside each workspace process before a route is cached. The security suite must
still prove that a warm model loaded by workspace A cannot be used by an
unauthorized workspace B and that SHM slots/results never cross client IDs.

## MPS role

L40S supports CUDA MPS but not MIG. MPS can overlap kernels from different
model backend processes and reduce CUDA-context scheduling overhead. It is not
a tenant-security boundary:

- active-thread percentages cap a model process, not a workspace, because the
  shared MMP has one backend process per model;
- MPS does not provide hard memory isolation; pinned-device-memory limits are
  useful guardrails, not reservations;
- a fatal client GPU fault is reported to all MPS clients sharing that GPU and
  can put the MPS server in `FAULT` until affected clients exit;
- abruptly terminating a client with outstanding GPU work can leave other MPS
  clients in an undefined state.

Therefore benchmark MPS as an optional execution mode, never as the safety
mechanism. Compare:

1. MPS off (normal multi-context CUDA scheduling).
2. MPS on, no active-thread cap.
3. MPS on with 50% and 25% per-model caps.
4. Same-model auto-batching versus different-model concurrent backends.

The upcoming server branch already has a basic `NVIDIA_MPS=1` launcher. The
runtime image must actually contain `nvidia-cuda-mps-control` and
`nvidia-cuda-mps-server`; its default plain Ubuntu runtime may not. Run
`mps_probe.py` in the built image before enabling the flag. Give each pod a
private `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` and verify the
daemon sees only the GPU assigned to the pod.

The MMP SHM pool also requires a memory-backed `/dev/shm` volume. At the branch
defaults, 32 slots times 25 MB already needs roughly 800 MB plus headers; use a
1–2 GiB `emptyDir.medium: Memory` mount initially and include it in pod memory
sizing.

## Staging image/build prerequisites

Build a dedicated immutable experimental image from
`origin/feat/new-model-manager`, not the current released inference-server base.
It needs:

- the legacy `inference` workflow package plus `inference-models`,
  `inference-model-manager`, and `inference-server` from the same commit;
- ffmpeg, PyAV, aiortc, Pub/Sub and the current video processor files;
- a CUDA runtime image that includes MPS control/server binaries;
- `/dev/shm` memory volume and writable MPS pipe/log directories;
- `PYTHONUNBUFFERED=1`, MMP slot/batch parameters, isolation mode and benchmark
  build identity exported as metrics labels.

Do not merge the long-lived feature branch into the POC blindly. First build a
compatibility image and run the existing workflow corpus; the branch is
hundreds of commits ahead and changes both model APIs and the inference server.

## Comprehensive staging matrix

Run each workload in thread baseline, workspace-process/MMP, and job-process
modes. Run GPU cases with MPS off/on. Test 1/2/4/8/12/16/24 streams until the
first SLO failure, and repeat the certified point for 1 hour and 24 hours.

Workload groups:

- same small YOLO model, same workspace;
- same small YOLO model across distinct workspaces;
- distinct YOLO models across workspaces;
- detector plus tracking;
- two-model workflow;
- instance/semantic segmentation;
- one heavy adversarial workflow joining steady light streams;
- output publishing off, RTSP on, and WHIP on;
- 720p/15, 1080p/30, and mixed source rates.

Capture per stream and per tenant:

- delivered FPS and dropped/decode frame counts;
- decode-to-result and glass-to-glass p50/p95/p99;
- MMP queue/inflight depth, batch fill, batch wait, infer time and error count;
- CPU, RSS, `/dev/shm`, GPU utilization, VRAM by backend PID, power, NVDEC and
  NVENC utilization;
- model load count/time, reloads, TTFR, output bitrate and reconnects;
- fairness: minimum/median stream throughput, Jain's fairness index, and the
  slowdown imposed on incumbent streams by a new heavy neighbor.

Fault-injection gates:

- normal workflow exception and parse error;
- kill one workspace process;
- kill one model backend process;
- model-load and runtime OOM;
- abrupt MPS client termination with GPU work outstanding;
- device-side assertion/illegal memory access on a dedicated staging worker;
- MPS server fault/recovery and whole-pod restart;
- repeated model/workspace churn for RAM, VRAM, SHM-slot and file-descriptor
  leaks;
- cancellation during load, inference, output publishing and process exit.

Security gates:

- unauthorized workspace tries a model already warm for another workspace;
- unique frames/watermarks per tenant detect result or preview crossover;
- unique output paths and tokens detect HTTP/WHEP crossover;
- verify API keys never appear in process names, domain keys, metrics, status,
  command-line arguments, or unsanitized log tails.

## Local lifecycle probe

```bash
PYTHONPATH=. pytest -q development/video_poc/experiments/process_isolation
python -m development.video_poc.experiments.process_isolation.mps_probe
```

Inside an exclusively assigned staging GPU pod only:

```bash
python -m development.video_poc.experiments.process_isolation.mps_probe --start
```

Starting MPS changes GPU process state; do it only on the dedicated experiment
pod and stop/delete the pod after the run.
