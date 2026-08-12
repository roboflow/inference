# Staging experiment ledger

This file is the reconciliation point between live staging experiments and
deployable Git state. Production changes are out of scope.

For every staging mutation, add a row before applying it. Update the same row
with measurements, disposition, and a Git/image reference before moving to the
next experiment. Do not leave useful live configuration represented only in a
`kubectl` command or shell history.

| ID | UTC window | Hypothesis | Live target and change | Git/image source | Revert | Result artifact | Disposition |
|---|---|---|---|---|---|---|---|
| `gpu-pack-current-001` | `2026-08-12T21:35Z`–pending | The current L40S worker has useful capacity above the existing four-job policy cap | `ck8s-stg/video-proc` Deployment UID `f001fcc0-4cc7-44a5-9b5f-c6f1010a1e4d`: GPU `MAX_CONCURRENT_JOBS` changed `4` to `24`; prior generation `8`, revision `7`, pod-template SHA-256 `b3eecf9e29751dcdd1c3717734824abe9e06b198aac273bed6355a6d07f88545`; new pod `video-processor-pool-9946c99c5-qzvj2` reports capacity/available `24/24`, post-change template SHA-256 `296f4c3f17f4302ebb1c14206b2074e210dc6d70f3ab7040059ae3b2a4a18ce4` | worker image `video-processor:2e4a97ee5`; harness `4db351b2a`; telemetry image source `ae2806b88`; the deployable chart change must be committed separately if retained | restore `MAX_CONCURRENT_JOBS=4`, wait for rollout, require one ready pool pod, and verify the resulting worker reports capacity 4 | ignored `api-corpus-gpu-current-unbounded-c8-20260812a.json`; c8: 55.464 delivered FPS, 41.5% spread, 104.3 ms worst sampled EMA p95, GPU 4.63% avg/8% max, CPU 2.00 avg/2.85 max | active for controlled-FPS tests; unbounded progression stopped at c8 SLO failure; cap is not certified capacity |
| _example_ | _2026-08-12T00:00Z_ | _Higher worker capacity exposes the GPU knee_ | _staging `video-proc` GPU pool: `MAX_CONCURRENT_JOBS=4` to `24`_ | _commit SHA; immutable image digest; rendered values SHA-256_ | _restore prior Deployment revision and verify ready replicas_ | _ignored `results/SUITE_ID/suite.json`; durable summary committed separately_ | _pending / keep / revert_ |

## Required evidence

- Record the cluster context, namespace, workload UID, prior revision, image
  digest, rendered configuration hash, and operator identity.
- Record the exact benchmark suite/run IDs and the Prometheus time window.
- Keep API keys, access tokens, source URLs containing credentials, and generated
  job files out of the ledger and Git.
- A successful experiment is not complete until its reusable code/config is
  committed and pushed. A rejected experiment is not complete until staging is
  reverted and the reason is recorded.
- Before a production proposal, verify the live staging workload can be recreated
  from the recorded commits and immutable images without manual patches.
