# Staging L40S process-image smoke

These Pods validate the rebuilt D/E/F images on one staging L40S without
claiming a video job or receiving platform credentials. They import the worker
in the supervisor and in a spawned child, prove the PIDs differ, and verify the
expected CUDA device and ingest mode.

The manifests are **not** a rollout. Creating or deleting these Pods is a
staging cluster write and requires explicit authorization. Never run them on a
context other than `ck8s-stg` or outside the `video-proc` namespace.

## Immutable manifests

| Leg | Manifest | Image purpose |
|---|---|---|
| D | `l40s-d-legacy-process.yaml` | legacy runtime + PyAV + per-job process |
| E | `l40s-e-v14-process-pyav.yaml` | v1.4 tensor-native + PyAV + per-job process |
| F | `l40s-f-v14-process-nvdec.yaml` | v1.4 tensor-native + NVDEC + per-job process |

All Pods disable service-account token mounting, contain no API keys or service
URLs, request exactly one L40S, use `restartPolicy: Never`, and have a unique
digest-derived name. The command exits nonzero if the parent/child import,
spawn boundary, CUDA visibility, device identity, or selected ingest mode is
wrong.

## Authorized execution procedure

Choose exactly one manifest and copy its exact Pod name. The shell variables
below are deliberately explicit; do not replace the context or namespace with
ambient defaults.

```bash
set -euo pipefail
MANIFEST=development/video_poc/experiments/process_isolation/smoke/l40s-d-legacy-process.yaml
POD=video-process-smoke-d-0e12efc9
CONTEXT=ck8s-stg
NAMESPACE=video-proc

test "$(kubectl config current-context)" = "$CONTEXT"
test "$(kubectl --context "$CONTEXT" create --dry-run=client -f "$MANIFEST" -o jsonpath='{.metadata.namespace}')" = "$NAMESPACE"
test "$(kubectl --context "$CONTEXT" -n "$NAMESPACE" get pod "$POD" --ignore-not-found -o name)" = ""
kubectl --context "$CONTEXT" -n "$NAMESPACE" create --dry-run=server -f "$MANIFEST" -o name
```

Only after all four guards pass and the staging cluster write has been
authorized, create the Pod with cleanup installed before the write:

```bash
set -euo pipefail
cleanup() {
  kubectl --context "$CONTEXT" -n "$NAMESPACE" delete pod "$POD" --ignore-not-found --wait=true
}
trap cleanup EXIT INT TERM
kubectl --context "$CONTEXT" -n "$NAMESPACE" create -f "$MANIFEST"
kubectl --context "$CONTEXT" -n "$NAMESPACE" wait pod "$POD" --for=jsonpath='{.status.phase}'=Succeeded --timeout=5m
kubectl --context "$CONTEXT" -n "$NAMESPACE" logs "$POD"
kubectl --context "$CONTEXT" -n "$NAMESPACE" get pod "$POD" -o json
test "$(kubectl --context "$CONTEXT" -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.status.containerStatuses[0].restartCount}')" = "0"
test "$(kubectl --context "$CONTEXT" -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.status.containerStatuses[0].imageID}')" = "$(kubectl --context "$CONTEXT" -n "$NAMESPACE" get -f "$MANIFEST" -o jsonpath='{.spec.containers[0].image}')"
cleanup
trap - EXIT INT TERM
```

Preserve the logs and terminal Pod JSON as evidence before cleanup. Expected
logs include `execution_mode=process`, `start_method=spawn`, distinct positive
`supervisor_pid` and `child_pid`, `child_exitcode=0`,
`cuda_available=True`, and an L40S device. E/D must report `ingest_mode=pyav`;
F must report `ingest_mode=gstreamer_cuda` and
`producer=GstreamerCudaVideoFrameProducer`.

Repeat with the E and F manifests and their exact names. A passing image smoke
does not authorize patching the ready-pool Deployment or running API workloads.
