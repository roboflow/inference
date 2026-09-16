---
template: redirect.html
redirect_url: https://docs.roboflow.com/deployment/self-hosted/inference-server/install/jetson
---

### Strict RF-DETR TensorRT qualification

To reject compatibility or runtime fallback in a deployed RF-DETR pipeline,
set both `INFERENCE_MODELS_RFDETR_ALLOW_COMPATIBILITY_FALLBACK=false` and
`INFERENCE_MODELS_RFDETR_ALLOW_RUNTIME_FAILURE_FALLBACK=false`. These controls
apply when no explicit `RFDetrExecutionPlan` is provided; an explicit plan takes
precedence. Both default to `true` for backwards compatibility. Select the GPU
stages with `INFERENCE_MODELS_RFDETR_PREPROCESSOR=triton-universal-v1` and
`INFERENCE_MODELS_RFDETR_POSTPROCESSOR=triton-fused-v1`. Backend selection is
separate: confirm that the loaded model is native TensorRT, and verify the
resolved plan and CUDA frame representation in the running pipeline.

For opt-in qualification, `INFERENCE_MODELS_RUNTIME_DIAGNOSTICS=true` exposes
`RFDetrForObjectDetectionTRT.last_inference_diagnostics`: the most recently
completed call's timestamp, tensor devices for each stage, and actual stage
selections captured in the inference thread. It contains no image pixels and
performs no tensor copies. Concurrent calls publish complete snapshots; it is a
latest-call diagnostic, not per-stream attribution or a performance counter.
It is disabled by default.
