# YOLOv8n-640 ONNX preprocessing experiment — Jetson 10.36

Status: **rejected; do not merge the implementation in this branch.**

This report records a measured attempt to reduce preprocessing overhead for the
`yolov8n-640` ONNX object-detection package. The optional implementation in
this branch, `torch-gpu-numpy-letterbox-v1`, moves NumPy letterbox processing to
Torch on `cuda:0` for a batch-one input. It preserves the default preprocessing
path and was only selected explicitly for the candidate measurements.

## Scope and method

| Item | Value |
| --- | --- |
| Host | `jetson-10-36` (Jetson Orin, compute capability 8.7) |
| Container baseline | `roboflow/roboflow-inference-server-jetson-6.2.0:1.3.5` |
| Model package | `yolov8n-640` ONNX FP32, package `9e6ce8e20dd9b4daaa0fa9b728fca2f2` |
| Input set | COCO validation, batch 1, 640 × 640 |
| Runtime | ONNX Runtime 1.20, CUDA Execution Provider then CPU fallback |
| Latency protocol | 20 warm-up requests; 100 measured requests; 3 repetitions |
| Baseline source | `083be2ef99d4587d2e6e5745d0ff87b44c98c820` |
| Candidate source | `d9a3fd650b3f244dec3ff35aefe0a3bb1639e75d` |

The acceptance gates were: mean **and** p50 latency improve by at least 3%; p95
and p99 do not regress by more than 2%; peak GPU memory does not increase by
more than 2%; and the YOLOv8-specific ONNX integration test passes.

## Results

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Mean latency | 23.4102 ms | 22.9805 ms | **1.84% faster** |
| p50 latency | 21.5598 ms | 21.3337 ms | **1.05% faster** |
| p95 latency | 32.8945 ms | 32.1777 ms | 2.18% faster |
| p99 latency | 33.5619 ms | 32.7093 ms | 2.54% faster |
| Peak request GPU memory | 791,343,104 B | Not measured | — |
| Model GPU residency | 94,908,416 B | Not measured | — |
| Incremental GPU memory | 699,658,240 B | Not measured | — |

The candidate passed the tail-latency safety gates, but failed both mandatory
central-latency gates: mean improved by 1.84% and p50 by 1.05%, each below the
required 3%. A candidate memory run was intentionally not performed after this
failure, because the implementation was already ineligible for acceptance.

## Validation

The required YOLOv8-specific ONNX integration test passed against the final
candidate source:

```text
/imgtests/tests/integration_tests/models/test_yolov8_object_detection_predictions_onnx.py
50 passed in 47.19s
```

The baseline test also passed (`50 passed in 41.12s`). The test exercises the
default, unchanged preprocessing path; the profiler explicitly selected
`torch-gpu-numpy-letterbox-v1` for candidate measurements.

## Trace evidence

Nsight Systems traces show why this candidate did not clear the latency gate.

| Preprocessing trace metric | Baseline | Candidate |
| --- | ---: | ---: |
| Preprocess NVTX wall time | 2.970 ms/request | 3.534 ms/request |
| Preprocess GPU-kernel time | 0.864 ms/request | 1.091 ms/request |
| Host-to-device copy volume | 4.915 MB/request | 3.272 MB/request |

Although the candidate reduced host-to-device traffic, its GPU interpolation
work made preprocessing slower. `cudaStreamSynchronize` remains necessary for
the ordering contract with ONNX Runtime, so removing it was not a safe
optimization target.

## Run artifacts

The raw JSON measurements and Nsight traces are retained in the profiling
workspace, not committed to this repository. These run IDs locate the complete
artifacts and their generated `profile-results.md` summaries:

| Purpose | Run ID |
| --- | --- |
| Baseline integration test | `model-integration-test-source-20260715-173111` |
| Baseline latency | `inference-profiler-source-20260715-173916` |
| Baseline GPU memory | `inference-profiler-source-20260715-174304` |
| Baseline postprocess trace | `inference-profiler-source-nsys-20260715-174730` |
| Baseline focused preprocess trace | `inference-profiler-source-nsys-20260715-175530` |
| Candidate integration test | `model-integration-test-source-20260715-183134` |
| Candidate latency | `inference-profiler-source-20260715-183346` |
| Candidate trace | `inference-profiler-source-nsys-20260715-183712` |

## Decision

Reject and close this PR without merging. The candidate has profiling evidence
of a small improvement but does not meet the pre-agreed mean and p50 acceptance
thresholds. No model package was registered or uploaded.
