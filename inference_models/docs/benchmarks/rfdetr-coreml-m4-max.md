# RF-DETR CoreML measurements on M4 Max

On October 1, 2026, RF-DETR Nano, Small and Large were measured locally with the native CoreML backend. `CPUAndGPU` had the lowest model-load and warmed-inference median, p95 and p99 for all three models on this machine. Instruments subsequently showed that none of these packages ran entirely on the Apple Neural Engine under either `CPUAndNeuralEngine` or `ALL`: the former executed CPU graph work, and the latter also used the GPU.

The benchmark ran [inference PR #3100](https://github.com/roboflow/inference/pull/3100) locally. The measured inference commit is `ee762b8072d35a4cfa62f17c222f11a0bb7ff446`. This report records a baseline for follow-up investigation; it does not change backend selection or compute-unit defaults.

## Environment and artifacts

- Apple M4 Max, 64 GiB unified memory, arm64, macOS 27.0.1 (26A434).
- Python 3.12.12, coremltools 9.0, PyTorch 2.8.0, torchvision 0.23.0, NumPy 1.26.4, Pillow 12.3.0, psutil 7.0.0.
- Xcode and Instruments 27.0 (27A266a) for the separate execution captures.
- Native RF-DETR FP16 object-detection packages, batch one. Nano input 384×384; Small 512×512; Large 704×704.
- Existing model artifacts were loaded locally; no model conversion was performed for this experiment. Package identity and SHA-256 hashes, inference source hashes, and all 200 image hashes are in the [measurement data](rfdetr-coreml-m4-max-2026-10-01.json).
- Profiling used the CoreML extension in [inference-profiler PR #44](https://github.com/roboflow/inference-profiler/pull/44). The measurement checkout was profiler commit `4328b4305f473b2b2cfbfdeab53b2b0299c3ec5e` plus that extension. The PR ports it to current `main`; the original measurements were not rerun after that port.

## Load and warmed inference

Values are milliseconds. Percentiles use linear interpolation over the pooled samples for each configuration.

| Model | Compute units | Load median / p95 / p99 (ms) | Inference median / p95 / p99 (ms) |
| --- | --- | ---: | ---: |
| Nano | `CPUAndGPU` | 318.16 / 367.06 / 395.06 | 10.12 / 11.20 / 11.77 |
| Nano | `CPUAndNeuralEngine` | 3369.06 / 3506.93 / 3572.63 | 11.06 / 11.78 / 12.44 |
| Nano | `ALL` | 2649.27 / 2691.09 / 2720.57 | 11.71 / 14.04 / 16.29 |
| Small | `CPUAndGPU` | 367.93 / 381.49 / 381.91 | 12.14 / 12.66 / 13.34 |
| Small | `CPUAndNeuralEngine` | 3925.42 / 4034.91 / 4061.89 | 21.22 / 22.07 / 22.68 |
| Small | `ALL` | 2696.30 / 2733.56 / 2841.59 | 22.55 / 26.66 / 28.05 |
| Large | `CPUAndGPU` | 394.84 / 420.15 / 421.90 | 19.26 / 19.81 / 20.28 |
| Large | `CPUAndNeuralEngine` | 4531.80 / 4584.12 / 4646.64 | 54.26 / 55.39 / 56.64 |
| Large | `ALL` | 3010.83 / 3119.42 / 3127.49 | 55.93 / 58.76 / 60.11 |

### Dataset and timing assumptions

1. Use COCO val2017, selecting the first 200 filenames in lexicographic order from the complete 5,000-image set. All configurations use the same selection and order. This is a deterministic subset, not a random sample or accuracy evaluation.
2. Decode with Pillow to RGB and resize to a fixed 640×480 camera scenario before timing. Pass `input_color_format="rgb"`. Each model then performs its own preprocessing to its native resolution. The source-image resize may alter aspect ratio; it is the same workload for every compute-unit setting.
3. Use batch one, confidence 0.4, IoU threshold 0.5 and maximum 300 detections. Preload inputs outside the measured window.
4. Prime each configuration in a separate excluded run. Run four rounds, each with five fresh worker processes per configuration; rotate/reverse configuration order between rounds. Each worker executes 30 warmup predictions and 200 measured predictions, restarting at the first image after warmup. This yields 20 load samples and 4,000 inference samples per configuration.
5. Measure load around `from_pretrained`, after interpreter/library imports. Package and OS caches are not cleared. These are cache-warm model-load measurements, not cold process startup or cold compilation. With only 20 load samples, p95/p99 estimates have limited statistical support.
6. Measure wall-clock inference including model preprocessing, synchronous CoreML prediction and postprocessing. Exclude disk reads, image decode and the initial camera-scenario resize. Latency runs do not run a memory sampler or Instruments.
7. Read the configured compute-unit value from the loaded `MLModel`. This verifies configuration only; actual execution is investigated separately below.

These results apply to the recorded model artifacts, source commit and M4 Max software stack. They do not establish results for other Apple chips, macOS releases, model exports, batch sizes, thermal conditions or concurrent workloads. No controlled power or energy comparison was performed. The measurements establish latency differences but do not isolate their cause.

## Process memory

Peak process RSS, taking the maximum of three fresh worker repetitions per configuration. Each repetition uses the same 200-image workload and 30 warmups, followed by 200 measured predictions with 10 ms RSS sampling.

| Model | CPUAndGPU (MiB) | CPUAndNeuralEngine (MiB) | All (MiB) |
| --- | ---: | ---: | ---: |
| Nano | 770.5 | 661.6 | 699.8 |
| Small | 777.8 | 673.6 | 716.9 |
| Large | 810.4 | 700.3 | 749.2 |

RSS includes predecoded image arrays and excludes memory held by other CoreML service processes. It is not total unified-memory consumption or GPU/ANE allocator memory, and sampling can miss short peaks. Accelerator admission metrics are explicitly unavailable. All 27 memory repetitions completed successfully.

## Executed allocation evidence

After the uninstrumented measurements, nine separate Instruments captures used the Core AI template plus the Core ML instrument. Each capture ran 30 warmups followed by 100 predictions over the first five images from the same COCO selection. Only the warmed prediction window was analyzed, trimming 2 ms at both edges because the exported trace start time has millisecond precision.

CPU samples and GPU events were filtered to the launched worker PID. ANE events have no PID column; attribution uses the unique `weights_<UUID>...Prediction` label during the isolated model run. Compilation events are excluded. GPU activity from unrelated applications is excluded.

| Model | Compute units | ANE prediction intervals | Process GPU intervals | CPU graph samples |
| --- | --- | ---: | ---: | ---: |
| Nano | `CPUAndGPU` | 0 | 598 | 0 |
| Nano | `CPUAndNeuralEngine` | 595 | 0 | 17 |
| Nano | `ALL` | 100 | 202 | 0 |
| Small | `CPUAndGPU` | 0 | 805 | 0 |
| Small | `CPUAndNeuralEngine` | 795 | 0 | 22 |
| Small | `ALL` | 100 | 410 | 0 |
| Large | `CPUAndGPU` | 0 | 1103 | 0 |
| Large | `CPUAndNeuralEngine` | 997 | 0 | 17 |
| Large | `ALL` | 100 | 402 | 0 |

- `CPUAndGPU`: process GPU activity and no ANE prediction events for each model.
- `CPUAndNeuralEngine`: ANE activity plus sampled `Espresso::topk_kernel_cpu` and `BnnsCpuInferenceOperation::ExecuteSync` calls for every model. Small also has `gather_nd_kernel_cpu`. The saved stacks place these calls beneath CoreML prediction, establishing CPU graph execution separately from Python preprocessing and postprocessing.
- `ALL`: both ANE and process GPU activity for each model.

This is positive evidence of executed work. It is not a complete per-node allocation map: the Core ML per-operation signpost table was empty in all nine captures. Zero sampled CPU graph kernels does not establish zero CPU graph work. Hardware event counts are not counts of graph nodes, and sampled CPU counts are not percentages of execution. Instrumented timings are excluded from the performance table. Requested compute-unit settings or an estimated compute plan alone would not establish these findings.

The [measurement data](rfdetr-coreml-m4-max-2026-10-01.json) contains event counts, CPU symbols and complete example stacks. Raw profile records and native `.trace` bundles remain local and are not committed; consequently the checked-in summaries cannot independently reconstruct every percentile or trace event.

## Reproducing the measurements

Use the profiler CoreML PR with an inference-models checkout at the measured commit and the native CoreML dependencies. Install the profiler's `coreml` extra. Keep the inference-models checkout selected explicitly so dependency synchronization cannot silently replace it with a release lacking native CoreML support.

Each downloaded package needs a `profiling-package.json` manifest as documented by the profiler. Use the original artifact hashes to distinguish rerunning these packages from testing a later export. Set compute units before starting Python because inference-models reads the environment variable at import time.

```bash
INFERENCE_MODELS_COREML_COMPUTE_UNITS=CPUAndGPU inference-profiler \
  --build-latency-profile --model-id rfdetr-nano \
  --local-package-dir /path/to/rfdetr-nano --device cpu \
  --profiling-scenarios camera_640x480_batch_1_base \
  --coco-validation-images-dir /path/to/coco/val2017 --coco-image-count 200 \
  --infer-kwargs-json '{"input_color_format":"rgb"}' \
  --warmup 30 --measured 200 --repetitions 5 --output-json latency-round1.json
```

Repeat for Nano, Small and Large with `CPUAndGPU`, `CPUAndNeuralEngine` and `ALL`. Repeat the full matrix for four rounds with rotated/reversed ordering; retain and pool all four rounds. Prime configurations separately first and exclude those runs. For memory, replace `--build-latency-profile` with `--build-memory-profile` and use three repetitions.

For executed allocation, launch the same native model workload under `xcrun xctrace record --template 'Core AI' --instrument 'Core ML'`, record the warmed prediction window, and inspect ANE hardware intervals, process-filtered GPU intervals and Time Profiler stacks. A complete per-node executed allocation map remains a follow-up investigation.

## Validation

The original profiler implementation passed 179 tests. After porting onto current profiler `main`, 264 tests passed; native Nano `CPUAndGPU` smoke checks exercised both latency and memory commands. All nine original benchmark configurations and all nine Instruments captures completed. No CUDA hardware validation was performed. This report changes documentation and adds derived data only; no inference runtime tests were needed for this PR.
