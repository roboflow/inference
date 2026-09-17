This folder contains the dockerfiles used to build and publish the official roboflow inference server docker images with the naming convention `roboflow/roboflow-inference-server-...` (and some peripheral images).

Some dockerfiles include roboflow enterprise code and require a license. See [the enterprise license](https://github.com/roboflow/inference/blob/main/inference/enterprise/LICENSE.txt) for more information.

| Dockerfile | Dockerhub Repository | Enterprise License Required |
| --- | --- | --- |
Dockerfile.device_manager | [roboflow/roboflow-device-manager](https://hub.docker.com/repository/docker/roboflow/roboflow-device-manager/general) | Yes |
Dockerfile.onnx.cpu | [roboflow/roboflow-inference-server-cpu](https://hub.docker.com/repository/docker/roboflow/roboflow-inference-server-cpu/general) | No |
Dockerfile.onnx.gpu | [roboflow/roboflow-inference-server-gpu](https://hub.docker.com/repository/docker/roboflow/roboflow-inference-server-gpu/general) | No |
Dockerfile.onnx.jetson.4.5.0 **DEPRECATED** | [roboflow/roboflow-inference-server-jetson-4.5.0](https://hub.docker.com/repository/docker/roboflow/roboflow-inference-server-jetson-4.5.0/general) | No |
Dockerfile.onnx.jetson.4.6.1 | [roboflow/roboflow-inference-server-jetson-4.6.1](https://hub.docker.com/repository/docker/roboflow/roboflow-inference-server-jetson-4.6.1/general) | No |
Dockerfile.onnx.jetson.5.1.0 | [roboflow/roboflow-inference-server-jetson-5.1.0](https://hub.docker.com/repository/docker/roboflow/roboflow-inference-server-jetson-5.1.0/general) | No |
Dockerfile.onnx.lambda | Not Published | Yes |
Dockerfile.onnx.trt | Deprecated | No |
Dockerfile.onnx.trt.base | Deprecated | No |
Dockerfile.onnx.udp.gpu | Deprecated | No |


## Base image

### JetPack 7.2 hardware verification

Run the on-device checks inside the built server image:

```bash
docker run --rm --runtime=nvidia --entrypoint python3 \
  roboflow/roboflow-inference-server-jetson-7.2.0:<candidate-tag> \
  /usr/local/bin/verify_jetson_tensor_runtime
```

These checks require a real Jetson GPU. They cover CUDA tensor decoding of
H.264, H.265 and baseline JPEG, repeated frame grabbing, threaded retrieval,
resolution changes, interrupted reads, and torchvision CUDA JPEG decoding
(baseline and progressive). The tensor bridge must not copy decoded pixels
through host memory. Set `ROBOFLOW_JETSON_TEST_RTSP_URL` to additionally test
a live RTSP source and the production `VideoSource` path.

The JP7.2 media image pins nvJPEG **13.2.1.68**, the component shipped in CUDA
13.3 Update 1, while retaining CUDA 13.2 and the JetPack driver. This fixes
NVIDIA issue 6176492 (`nvjpegCreate*` fails on Orin). The archive is verified
against NVIDIA's redistribution SHA-256. See the
[NVIDIA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#nvjpeg-release-13-3-update-1).
Updating only the Python wheel does not fix this runtime-library failure.

The NVIDIA GStreamer JPEG decoder accepts baseline JPEG; use the CUDA
nvJPEG/torchvision path for progressive JPEG. A successful image build or
framework import alone does not qualify hardware decode or TensorRT model
execution; run these device checks and a representative TensorRT model too.

For an RF-DETR package compiled on the target GPU, mount this repository's
`docker/scripts/verify_jetson_rfdetr_trt_runtime.py` into the container and run:

```bash
RUNS_ON_JETSON=true ENABLE_TENSOR_DATA_REPRESENTATION=true \
  python3 /tmp/verify_jetson_rfdetr_trt_runtime.py \
  --model-package /path/to/package --video-source /path/to/video.h264 \
  --frames 10 --expected-class dog
```

The package contains `engine.plan`, `inference_config.json`, `trt_config.json`
and `class_names.txt`. Use a clip containing the expected class, or omit
`--expected-class` for a source without known labels. This test uses the
production `VideoSource`, requires the Jetson producer, disables model-stage
fallback, and verifies CUDA tensors through preprocessing, TensorRT and
postprocessing. Only final class IDs are copied to the CPU for assertions.
Hardware decoder tensors are RGB; pass `input_color_format="rgb"` when using
the model directly. Tensor inputs do not imply RGB in RF-DETR's default
preprocessing contract.

The `roboflow/l46-ml` image is based on the `l4t-ml` image from the [jetson-containers](https://github.com/dusty-nv/jetson-containers/tree/master/packages/l4t/l4t-ml) repository. The image is built on a Jetson with support for GPU acceleration using common ML tools.

To build the image, run the following command:

```bash
jetson-containers build l4t-ml
```

This requires that you have the `jetson-containers` tool installed on your system. Follow the instructions in the [jetson-containers](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md) repository to install the tool.


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
It is disabled by default. It allocates a small metadata snapshot per completed call;
no tensor copies does not mean zero CPU overhead.


Jetson bridge drop counters distinguish layers. `frames_dropped_by_consumer`
counts replacements in the native ready queue. `frames_discarded_before_retrieve`
counts a Python-reserved tensor discarded by the next grab without retrieval,
including source FPS subsampling or adaptive buffering. Source-level drop events
can describe that same discard, so these layers must not be added together.
Repeated `retrieve()` returns the same selected frame; one bounded pooled buffer
remains referenced until the next grab or close, including while paused.
