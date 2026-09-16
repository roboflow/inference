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

The `roboflow/l46-ml` image is based on the `l4t-ml` image from the [jetson-containers](https://github.com/dusty-nv/jetson-containers/tree/master/packages/l4t/l4t-ml) repository. The image is built on a Jetson with support for GPU acceleration using common ML tools.

To build the image, run the following command:

```bash
jetson-containers build l4t-ml
```

This requires that you have the `jetson-containers` tool installed on your system. Follow the instructions in the [jetson-containers](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md) repository to install the tool.
