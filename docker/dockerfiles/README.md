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

The `roboflow/l46-ml` image is based on the `l4t-ml` image from the [jetson-containers](https://github.com/dusty-nv/jetson-containers/tree/master/packages/l4t/l4t-ml) repository. The image is built on a Jetson with support for GPU acceleration using common ML tools.

To build the image, run the following command:

```bash
jetson-containers build l4t-ml
```

This requires that you have the `jetson-containers` tool installed on your system. Follow the instructions in the [jetson-containers](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md) repository to install the tool.
