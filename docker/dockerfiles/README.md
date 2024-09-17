This folder contains the dockerfiles used to build and publish the official roboflow inference server docker images with the naming convention `roboflow/roboflow-inference-server-...` (and some peripheral images).

Some dockerfiles include roboflow enterprise code and require a license. See [the enterprise license](https://github.com/roboflow/inference/blob/main/inference/enterprise/LICENSE.txt) for more information.

| Dockerfile | Dockerhub Repository | Enterprise License Required |
| --- | --- | --- |
Dockerfile.device_manager | [roboflow/roboflow-device-manager](https://hub.docker.com/repository/docker/roboflow/roboflow-device-manager/general) | Yes |
Dockerfile.onnx.cpu | [roboflow/roboflow-inference-server-cpu](https://hub.docker.com/repository/docker/roboflow/roboflow-inference-server-cpu/general) | No |
Dockerfile.onnx.gpu | [roboflow/roboflow-inference-server-gpu](https://hub.docker.com/repository/docker/roboflow/roboflow-inference-server-gpu/general) | No |
Dockerfile.onnx.jetson.4.5.0 | [roboflow/roboflow-inference-server-jetson-4.5.0](https://hub.docker.com/repository/docker/roboflow/roboflow-inference-server-jetson-4.5.0/general) | No |
Dockerfile.onnx.jetson.4.6.1 | [roboflow/roboflow-inference-server-jetson-4.6.1](https://hub.docker.com/repository/docker/roboflow/roboflow-inference-server-jetson-4.6.1/general) | No |
Dockerfile.onnx.jetson.5.1.0 | [roboflow/roboflow-inference-server-jetson-5.1.0](https://hub.docker.com/repository/docker/roboflow/roboflow-inference-server-jetson-5.1.0/general) | No |
Dockerfile.onnx.lambda | Not Published | Yes |
Dockerfile.onnx.trt | Deprecated | No |
Dockerfile.onnx.trt.base | Deprecated | No |
Dockerfile.onnx.udp.gpu | Deprecated | No |

