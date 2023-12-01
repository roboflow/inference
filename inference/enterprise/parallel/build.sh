#!/bin/bash
# run from root of inference like ./inference/enterpise/parallel/build.sh
docker build --pull -t roboflow/roboflow-inference-server-gpu-parallel:latest -f inference/enterprise/parallel/Dockerfile.onnx.gpu.parallel .