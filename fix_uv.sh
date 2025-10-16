#!/bin/bash
# Restore uv pip in builder stage (lines 1-121)
# Keep python -m pip in runtime stage (lines 122+)

# Line 37 - builder stage, should use uv
sed -i '37s/.*/RUN uv pip install --system --break-system-packages \\/' docker/dockerfiles/Dockerfile.onnx.gpu

# Line 96 - builder stage, should use uv
sed -i '96s/.*/RUN uv pip install --system --break-system-packages \/tmp\/onnxruntime\/build\/cuda13\/Release\/dist\/onnxruntime_gpu-\*.whl/' docker/dockerfiles/Dockerfile.onnx.gpu

# Line 100 - builder stage, should use uv
sed -i '100s/python -m pip install/uv pip install --system/' docker/dockerfiles/Dockerfile.onnx.gpu

# Line 108 - builder stage, should use uv
sed -i '108s/python -m pip install/uv pip install --system/' docker/dockerfiles/Dockerfile.onnx.gpu

# Line 117 - builder stage, should use uv
sed -i '117s/python -m pip install/uv pip install --system/' docker/dockerfiles/Dockerfile.onnx.gpu

# Lines 180 and 183 are in runtime stage - keep as python -m pip
