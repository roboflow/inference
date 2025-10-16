#!/bin/bash
# Replace the CCCL installation section in Dockerfile
sed -i '59,63d' docker/dockerfiles/Dockerfile.onnx.gpu

# Insert the corrected CCCL installation
sed -i '58a\
# Fix CUDA 13.0 missing CCCL headers - use CCCL 3.0 for CUDA 13\
# The base image has CUDA at /usr/local/cuda-13.0/targets/sbsa-linux/\
RUN git clone --depth=1 --branch v3.0.0 https://github.com/NVIDIA/cccl.git /tmp/cccl \&\& \\\
    mkdir -p /usr/local/cuda-13.0/targets/sbsa-linux/include/cccl \&\& \\\
    cp -r /tmp/cccl/libcudacxx/include/* /usr/local/cuda-13.0/targets/sbsa-linux/include/cccl/ \&\& \\\
    cp -r /tmp/cccl/cub/cub /usr/local/cuda-13.0/targets/sbsa-linux/include/cccl/ \&\& \\\
    cp -r /tmp/cccl/thrust/thrust /usr/local/cuda-13.0/targets/sbsa-linux/include/cccl/ \&\& \\\
    rm -rf /tmp/cccl\
' docker/dockerfiles/Dockerfile.onnx.gpu
