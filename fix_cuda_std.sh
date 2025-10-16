#!/bin/bash
# Update the CCCL installation to properly include cuda/std headers
sed -i '59,66d' docker/dockerfiles/Dockerfile.onnx.gpu

# Insert the corrected CCCL installation with cuda/std headers
sed -i '58a\
# Fix CUDA 13.0 missing CCCL headers - include cuda/std headers\
RUN git clone --depth=1 --branch v3.0.0 https://github.com/NVIDIA/cccl.git /tmp/cccl \&\& \\\
    mkdir -p /usr/local/cuda-13.0/targets/sbsa-linux/include/cccl \&\& \\\
    mkdir -p /usr/local/cuda-13.0/targets/sbsa-linux/include/cuda \&\& \\\
    cp -r /tmp/cccl/libcudacxx/include/* /usr/local/cuda-13.0/targets/sbsa-linux/include/ \&\& \\\
    cp -r /tmp/cccl/cub/cub /usr/local/cuda-13.0/targets/sbsa-linux/include/cccl/ \&\& \\\
    cp -r /tmp/cccl/thrust/thrust /usr/local/cuda-13.0/targets/sbsa-linux/include/cccl/ \&\& \\\
    ln -sf /usr/local/cuda-13.0/targets/sbsa-linux /usr/local/cuda/targets/sbsa-linux \&\& \\\
    rm -rf /tmp/cccl\
' docker/dockerfiles/Dockerfile.onnx.gpu
