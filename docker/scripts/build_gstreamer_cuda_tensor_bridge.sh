#!/bin/sh

set -eu

SOURCE_DIR="${GSTREAMER_CUDA_TENSOR_BRIDGE_SOURCE_DIR:-/src/gstreamer_cuda_tensor_bridge}"
OUTPUT_DIR="${GSTREAMER_CUDA_TENSOR_BRIDGE_OUTPUT_DIR:-/opt/roboflow/lib}"
CUDA_INCLUDE_DIR="${CUDA_INCLUDE_DIR:-$(
    find /usr/local/cuda/targets -path '*/include/cuda.h' -print -quit |
        sed 's|/cuda.h$||'
)}"
CUDA_STUB_LIBRARY_DIR="${CUDA_STUB_LIBRARY_DIR:-$(
    find /usr/local/cuda/targets -path '*/lib/stubs/libcuda.so' -print -quit |
        sed 's|/libcuda.so$||'
)}"

test -f "${SOURCE_DIR}/gstreamer_cuda_tensor_bridge.cpp"
test -f "${CUDA_INCLUDE_DIR}/cuda.h"
test -f "${CUDA_STUB_LIBRARY_DIR}/libcuda.so"
pkg-config --exists gstreamer-app-1.0 gstreamer-cuda-1.0 gstreamer-video-1.0
mkdir -p "${OUTPUT_DIR}"

c++ \
    -std=c++17 \
    -O3 \
    -shared \
    -fPIC \
    -fvisibility=hidden \
    -I"${CUDA_INCLUDE_DIR}" \
    $(pkg-config --cflags gstreamer-app-1.0 gstreamer-cuda-1.0 gstreamer-video-1.0) \
    "${SOURCE_DIR}/gstreamer_cuda_tensor_bridge.cpp" \
    -o "${OUTPUT_DIR}/libroboflow_gstreamer_cuda_tensor.so.1" \
    $(pkg-config --libs gstreamer-app-1.0 gstreamer-cuda-1.0 gstreamer-video-1.0) \
    -L"${CUDA_STUB_LIBRARY_DIR}" \
    -Wl,-rpath,/opt/gstreamer/lib:/usr/local/cuda/lib64 \
    -lcuda

strip --strip-unneeded \
    "${OUTPUT_DIR}/libroboflow_gstreamer_cuda_tensor.so.1"
ln -sf libroboflow_gstreamer_cuda_tensor.so.1 \
    "${OUTPUT_DIR}/libroboflow_gstreamer_cuda_tensor.so"
