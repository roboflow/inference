#!/bin/sh

set -eu

SOURCE_DIR="${JETSON_TENSOR_BRIDGE_SOURCE_DIR:-/src/jetson_tensor_bridge}"
OUTPUT_DIR="${JETSON_TENSOR_BRIDGE_OUTPUT_DIR:-/opt/roboflow/lib}"
CUDA_ARCHITECTURES="${JETSON_TENSOR_BRIDGE_CUDA_ARCHITECTURES:-87}"
NVBUF_SURFACE_INCLUDE_DIR="${NVBUF_SURFACE_INCLUDE_DIR:-/usr/src/jetson_multimedia_api/include}"
CUDA_STUB_LIBRARY_DIR="${CUDA_STUB_LIBRARY_DIR:-$(
    find /usr/local/cuda/targets -path '*/lib/stubs/libcuda.so' -print -quit |
        sed 's|/libcuda.so$||'
)}"

test -f "${SOURCE_DIR}/jetson_tensor_bridge.cu"
test -f "${NVBUF_SURFACE_INCLUDE_DIR}/nvbufsurface.h"
test -f "${CUDA_STUB_LIBRARY_DIR}/libcuda.so"
pkg-config --exists gstreamer-app-1.0

mkdir -p "${OUTPUT_DIR}"

gencode_flags=""
old_ifs="${IFS}"
IFS=';, '
for architecture in ${CUDA_ARCHITECTURES}; do
    test -n "${architecture}"
    gencode_flags="${gencode_flags} -gencode arch=compute_${architecture},code=sm_${architecture}"
done
IFS="${old_ifs}"

# shellcheck disable=SC2086
nvcc \
    -std=c++17 \
    -O3 \
    --shared \
    --compiler-options=-fPIC,-fvisibility=hidden,-pthread \
    ${gencode_flags} \
    -I"${NVBUF_SURFACE_INCLUDE_DIR}" \
    $(pkg-config --cflags-only-I gstreamer-app-1.0) \
    "${SOURCE_DIR}/jetson_tensor_bridge.cu" \
    -o "${OUTPUT_DIR}/libroboflow_jetson_tensor.so.1" \
    $(pkg-config --libs-only-L --libs-only-l gstreamer-app-1.0) \
    -L"${CUDA_STUB_LIBRARY_DIR}" \
    -Xlinker=-rpath \
    -Xlinker=/opt/gstreamer/lib:/usr/local/cuda/lib64 \
    -lcuda \
    -lcudart \
    -ldl

old_ifs="${IFS}"
IFS=';, '
for architecture in ${CUDA_ARCHITECTURES}; do
    cuobjdump --list-elf \
        "${OUTPUT_DIR}/libroboflow_jetson_tensor.so.1" |
        grep -q "\.sm_${architecture}\.cubin$"
done
IFS="${old_ifs}"

strip --strip-unneeded "${OUTPUT_DIR}/libroboflow_jetson_tensor.so.1"
ln -sf libroboflow_jetson_tensor.so.1 \
    "${OUTPUT_DIR}/libroboflow_jetson_tensor.so"
