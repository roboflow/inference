#!/bin/sh

set -eu

BRIDGE_LIBRARY="${GSTREAMER_CUDA_TENSOR_BRIDGE_LIBRARY:-/opt/roboflow/lib/libroboflow_gstreamer_cuda_tensor.so.1}"

test -s "${BRIDGE_LIBRARY}"
readelf -Ws "${BRIDGE_LIBRARY}" | grep -q 'rf_gstreamer_cuda_pipeline_create'
readelf -Ws "${BRIDGE_LIBRARY}" | grep -q 'rf_gstreamer_cuda_pipeline_retrieve'
readelf -Ws "${BRIDGE_LIBRARY}" | grep -q 'rf_gstreamer_cuda_pipeline_get_stats'
readelf -Ws "${BRIDGE_LIBRARY}" | grep -q 'rf_gstreamer_cuda_pipeline_interrupt'

if readelf -d "${BRIDGE_LIBRARY}" | grep -Eq 'libopencv|libtorch|libpython'; then
    exit 1
fi

missing="$(ldd "${BRIDGE_LIBRARY}" 2>/dev/null | awk '/not found/ { print $1 }' | sort -u)"
case "${missing}" in
    ''|libcuda.so.1) ;;
    *) printf '%s\n' "${missing}" >&2; exit 1 ;;
esac
