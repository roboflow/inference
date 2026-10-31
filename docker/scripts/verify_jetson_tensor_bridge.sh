#!/bin/sh

set -eu

BRIDGE_LIBRARY="${JETSON_TENSOR_BRIDGE_LIBRARY:-/opt/roboflow/lib/libroboflow_jetson_tensor.so.1}"

test -s "${BRIDGE_LIBRARY}"
readelf -Ws "${BRIDGE_LIBRARY}" | grep -q 'rf_jetson_pipeline_create'
readelf -Ws "${BRIDGE_LIBRARY}" | grep -q 'rf_jetson_pipeline_retrieve'
readelf -Ws "${BRIDGE_LIBRARY}" | grep -q 'rf_jetson_pipeline_get_stats'
readelf -Ws "${BRIDGE_LIBRARY}" | grep -q 'rf_jetson_pipeline_interrupt'

if readelf -d "${BRIDGE_LIBRARY}" | grep -Eq 'libopencv|libtorch|libpython'; then
    exit 1
fi

missing="$(ldd "${BRIDGE_LIBRARY}" 2>/dev/null | awk '/not found/ { print $1 }' | sort -u)"
case "${missing}" in
    ''|libcuda.so.1) ;;
    *) printf '%s\n' "${missing}" >&2; exit 1 ;;
esac
