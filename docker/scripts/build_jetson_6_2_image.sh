#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_directory}/../.." && pwd)"

"${script_directory}/fetch_jetson_6_2_tensorrt.sh"

cd -- "${repository_root}"
exec docker buildx build \
    --platform linux/arm64 \
    --file docker/dockerfiles/Dockerfile.onnx.jetson.6.2.0 \
    "$@" \
    .
