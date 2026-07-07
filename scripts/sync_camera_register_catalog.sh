#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:-}"
if [[ -z "${SOURCE}" ]]; then
    echo "Usage: $0 /path/to/roboflow/packages/shared/device/cameraRegisterCatalog.json" >&2
    exit 1
fi

TARGET="$(cd "$(dirname "$0")/.." && pwd)/inference/enterprise/workflows/edge_camera_parameters_client/camera_register_catalog.json"
cp "${SOURCE}" "${TARGET}"
echo "Synced camera register catalog to ${TARGET}"
