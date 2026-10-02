#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_directory}/../.." && pwd)"

tensorrt_deb_name="nv-tensorrt-local-tegra-repo-ubuntu2204-10.7.0-cuda-12.6_1.0-1_arm64.deb"
tensorrt_deb_url="${TENSORRT_DEB_URL:-https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/local_repo/${tensorrt_deb_name}}"
tensorrt_deb_sha256="${TENSORRT_DEB_SHA256:-cf8bd26b3b9c0f65ee8f3358bbc7abfcd4bbed4940b2220140fb108f0852e5c0}"
tensorrt_deb_destination="${TENSORRT_DEB_DESTINATION:-${repository_root}/docker/vendor/tensorrt/${tensorrt_deb_name}}"

verify_sha256() {
    local file_path="$1"

    if command -v sha256sum >/dev/null 2>&1; then
        echo "${tensorrt_deb_sha256}  ${file_path}" | sha256sum --check -
        return
    fi
    if command -v shasum >/dev/null 2>&1; then
        echo "${tensorrt_deb_sha256}  ${file_path}" | shasum --algorithm 256 --check
        return
    fi
    echo "Neither sha256sum nor shasum is available; cannot verify TensorRT." >&2
    return 1
}

mkdir -p -- "$(dirname -- "${tensorrt_deb_destination}")"

if [[ -f "${tensorrt_deb_destination}" ]] &&
    verify_sha256 "${tensorrt_deb_destination}"; then
    echo "Using verified TensorRT package at ${tensorrt_deb_destination}"
    exit 0
fi

temporary_file="$(mktemp "${tensorrt_deb_destination}.part.XXXXXX")"
cleanup() {
    rm -f -- "${temporary_file}"
}
trap cleanup EXIT

curl \
    --fail \
    --location \
    --retry 10 \
    --retry-all-errors \
    --continue-at - \
    --connect-timeout 30 \
    --max-time 600 \
    --output "${temporary_file}" \
    "${tensorrt_deb_url}"
verify_sha256 "${temporary_file}"
mv -f -- "${temporary_file}" "${tensorrt_deb_destination}"
trap - EXIT

echo "Fetched and verified TensorRT package at ${tensorrt_deb_destination}"
