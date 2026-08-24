#!/usr/bin/env bash
# Cloud Agent install script for Roboflow Inference.
#
# Idempotent bootstrap that installs the system libraries needed by OpenCV /
# video handling and then installs the repository in editable mode into the
# system Python interpreter (mirroring the project's own Docker images, so
# `python3`, `pytest` and the `inference` CLI are available in any shell without
# activating a virtualenv).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- System libraries -------------------------------------------------------
# Runtime shared libraries required by opencv-python / scikit-image / ffmpeg.
# Guarded so the script stays fast and idempotent when they are already present
# (e.g. when booting from a prebuilt environment snapshot).
if command -v apt-get >/dev/null 2>&1; then
    APT_PACKAGES=(
        python3-venv
        python3-dev
        build-essential
        libgl1
        libglib2.0-0
        libsm6
        libxext6
        libxrender1
        ffmpeg
    )
    MISSING=()
    for pkg in "${APT_PACKAGES[@]}"; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            MISSING+=("$pkg")
        fi
    done
    if [ "${#MISSING[@]}" -gt 0 ]; then
        sudo apt-get update -qq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${MISSING[@]}"
    fi
fi

# --- Python dependencies ----------------------------------------------------
PIP="python3 -m pip"
PIP_FLAGS="--break-system-packages"

$PIP install $PIP_FLAGS --upgrade pip "wheel>=0.38.1,<=0.45.1" "setuptools>=83.0.0"

# Editable install of the full development stack (server, CLI, SDK, workflows,
# torch-cpu based models). Mirrors the `pip install -e .` flow documented in
# CONTRIBUTING.md / AGENTS.md.
$PIP install $PIP_FLAGS -e .

echo "inference install complete: $(python3 -c 'import inference; print("import OK")')"
