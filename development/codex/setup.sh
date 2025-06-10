#!/usr/bin/env bash
# Fast "uv"-based setup for Roboflow Inference on Ubuntu 24.04

# --------------------------
# System prerequisites
# --------------------------
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake git curl \
  libopencv-dev libxext6 libgdal-dev libvips-dev

# --------------------------
# Install uv (Rust binary)
# --------------------------
curl -LsSf https://astral.sh/uv/install.sh | sh      # installs to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

# core + extras
uv pip install --system --prerelease allow -e .
