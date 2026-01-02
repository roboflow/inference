#!/bin/bash
# Quick script to rebuild workflow block documentation after making changes

set -e

echo "Rebuilding workflow block documentation..."

# Use the inference Python environment (same one the server uses)
PYTHON_ENV="/Users/patricknihranz/.pyenv/versions/3.12.12/envs/inference/bin/python"

# Check if Python exists, otherwise try to find it
if [ ! -f "$PYTHON_ENV" ]; then
    echo "Warning: Expected Python at $PYTHON_ENV not found"
    echo "Trying to find Python in current environment..."
    PYTHON_ENV=$(which python3)
    echo "Using: $PYTHON_ENV"
fi

# Install jinja2 if needed (silently)
$PYTHON_ENV -m pip install -q jinja2 2>/dev/null || true

# Rebuild the docs
echo "Running build script..."
$PYTHON_ENV -m development.docs.build_block_docs

echo ""
echo "✓ Documentation rebuilt successfully!"
echo ""
echo "If you're viewing docs locally, refresh your browser or restart the docs server."

