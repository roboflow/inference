#!/bin/bash
# Replace the Makefile command with direct commands that use --break-system-packages
sed -i '152s/.*/# Build wheels directly without Makefile to handle PEP 668\
RUN python -m pip install --system --break-system-packages --upgrade pip \&\& \\\
    python -m pip install --system --break-system-packages wheel twine requests \&\& \\\
    rm -f dist\/* \&\& \\\
    python .release\/pypi\/inference.core.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.gpu.setup.py bdist_wheel/' docker/dockerfiles/Dockerfile.onnx.gpu
