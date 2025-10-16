#!/bin/bash
# Update line 153-157 to build ALL wheels including CLI and SDK
sed -i '153,157s/.*/# Build wheels directly without upgrading pip (Debian-installed pip issue)\
RUN python -m pip install --break-system-packages wheel twine requests \&\& \\\
    rm -f dist\/\* \&\& \\\
    python .release\/pypi\/inference.core.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.gpu.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.cli.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.sdk.setup.py bdist_wheel/' docker/dockerfiles/Dockerfile.onnx.gpu

# Also restore the installation of CLI and SDK wheels on lines 183-186
sed -i '183,186s/.*/# Install all the built wheels with dependency resolution\
RUN python -m pip install --break-system-packages \\\
    dist\/inference_core\*.whl \\\
    dist\/inference_cli\*.whl \\\
    dist\/inference_sdk\*.whl \\\
    "setuptools\<\=75.5.0"/' docker/dockerfiles/Dockerfile.onnx.gpu
