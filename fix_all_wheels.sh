#!/bin/bash
# Fix ALL instances where wheels are built (lines 160-177 are duplicates)

# Line 160-162
sed -i '160,162s/.*/RUN python -m pip install --break-system-packages wheel twine requests \&\& \\\
    rm -f dist\/\* \&\& \\\
    python .release\/pypi\/inference.core.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.gpu.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.cli.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.sdk.setup.py bdist_wheel/' docker/dockerfiles/Dockerfile.onnx.gpu

# Line 165-167  
sed -i '165,167s/.*/RUN python -m pip install --break-system-packages wheel twine requests \&\& \\\
    rm -f dist\/\* \&\& \\\
    python .release\/pypi\/inference.core.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.gpu.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.cli.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.sdk.setup.py bdist_wheel/' docker/dockerfiles/Dockerfile.onnx.gpu

# Line 170-172
sed -i '170,172s/.*/RUN python -m pip install --break-system-packages wheel twine requests \&\& \\\
    rm -f dist\/\* \&\& \\\
    python .release\/pypi\/inference.core.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.gpu.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.cli.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.sdk.setup.py bdist_wheel/' docker/dockerfiles/Dockerfile.onnx.gpu

# Line 175-177
sed -i '175,177s/.*/RUN python -m pip install --break-system-packages wheel twine requests \&\& \\\
    rm -f dist\/\* \&\& \\\
    python .release\/pypi\/inference.core.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.gpu.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.cli.setup.py bdist_wheel \&\& \\\
    python .release\/pypi\/inference.sdk.setup.py bdist_wheel/' docker/dockerfiles/Dockerfile.onnx.gpu
