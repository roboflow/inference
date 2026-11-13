FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends make git \
    && rm -rf /var/lib/apt/lists/*
# Match the existing setuptools build path; do not install runtime packages.
COPY build-requirements.txt /build-requirements.txt
RUN python -m pip install --no-cache-dir \
    -r /build-requirements.txt click==8.1.8 \
    pytest==8.3.5 pyyaml==6.0.2 black==25.1.0 isort==6.0.1
# Exercise real isolated uv builds without network access in contract tests.
RUN python -m pip download --no-deps --only-binary=:all: \
    --dest /build-wheels -r /build-requirements.txt
COPY release.py /release.py
COPY dependencies.py /dependencies.py
COPY publish.py /publish.py
WORKDIR /work
