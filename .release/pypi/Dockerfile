FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends make git jq \
    && rm -rf /var/lib/apt/lists/*
# Match the existing setuptools build path; do not install runtime packages.
COPY build-requirements.txt /build-requirements.txt
RUN python -m pip install --no-cache-dir \
    -r /build-requirements.txt click==8.1.8 \
    pytest==8.3.5 pyyaml==6.0.2 black==25.1.0 isort==6.0.1
# Exercise real isolated uv builds without network access in contract tests.
RUN python -m pip download --no-deps --only-binary=:all: \
    --dest /build-wheels -r /build-requirements.txt
# Competing versions make removal of a pin observable in offline tests.
RUN python -m pip download --no-deps --only-binary=:all: --dest /build-wheels \
    uv==0.13.9 setuptools==84.2.1 wheel==0.48.0 packaging==26.4 \
    && python -m pip download --no-deps --only-binary=:all: --dest /build-wheels \
    wheel==0.45.1 packaging==24.2
COPY release.py /release.py
COPY dependencies.py /dependencies.py
COPY publish.py /publish.py
COPY candidate.py /candidate.py
COPY prepare.py /prepare.py
COPY image_inputs.py /image_inputs.py
WORKDIR /work
