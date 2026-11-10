FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends make git \
    && rm -rf /var/lib/apt/lists/*
# Match the existing setuptools build path; do not install runtime packages.
RUN python -m pip install --no-cache-dir \
    setuptools==83.0.0 wheel==0.45.1 packaging==25.0 click==8.1.8 \
    pytest==8.3.5 pyyaml==6.0.2 black==25.1.0 isort==6.0.1
COPY release.py /release.py
COPY dependencies.py /dependencies.py
WORKDIR /work
