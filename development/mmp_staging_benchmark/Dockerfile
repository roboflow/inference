# Staging-only MMP experiment image. Build for linux/amd64 from the repository
# root. Base images are pinned to platform-specific manifest digests so the
# runtime does not drift underneath an experiment.

ARG CUDA_BUILDER_IMAGE=nvcr.io/nvidia/cuda@sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4
ARG CUDA_RUNTIME_IMAGE=nvcr.io/nvidia/cuda@sha256:0bb88834d973ca1b450fcc2a05333c6fe45510bee289912a5391274c351c4a4d
ARG UV_IMAGE=ghcr.io/astral-sh/uv@sha256:dfd1e6972e100ca2fbf1f391effc3dd4aa57f319bf03c3e321e0a3f3341ed5af
ARG BENCHMARK_PLATFORM=linux/amd64
ARG PYTHON_VERSION=3.12
ARG PYTHON_DEPENDENCY_CUTOFF=2026-08-12T00:00:00Z
ARG SOURCE_REVISION=unknown

FROM --platform=${BENCHMARK_PLATFORM} ${UV_IMAGE} AS uv

FROM --platform=${BENCHMARK_PLATFORM} ${CUDA_BUILDER_IMAGE} AS builder
ARG PYTHON_VERSION
ARG PYTHON_DEPENDENCY_CUTOFF

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

# Keep the expensive dependency layer independent of source edits. This
# mirrors inference_server/docker/Dockerfile.gpu while fixing all image bases.
WORKDIR /app
COPY inference_models/pyproject.toml        inference_models/README.md        /app/inference_models/
COPY inference_model_manager/pyproject.toml inference_model_manager/README.md /app/inference_model_manager/
COPY inference_server/pyproject.toml        inference_server/README.md        /app/inference_server/
RUN mkdir -p /app/inference_models/inference_models \
             /app/inference_model_manager/inference_model_manager \
             /app/inference_server/inference_server \
    && touch /app/inference_models/inference_models/__init__.py \
             /app/inference_model_manager/inference_model_manager/__init__.py \
             /app/inference_server/inference_server/__init__.py

WORKDIR /app/inference_models
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install ${PYTHON_VERSION} \
    && uv venv --python ${PYTHON_VERSION} /opt/venv \
    && uv pip install --exclude-newer ${PYTHON_DEPENDENCY_CUTOFF} \
        -e ".[torch-cu124,onnx-cu12,trt10-runtime]" \
        -e "../inference_model_manager[cuda]" \
        -e "../inference_server"

COPY inference_models        /app/inference_models
COPY inference_model_manager /app/inference_model_manager
COPY inference_server        /app/inference_server

FROM --platform=${BENCHMARK_PLATFORM} ${CUDA_RUNTIME_IMAGE} AS runtime
ARG SOURCE_REVISION

LABEL org.opencontainers.image.title="Roboflow staging MMP benchmark" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      com.roboflow.environment="staging-only" \
      com.roboflow.experiment="mmp-capacity" \
      com.roboflow.cuda.builder.digest="sha256:0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4" \
      com.roboflow.cuda.runtime.digest="sha256:0bb88834d973ca1b450fcc2a05333c6fe45510bee289912a5391274c351c4a4d"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    API_BASE_URL=https://api.roboflow.one \
    MODEL_CACHE_DIR=/models/cache \
    NUM_WORKERS=4 \
    INFERENCE_N_SLOTS=128 \
    INFERENCE_INPUT_MB=20 \
    INFERENCE_BATCH_MAX_SIZE=0 \
    INFERENCE_BATCH_MAX_WAIT_MS=5 \
    INFERENCE_DEPLOYMENT_MODE=mmp \
    NVIDIA_MPS=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MMP_BENCHMARK_SOURCE_REVISION=${SOURCE_REVISION}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates openssl \
        libgl1 libglib2.0-0 libvips ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local/share/uv/python /root/.local/share/uv/python
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Register CUDA/cuDNN/TensorRT wheels with the dynamic loader. The system CUDA
# runtime remains pinned above; the host driver is injected by the NVIDIA
# container runtime.
RUN { ls -d /opt/venv/lib/python*/site-packages/nvidia/*/lib \
            /opt/venv/lib/python*/site-packages/tensorrt_libs 2>/dev/null || true; } \
      > /etc/ld.so.conf.d/nvidia-pip.conf \
    && ldconfig \
    && mkdir -p /models/cache /opt/mmp-benchmark

RUN python -m pip freeze | LC_ALL=C sort > /opt/mmp-benchmark/python-packages.txt

COPY development/mmp_staging_benchmark/capability_probe.py /opt/mmp-benchmark/capability_probe.py
COPY development/mmp_staging_benchmark/run_concurrent_clients.py /opt/mmp-benchmark/run_concurrent_clients.py
COPY development/mmp_staging_benchmark/spec.same-model.example.json /opt/mmp-benchmark/spec.same-model.example.json
COPY development/mmp_staging_benchmark/spec.same-model-isolated.example.json /opt/mmp-benchmark/spec.same-model-isolated.example.json
COPY development/mmp_staging_benchmark/spec.mixed-model.example.json /opt/mmp-benchmark/spec.mixed-model.example.json
WORKDIR /app/inference_server
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD curl -fsS http://127.0.0.1:8000/v2/server/health || exit 1

CMD ["/bin/bash", "start_server.sh"]
