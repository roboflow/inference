# Video POC processor image. The inference server image ships the full
# `inference` package; its runtime stage lacks the ffmpeg CLI, which the
# processor needs for stream-simulation replays and for encoding batch results
# to mp4. Two tiers from one Dockerfile:
#   gpu (default) — CUDA onnxruntime base
#   cpu — --build-arg BASE_IMAGE=roboflow/roboflow-inference-server-cpu:latest
# For an A/B experiment that only changes these worker files, prefer
# Dockerfile.overlay with the currently deployed immutable processor image as
# its base. That avoids silently changing the inference runtime under test.
ARG BASE_IMAGE=roboflow/roboflow-inference-server-gpu:latest
FROM ${BASE_IMAGE}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir google-cloud-pubsub requests aiortc

WORKDIR /app
COPY processor.py /app/processor.py
COPY processor_metrics.py /app/processor_metrics.py
COPY job_telemetry.py /app/job_telemetry.py
COPY worker_lifecycle.py /app/worker_lifecycle.py
COPY execution_domains.py /app/execution_domains.py
COPY workspace_model_managers.py /app/workspace_model_managers.py
COPY security.py /app/security.py
# low-latency RTSP ingest (PyAV, codec-level low_delay); without it the
# processor silently falls back to cv2 ingest and its ~550ms decoder buffer
COPY low_latency_producer.py /app/low_latency_producer.py

# ffmpeg is on PATH in this image (no bundled ../bin), results under /tmp
ENV FFMPEG_BIN=ffmpeg

ENTRYPOINT ["python", "/app/processor.py"]
