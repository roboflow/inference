# Video POC processor image. The inference server image ships the full
# `inference` package; its runtime stage lacks the ffmpeg CLI, which the
# processor needs for stream-simulation replays and for encoding batch results
# to mp4. Two tiers from one Dockerfile:
#   gpu (default) — CUDA onnxruntime base
#   cpu — --build-arg BASE_IMAGE=roboflow/roboflow-inference-server-cpu:latest
ARG BASE_IMAGE=roboflow/roboflow-inference-server-gpu:latest
FROM ${BASE_IMAGE}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir google-cloud-pubsub requests aiortc

WORKDIR /app
COPY processor.py /app/processor.py
# low-latency RTSP ingest (PyAV, codec-level low_delay); without it the
# processor silently falls back to cv2 ingest and its ~550ms decoder buffer
COPY low_latency_producer.py /app/low_latency_producer.py

# ffmpeg is on PATH in this image (no bundled ../bin), results under /tmp
ENV FFMPEG_BIN=ffmpeg

ENTRYPOINT ["python", "/app/processor.py"]
