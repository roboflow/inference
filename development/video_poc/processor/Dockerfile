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

ARG VIDEO_PROC_GIT_SHA
ARG VIDEO_PROC_RUNTIME_VARIANT=video-poc

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Inference v1.4 deliberately strips its build-time package manager from the
# runtime image. The video worker still needs three runtime-only integrations,
# so install them after restoring Ubuntu's Python 3 pip package explicitly.
RUN python3 -m pip install --no-cache-dir google-cloud-pubsub requests aiortc

WORKDIR /app
COPY processor.py /app/processor.py
COPY processor_metrics.py /app/processor_metrics.py
COPY job_telemetry.py /app/job_telemetry.py
COPY job_process.py /app/job_process.py
COPY inference_runtime_compat.py /app/inference_runtime_compat.py
COPY run_lifecycle.py /app/run_lifecycle.py
COPY worker_lifecycle.py /app/worker_lifecycle.py
COPY execution_domains.py /app/execution_domains.py
COPY security.py /app/security.py
COPY claim_proof.py /app/claim_proof.py
COPY video_ingest.py /app/video_ingest.py
COPY file_replay.py /app/file_replay.py
# low-latency RTSP ingest (PyAV, codec-level low_delay); without it the
# processor silently falls back to cv2 ingest and its ~550ms decoder buffer
COPY low_latency_producer.py /app/low_latency_producer.py

# ffmpeg is on PATH in this image (no bundled ../bin), results under /tmp
ENV FFMPEG_BIN=ffmpeg
ENV VIDEO_PROC_GIT_SHA=${VIDEO_PROC_GIT_SHA}
ENV VIDEO_PROC_RUNTIME_VARIANT=${VIDEO_PROC_RUNTIME_VARIANT}

ENTRYPOINT ["python", "/app/processor.py"]
