# Video POC processor image. The inference GPU server image ships the full
# `inference` package with CUDA onnxruntime; its runtime stage lacks the ffmpeg
# CLI, which the processor needs for stream-simulation replays and for encoding
# batch results to mp4.
FROM roboflow/roboflow-inference-server-gpu:latest

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir google-cloud-pubsub requests aiortc

WORKDIR /app
COPY processor.py /app/processor.py

# ffmpeg is on PATH in this image (no bundled ../bin), results under /tmp
ENV FFMPEG_BIN=ffmpeg

ENTRYPOINT ["python", "/app/processor.py"]
