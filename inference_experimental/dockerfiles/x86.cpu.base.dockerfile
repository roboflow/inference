FROM --platform=linux/amd64 python:3.12

RUN rm -rf /var/lib/apt/lists/* && apt-get clean && apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopencv-dev \
    ffmpeg \
    libxext6 \
    libvips-dev \
    curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /build

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
RUN UV_PROJECT_ENVIRONMENT=/usr/local $HOME/.local/bin/uv sync --locked --extra torch-cpu --extra onnx-cpu --extra mediapipe --extra grounding-dino
COPY inference_exp inference_exp
RUN $HOME/.local/bin/uv build
RUN WHEEL=$(ls dist/inference_exp-*.whl) && $HOME/.local/bin/uv pip install --system "${WHEEL}"

WORKDIR /
RUN rm -r /build

ENTRYPOINT ["bash"]
