FROM python:3.12

RUN rm -rf /var/lib/apt/lists/* && apt-get clean && apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopencv-dev \
    ffmpeg \
    libxext6 \
    curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /build

COPY inference_experimental/uv.lock uv.lock
COPY inference_experimental/pyproject.toml pyproject.toml

RUN . $HOME/.local/bin/env
RUN $HOME/.local/bin/uv pip install --system -r pyproject.toml --extra torch-cpu --extra onnx-cpu --extra mediapipe --extra grounding-dino
COPY inference_experimental/inference_exp inference_exp
RUN $HOME/.local/bin/uv build
RUN WHEEL=$(ls dist/inference_exp-*.whl) && $HOME/.local/bin/uv pip install --system "${WHEEL}"

WORKDIR /
RUN rm -r /build

ENTRYPOINT ["bash"]
