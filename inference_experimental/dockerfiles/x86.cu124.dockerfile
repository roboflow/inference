FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN rm -rf /var/lib/apt/lists/* && apt-get clean && apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    libopencv-dev \
    ffmpeg \
    libxext6 \
    curl
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update -y && apt install -y python3.12 python3.12-dev
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /build

COPY inference_experimental/uv.locl uv.locl
COPY inference_experimental/pyproject.toml pyproject.toml

RUN . $HOME/.local/bin/env
RUN $HOME/.local/bin/uv pip install --system -r pyproject.toml --extra torch-cu124 --extra onnx-cu12 --extra mediapipe --extra grounding-dino --extra trt10
RUN MAX_JOBS=$(nproc) $HOME/.local/bin/uv pip install --system --no-build-isolation -r pyproject.toml --extra
RUN $HOME/.local/bin/uv build
RUN WHEEL=$(ls dist/inference_exp-*.whl) && $HOME/.local/bin/uv pip install --system "${WHEEL}"

WORKDIR /
RUN rm -r /build

ENTRYPOINT ["bash"]
