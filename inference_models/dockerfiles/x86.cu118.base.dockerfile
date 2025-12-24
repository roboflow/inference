FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN rm -rf /var/lib/apt/lists/* && apt-get clean && apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    libopencv-dev \
    ffmpeg \
    libxext6 \
    libvips-dev \
    curl
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update -y && apt install -y python3.12 python3.12-dev
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /build

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN UV_PROJECT_ENVIRONMENT=/usr $HOME/.local/bin/uv sync --locked --extra torch-cu118 --extra onnx-cu118 --extra mediapipe --extra trt10
COPY inference_exp inference_exp
RUN $HOME/.local/bin/uv build
RUN WHEEL=$(ls dist/inference_exp-*.whl) && $HOME/.local/bin/uv pip install --system "${WHEEL}"

WORKDIR /
RUN rm -r /build

ENTRYPOINT ["bash"]
