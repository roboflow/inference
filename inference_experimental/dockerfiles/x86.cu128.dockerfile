FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

RUN rm -rf /var/lib/apt/lists/* && apt-get clean && apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common \
    libopencv-dev \
    ffmpeg \
    libxext6 \
    curl
RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update -y && apt install -y python3.12-devel
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY inference_experimental /build

WORKDIR /build

RUN . $HOME/.local/bin/env
RUN $HOME/.local/bin/uv build

RUN WHEEL=$(ls dist/inference_exp-*.whl) && $HOME/.local/bin/uv pip install --system "${WHEEL}[torch-cu128,onnx-cu12,mediapipe,grounding-dino,trt10]"
RUN WHEEL=$(ls dist/inference_exp-*.whl) && $HOME/.local/bin/uv pip install --system --no-build-isolation "${WHEEL}[flash-attn]"

WORKDIR /
RUN rm -r /build

ENTRYPOINT ["bash"]
