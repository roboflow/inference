FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN rm -rf /var/lib/apt/lists/* && apt-get clean && apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    libopencv-dev \
    ffmpeg \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY inference_experimental /build

WORKDIR /build

RUN . $HOME/.local/bin/env
RUN $HOME/.local/bin/uv build

RUN WHEEL=$(ls dist/inference_exp-*.whl) && python -m pip install "${WHEEL}[torch-cu126,onnx-cu12,mediapipe,grounding-dino,trt10]"
RUN WHEEL=$(ls dist/inference_exp-*.whl) && python -m pip install --no-build-isolation "${WHEEL}[flash-attn]"

WORKDIR /
RUN rm -r /build

ENTRYPOINT ["bash"]
