FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN rm -rf /var/lib/apt/lists/* && apt-get clean && apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    libopencv-dev \
    ffmpeg \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY inference_experimental /build

WORKDIR /build

RUN source $HOME/.local/bin/env
RUN uv build

RUN python3 -m pip install dist/inference_exp-*.whl

ENTRYPOINT ["bash"]