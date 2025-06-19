FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    lshw \
    git \
    gfortran \
    build-essential \
    libopenblas-dev \
    libatlas-base-dev \
    libsm6 \
    libxext6 \
    wget \
    gdal-bin \
    libgdal-dev \
    rustc \
    cargo \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /build

COPY inference_experimental/dockerfiles/build_scripts/compile_opencv_jetpack_6.sh compile_opencv_jetpack_6.sh
RUN chmod ugo+x compile_opencv_jetpack_6.sh
RUN ./compile_opencv_jetpack_6.sh

WORKDIR /build/inference_experimental

COPY inference_experimental/uv.lock uv.lock
COPY inference_experimental/pyproject.toml pyproject.toml
RUN UV_PROJECT_ENVIRONMENT=/usr $HOME/.local/bin/uv sync --locked --no-install-package opencv-python --extra torch-jp6-cu126 --extra onnx-jp6-cu126
COPY inference_experimental/inference_exp inference_exp
RUN $HOME/.local/bin/uv pip install --no-deps .

WORKDIR /
RUN rm -r /build/inference_experimental

ENTRYPOINT ["bash"]
