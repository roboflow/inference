FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8

RUN chmod 1777 /tmp
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
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    gdal-bin \
    libgdal-dev \
    rustc \
    cargo \
    curl \
    libvips-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /build

COPY dockerfiles/build_scripts/compile_opencv_jetpack_6.sh compile_opencv_jetpack_6.sh
RUN chmod ugo+x compile_opencv_jetpack_6.sh
RUN ./compile_opencv_jetpack_6.sh

WORKDIR /build/inference_models

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
RUN UV_PROJECT_ENVIRONMENT=/usr $HOME/.local/bin/uv sync --locked --no-install-package opencv-python --extra torch-jp6-cu126 --extra onnx-jp6-cu126
RUN $HOME/.local/bin/uv pip install --system /build/opencv_compilation/opencv-4.11.0/release/python_loader/dist/opencv-4.11.0-py3-none-any.whl
COPY inference_models inference_models
RUN $HOME/.local/bin/uv build
RUN WHEEL=$(ls dist/inference_models-*.whl) && $HOME/.local/bin/uv pip install --system --no-deps "${WHEEL}"

WORKDIR /
RUN rm -r /build/inference_models

ENTRYPOINT ["bash"]
