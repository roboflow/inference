FROM nvcr.io/nvidia/l4t-jetpack:r36.4.3

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
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN . $HOME/.local/bin/env

WORKDIR /build

COPY inference_experimental/dockerfiles/build_scripts/compile_opencv_jetpack_6.sh compile_opencv_jetpack_6.sh
RUN chmod ugo+x compile_opencv_jetpack_6.sh
RUN ./compile_opencv_jetpack_6.sh

COPY inference_experimental/uv.lock uv.lock
COPY inference_experimental/pyproject.toml pyproject.toml
RUN $HOME/.local/bin/uv pip install --system -r pyproject.toml --extra jp6-cu128 --extra onnx-jp6-cu128 --extra trt10
RUN MAX_JOBS=$(nproc) $HOME/.local/bin/uv pip install -v --system --no-build-isolation -r pyproject.toml --extra flash-attn-jp6
COPY inference_experimental/inference_exp inference_exp
RUN $HOME/.local/bin/uv build
RUN WHEEL=$(ls dist/inference_exp-*.whl) && $HOME/.local/bin/uv pip install --system "${WHEEL}"

WORKDIR /
RUN rm -r /build

ENTRYPOINT ["bash"]
