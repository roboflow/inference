FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# install Python 3.12
RUN apt-get update -y && apt-get install -y \
    libssl-dev \
    git

RUN mkdir -p /build/python-3.12
WORKDIR /build/python-3.12
RUN wget https://www.python.org/ftp/python/3.12.12/Python-3.12.12.tgz && tar -xzf Python-3.12.12.tgz
WORKDIR /build/python-3.12/Python-3.12.12
RUN ./configure --enable-optimizations
RUN make -j$(nproc) && make altinstall

RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1

# install TensorRT
RUN mkdir -p /build/tensorrt
WORKDIR /build/tensorrt
RUN git clone https://github.com/NVIDIA/TensorRT.git
WORKDIR /build/tensorrt/TensorRT
RUN git checkout 8.5.2 && git submodule update --init --recursive
ENV TRT_OSSPATH=/build/tensorrt/TensorRT
ENV TRT_LIBPATH=/usr/lib/aarch64-linux-gnu
ENV EXT_PATH=/build/tensorrt/extenral
RUN mkdir -p /build/tensorrt/extenral/python3.12/include/
RUN cp -r /usr/local/include/python3.12/* /build/tensorrt/extenral/python3.12/include/
WORKDIR /build/tensorrt/extenral/
RUN git clone https://github.com/pybind/pybind11.git
WORKDIR /build/tensorrt/extenral/pybind11
RUN git checkout v3.0.1
WORKDIR  /build/tensorrt/TensorRT/python
RUN  PYTHON_MAJOR_VERSION=3 PYTHON_MINOR_VERSION=12 TARGET_ARCHITECTURE=aarch64 bash ./build.sh
RUN python3.12 -m pip install build/dist/tensorrt-*.whl

# Install numpy


# Install OpenCV


# Install ONNX-runtime GPU


# Install PyTorch


# Install flash-attention