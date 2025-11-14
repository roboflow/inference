FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

# install Python 3.12
RUN apt-get update -y && apt-get install -y \
    libssl-dev \
    git \
    unzip \
    libbz2-dev \
    libssl-dev \
    libsqlite3-dev \
    zlib1g-dev \
    liblzma-dev


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
RUN git checkout v10.3.0 && git submodule update --init --recursive
ENV TRT_OSSPATH=/build/tensorrt/TensorRT
ENV TRT_LIBPATH=/usr/lib/aarch64-linux-gnu
ENV EXT_PATH=/build/tensorrt/extenral
ENV TENSORRT_MODULE=tensorrt
RUN mkdir build
WORKDIR /build/tensorrt/TensorRT/build
RUN cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=11.4
RUN make -j$(nproc)
RUN mkdir -p /build/tensorrt/extenral/python3.12/include/
RUN cp -r /usr/local/include/python3.12/* /build/tensorrt/extenral/python3.12/include/
WORKDIR /build/tensorrt/extenral/
RUN git clone https://github.com/pybind/pybind11.git
WORKDIR /build/tensorrt/extenral/pybind11
RUN git checkout v3.0.1
WORKDIR  /build/tensorrt/TensorRT/python
RUN python3.12 -m pip install --upgrade pip setuptools
RUN  PYTHON_MAJOR_VERSION=3 PYTHON_MINOR_VERSION=12 TARGET_ARCHITECTURE=aarch64 bash ./build.sh
RUN python3.12 -m pip install ./build/bindings_wheel/dist/tensorrt-*.whl
#
## Install OpenCV
#RUN python3.12 -m pip install "numpy~=2.3.4"
#RUN mkdir -p /build/opencv
#WORKDIR  /build/opencv
#RUN curl -L https://github.com/opencv/opencv/archive/4.12.0.zip -o opencv-4.12.0.zip
#RUN curl -L https://github.com/opencv/opencv_contrib/archive/4.12.0.zip -o opencv_contrib-4.12.0.zip
#RUN unzip opencv-4.12.0.zip
#RUN unzip opencv_contrib-4.12.0.zip
#WORKDIR /build/opencv/opencv-4.12.0
#RUN mkdir release
#WORKDIR /build/opencv/opencv-4.12.0/release
#RUN cmake -D WITH_CUDA=ON -D WITH_CUDNN=ON -D CUDA_ARCH_BIN="8.7" -D CUDA_ARCH_PTX="" -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.12.0/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_opencv_python3=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
#RUN make -j$(nproc)
#RUN make install
#RUN python3.12 -m pip wheel ./python_loader --wheel-dir ./my_wheels --verbose
#RUN python3.12 -m pip install ./my_wheels/opencv-4.12.0-py3-none-any.whl
#
## Install newer Cmake for builds
#RUN mkdir -p /build/cmake
#WORKDIR /build/cmake
#RUN wget https://github.com/Kitware/CMake/releases/download/v4.1.2/cmake-4.1.2-linux-aarch64.sh
#RUN mkdir build && chmod ugo+x cmake-4.1.2-linux-aarch64.sh && bash cmake-4.1.2-linux-aarch64.sh --skip-license --prefix=./build
#
#RUN mkdir -p /build/eigen3
#WORKDIR /build/eigen3
#RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.1/eigen-3.4.1.tar.gz
#RUN tar xzf eigen-3.4.1.tar.gz
#WORKDIR /build/eigen3/eigen-3.4.1
#
## Install ONNX-runtime GPU
#RUN mkdir -p /build/onnxruntime
#WORKDIR /build/onnxruntime
#RUN git clone https://github.com/microsoft/onnxruntime.git
#WORKDIR /build/onnxruntime/onnxruntime
#RUN git checkout v1.16.3
#RUN python3.12 -m pip install packaging
#RUN PATH=/build/cmake/build/bin:$PATH CMAKE_POLICY_VERSION_MINIMUM=3.5 ./build.sh --update --config Release --build --build_wheel --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu --allow_running_as_root --parallel 0 --use_preinstalled_eigen --eigen_path /build/eigen3/eigen-3.4.1 --skip_tests --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF
#RUN python3.12 -m pip install ./build/Linux/Release/dist/onnxruntime_gpu-1.16.3-cp312-cp312-linux_aarch64.whl
#
## Install PyTorch
#RUN mkdir -p /build/torch
#WORKDIR /build/torch
#RUN git clone https://github.com/pytorch/pytorch.git
#WORKDIR /build/torch/pytorch
#RUN git checkout v2.4.1
#RUN git submodule sync && git submodule update --init --recursive
#RUN PATH=/build/cmake/build/bin:$PATH python3.12 -m pip install setuptools wheel astunparse numpy ninja pyyaml cmake "typing-extensions>=4.10.0" requests
#ARG MAX_TORCH_COMPILATION_JOBS=4
#RUN PATH=/build/cmake/build/bin:$PATH PYTORCH_BUILD_VERSION=2.4.1 PYTORCH_BUILD_NUMBER=1 MAX_JOBS=${MAX_TORCH_COMPILATION_JOBS} CUDA_HOME=/usr/local/cuda CUDACXX=/usr/local/cuda/bin/nvcc TORCH_CUDA_ARCH_LIST="8.7" USE_NCCL=0 USE_DISTRIBUTED=0 USE_MKLDNN=0 BUILD_TEST=0 CMAKE_POLICY_VERSION_MINIMUM=3.5 python3.12 setup.py bdist_wheel
#RUN python3.12 -m pip install dist/torch-*.whl
#
## Install Torchvision
#RUN mkdir -p /build/torchvision
#WORKDIR /build/torchvision
#RUN git clone https://github.com/pytorch/vision.git
#WORKDIR /build/torchvision/vision
#RUN git checkout v0.19.1
#RUN git submodule sync && git submodule update --init --recursive
#RUN PATH=/build/cmake/build/bin:$PATH BUILD_VERSION=0.19.1 TORCH_CUDA_ARCH_LIST="8.7" CMAKE_POLICY_VERSION_MINIMUM=3.5 python3.12 setup.py bdist_wheel
#RUN python3.12 -m pip install dist/torchvision-*.whl
