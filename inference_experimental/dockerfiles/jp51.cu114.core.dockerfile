FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3 AS builder

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

# Get rid of tensorrt-8.X
RUN apt remove -y 'libnvinfer*' 'libnvonnxparsers*' 'libnvparsers*' 'libnvinfer-plugin*' 'python3-libnvinfer*' 'tensorrt*' 'uff-converter*' 'graphsurgeon*'

# Create out dir where all wheels will be stored
RUN mkdir -p /build/out/wheels

# Install tensorrt-10.x
RUN mkdir -p /build/tensorrt-10.x
WORKDIR /build/tensorrt-10.x
RUN wget https://storage.googleapis.com/roboflow-tests-assets/TensorRT/TensorRT-10.8.0.43.l4t.aarch64-gnu.cuda-11.4.tar.gz
RUN tar xzf TensorRT-10.8.0.43.l4t.aarch64-gnu.cuda-11.4.tar.gz
WORKDIR /build/tensorrt-10.x/TensorRT-10.8.0.43/targets/aarch64-linux-gnu
RUN mkdir -p /usr/src/tensorrt/bin
RUN cp bin/trtexec /usr/src/tensorrt/bin/trtexec
RUN cp include/* /usr/include/aarch64-linux-gnu/
RUN mkdir -p /usr/lib/aarch64-linux-gnu/stubs
RUN cp -r lib/stubs/* /usr/lib/aarch64-linux-gnu/stubs/
RUN cp lib/libnvinfer.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer.so.10.8.0 && \
    cp lib/libnvinfer_builder_resource.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource.so.10.8.0 && \
    cp lib/libnvinfer_static.a /usr/lib/aarch64-linux-gnu/libnvinfer_static.a && \
    cp lib/libnvinfer_dispatch.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_dispatch.so.10.8.0 && \
    cp lib/libnvinfer_dispatch_static.a /usr/lib/aarch64-linux-gnu/libnvinfer_dispatch_static.a && \
    cp lib/libnvinfer_lean.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_lean.so.10.8.0 && \
    cp lib/libnvinfer_lean_static.a /usr/lib/aarch64-linux-gnu/libnvinfer_lean_static.a && \
    cp lib/libnvinfer_plugin.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10.8.0 && \
    cp lib/libnvinfer_plugin_static.a /usr/lib/aarch64-linux-gnu/libnvinfer_plugin_static.a && \
    cp lib/libnvinfer_vc_plugin.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_vc_plugin.so.10.8.0 && \
    cp lib/libnvinfer_vc_plugin_static.a /usr/lib/aarch64-linux-gnu/libnvinfer_vc_plugin_static.a && \
    cp lib/libnvonnxparser.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvonnxparser.so.10.8.0 && \
    cp lib/libnvonnxparser_static.a /usr/lib/aarch64-linux-gnu/libnvonnxparser_static.a && \
    cp lib/libonnx_proto.a /usr/lib/aarch64-linux-gnu/libonnx_proto.a

RUN ln -s /usr/lib/aarch64-linux-gnu/libnvinfer.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer.so.10 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnvinfer.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer.so && \
    ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_dispatch.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_dispatch.so.10 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_dispatch.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_dispatch.so && \
    ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_lean.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_lean.so.10 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_lean.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_lean.so && \
    ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so && \
    ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_vc_plugin.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_vc_plugin.so.10 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_vc_plugin.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvinfer_vc_plugin.so && \
    ln -s /usr/lib/aarch64-linux-gnu/libnvonnxparser.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvonnxparser.so.10 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnvonnxparser.so.10.8.0 /usr/lib/aarch64-linux-gnu/libnvonnxparser.so

WORKDIR /build/tensorrt-10.x/TensorRT-10.8.0.43/python
RUN cp -r * /build/out/wheels
RUN python3.12 -m pip install /build/out/wheels/tensorrt-10.8.0.43-cp312-none-linux_aarch64.whl

# Install newer Cmake for builds
RUN mkdir -p /build/cmake
WORKDIR /build/cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v4.1.2/cmake-4.1.2-linux-aarch64.sh
RUN mkdir build && chmod ugo+x cmake-4.1.2-linux-aarch64.sh && bash cmake-4.1.2-linux-aarch64.sh --skip-license --prefix=./build

# Install gcc-11
WORKDIR /build/gcc/
RUN wget https://ftp.gnu.org/gnu/gcc/gcc-11.1.0/gcc-11.1.0.tar.gz
RUN tar xzf gcc-11.1.0.tar.gz
WORKDIR /build/gcc/gcc-11.1.0
RUN ./contrib/download_prerequisites
WORKDIR /build/gcc/
RUN mkdir objdir
WORKDIR /build/gcc/objdir
RUN $PWD/../gcc-11.1.0/configure --prefix=$HOME/GCC-11 --enable-languages=c,c++
RUN make -j$(nproc)
RUN make install
RUN export PATH=/root/GCC-11/bin:$PATH
RUN export LD_LIBRARY_PATH=/root/GCC-11/lib64/:$LD_LIBRARY_PATH
RUN ldconfig

# upgrade to CUDA 11.8
WORKDIR /build/cuda-118
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-ubuntu2004.pin -O /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb && \
    dpkg -i cuda-tegra-repo-*.deb && \
    rm cuda-tegra-repo-*.deb

RUN cp /var/cuda-tegra-repo-*/cuda-tegra-*-keyring.gpg /usr/share/keyrings/

RUN mkdir /var/cuda-compat && \
    cd /var/cuda-compat && \
    ar x ../cuda-tegra-repo-*/cuda-compat-*.deb && \
    tar xvf data.tar.xz -C / && \
    rm -rf /var/cuda-compat

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            cuda-toolkit-* \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install ONNX-runtime GPU
RUN mkdir -p /build/onnxruntime
WORKDIR /build/onnxruntime
RUN git clone https://github.com/microsoft/onnxruntime.git
WORKDIR /build/onnxruntime/onnxruntime
RUN git checkout v1.21.1
# Hash aligned with the source code that had this problem fixed on main branch - we need to stick to this version and patch, as our env is cuda 11 and the patched version do only support cuda 12
RUN sed -i 's|eigen;https://gitlab.com/libeigen/eigen/-/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33/eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip;5ea4d05e62d7f954a46b3213f9b2535bdd866803|eigen;https://github.com/eigen-mirror/eigen/archive/1d8b82b0740839c0de7f1242a3585e3390ff5f33/eigen-1d8b82b0740839c0de7f1242a3585e3390ff5f33.zip;05b19b49e6fbb91246be711d801160528c135e34|' cmake/deps.txt
RUN python3.12 -m pip install packaging setuptools "numpy==2.3.5"
RUN LD_LIBRARY_PATH=/root/GCC-11/lib64/:$LD_LIBRARY_PATH CC=/root/GCC-11/bin/gcc CXX=/root/GCC-11/bin/g++ PATH=/build/cmake/build/bin:$PATH CMAKE_POLICY_VERSION_MINIMUM=3.5 ./build.sh --update --config Release --build --build_wheel --use_cuda --cuda_version=11.8 --cuda_home /usr/local/cuda-11.8 --cudnn_home /usr/lib/aarch64-linux-gnu --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu --allow_running_as_root --parallel 4 --disable_types float8 --skip_tests --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF
RUN python3.12 -m pip install ./build/Linux/Release/dist/onnxruntime_gpu-1.21.1-cp312-cp312-linux_aarch64.whl
RUN cp ./build/Linux/Release/dist/onnxruntime_gpu-1.21.1-cp312-cp312-linux_aarch64.whl /build/out/wheels/onnxruntime_gpu-1.21.1-cp312-cp312-linux_aarch64.whl

# Install PyTorch
RUN mkdir -p /build/torch
WORKDIR /build/torch
RUN git clone https://github.com/pytorch/pytorch.git
WORKDIR /build/torch/pytorch
RUN git checkout v2.4.1
RUN git submodule sync && git submodule update --init --recursive
RUN PATH=/build/cmake/build/bin:$PATH python3.12 -m pip install setuptools wheel astunparse numpy ninja pyyaml cmake "typing-extensions>=4.10.0" requests
ARG MAX_TORCH_COMPILATION_JOBS=4
RUN PATH=/build/cmake/build/bin:$PATH PYTORCH_BUILD_VERSION=2.4.1 PYTORCH_BUILD_NUMBER=1 MAX_JOBS=${MAX_TORCH_COMPILATION_JOBS} FORCE_CUDA=1 CUDA_HOME=/usr/local/cuda-11.8 CUDACXX=/usr/local/cuda-11.8/bin/nvcc TORCH_CUDA_ARCH_LIST="8.7" USE_NCCL=0 USE_DISTRIBUTED=0 USE_MKLDNN=0 BUILD_TEST=0 CMAKE_POLICY_VERSION_MINIMUM=3.5 python3.12 setup.py bdist_wheel
RUN python3.12 -m pip install dist/torch-*.whl
RUN cp dist/torch-*.whl /build/out/wheels/

# Install Torchvision
RUN mkdir -p /build/torchvision
WORKDIR /build/torchvision
RUN git clone https://github.com/pytorch/vision.git
WORKDIR /build/torchvision/vision
RUN git checkout v0.19.1
RUN git submodule sync && git submodule update --init --recursive
RUN PATH=/build/cmake/build/bin:$PATH BUILD_VERSION=0.19.1 TORCH_CUDA_ARCH_LIST="8.7" CUDA_HOME=/usr/local/cuda-11.8 CMAKE_POLICY_VERSION_MINIMUM=3.5 python3.12 setup.py bdist_wheel
RUN python3.12 -m pip install dist/torchvision-*.whl
RUN cp dist/torchvision-*.whl /build/out/wheels/

FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3 AS target

RUN apt-get update -y && apt-get install -y \
    libssl-dev \
    git \
    unzip \
    libbz2-dev \
    libssl-dev \
    libsqlite3-dev \
    zlib1g-dev \
    liblzma-dev

RUN apt remove -y 'libnvinfer*' 'libnvonnxparsers*' 'libnvparsers*' 'libnvinfer-plugin*' 'python3-libnvinfer*' 'tensorrt*' 'uff-converter*' 'graphsurgeon*'


COPY --from=builder /root/GCC-11 /opt/gcc-11
COPY --from=builder /build/out/wheels /compiled_python_packages
COPY --from=builder /usr/include /usr/include
COPY --from=builder /usr/lib /usr/lib
COPY --from=builder /usr/share /usr/share
COPY --from=builder /usr/src /usr/src
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/include /usr/local/include
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/share /usr/local/share
COPY --from=builder /usr/local/cuda-11.8 /usr/local/cuda-11.8
RUN rm /etc/alternatives/cuda /etc/alternatives/cuda-11
RUN ln -s /usr/local/cuda-11.8 /etc/alternatives/cuda
RUN ln -s /usr/local/cuda-11.8 /etc/alternatives/cuda-11
RUN rm -rf /usr/local/cuda-11.4
ENV LD_LIBRARY_PATH="/opt/gcc-11/lib64:$$LD_LIBRARY_PATH"


RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1

# Install OpenCV
RUN mkdir -p /build/opencv
WORKDIR  /build/opencv
RUN curl -L https://github.com/opencv/opencv/archive/4.12.0.zip -o opencv-4.12.0.zip
RUN curl -L https://github.com/opencv/opencv_contrib/archive/4.12.0.zip -o opencv_contrib-4.12.0.zip
RUN unzip opencv-4.12.0.zip
RUN unzip opencv_contrib-4.12.0.zip
WORKDIR /build/opencv/opencv-4.12.0
RUN mkdir release
WORKDIR /build/opencv/opencv-4.12.0/release
RUN cmake -D WITH_CUDA=ON -D WITH_CUDNN=ON -D CUDA_ARCH_BIN="8.7" -D CUDA_ARCH_PTX="" -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.12.0/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_opencv_python3=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=OFF -DWITH_OPENCLAMDFFT=OFF -DWITH_OPENCLAMDBLAS=OFF -DWITH_VA_INTEL=OFF ..
RUN make -j$(nproc)
RUN make install
RUN python3.12 -m pip wheel ./python_loader --wheel-dir /build/out/wheels --verbose
RUN python3.12 -m pip install /build/out/wheels/opencv-4.12.0-py3-none-any.whl

WORKDIR /

ENTRYPOINT ["bash"]
