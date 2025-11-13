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
RUN python3.12 -m pip install --upgrade pip setuptools
RUN  PYTHON_MAJOR_VERSION=3 PYTHON_MINOR_VERSION=12 TARGET_ARCHITECTURE=aarch64 bash ./build.sh
RUN python3.12 -m pip install build/dist/tensorrt-*.whl

# Install OpenCV
RUN python3.12 -m pip install "numpy~=2.3.4"
RUN apt-get install unzip
RUN mkdir -p /build/opencv
WORKDIR  /build/opencv
RUN curl -L https://github.com/opencv/opencv/archive/4.12.0.zip -o opencv-4.12.0.zip
RUN curl -L https://github.com/opencv/opencv_contrib/archive/4.12.0.zip -o opencv_contrib-4.12.0.zip
RUN unzip opencv-4.12.0.zip
RUN unzip opencv_contrib-4.12.0.zip
WORKDIR /build/opencv/opencv-4.12.0
RUN mkdir release
WORKDIR /build/opencv/opencv-4.12.0/release
RUN cmake -D WITH_CUDA=ON -D WITH_CUDNN=ON -D CUDA_ARCH_BIN="8.7" -D CUDA_ARCH_PTX="" -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.12.0/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_opencv_python3=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
RUN make -j$(nproc)
RUN make install
# building wheel pip wheel ./python_loader --wheel-dir ~/my_wheels --verbose

# Install newer Cmake for builds
RUN mkdir -p /build/cmake
WORKDIR /build/cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v4.1.2/cmake-4.1.2-linux-aarch64.sh
RUN mkdir build && chmod ugo+x cmake-4.1.2-linux-aarch64.sh && bash cmake-4.1.2-linux-aarch64.sh --skip-license --prefix=./build
RUN export PATH=/build/cmake/build/bin:$PATH

# Install newer GCC for builds
RUN mkdir -p /build/gcc
WORKDIR /build/gcc

# version 14 turned out to be to new :)
#RUN wget https://ftp.gnu.org/gnu/gcc/gcc-14.2.0/gcc-14.2.0.tar.gz
#RUN tar xzf gcc-14.2.0.tar.gz
#WORKDIR /build/gcc/gcc-14.2.0
#RUN ./contrib/download_prerequisites
#WORKDIR /build/gcc/
#RUN mkdir objdir
#WORKDIR /build/gcc/objdir
#RUN $PWD/../gcc-14.2.0/configure --prefix=$HOME/GCC-14 --enable-languages=c,c++
#RUN make -j$(nproc)
#RUN make install
#RUN export PATH=/root/GCC-14/bin:$PATH
#RUN export LD_LIBRARY_PATH=/root/GCC-14/lib64/:$LD_LIBRARY_PATH
#RUN ldconfig

# hoping that version 11 will be fine :)
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
RUN export PATH=/root/GCC-14/bin:$PATH
RUN export LD_LIBRARY_PATH=/root/GCC-11/lib64/:$LD_LIBRARY_PATH
RUN ldconfig

# Install ONNX-runtime GPU
RUN mkdir -p /build/onnxruntime
WORKDIR /build/onnxruntime
RUN git clone https://github.com/microsoft/onnxruntime.git
WORKDIR /build/onnxruntime/onnxruntime
RUN git checkout v1.23.2
RUN CC=/root/GCC-11/bin/gcc CXX=/root/GCC-11/bin/g++ ./build.sh --update --config Release --build --build_wheel --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu --use_tensorrt --tensorrt_home /usr/lib/aarch64-linux-gnu --allow_running_as_root

# Install PyTorch


# Install flash-attention