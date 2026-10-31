#!/bin/sh

set -eux

OPENCV_VERSION="${OPENCV_VERSION:-4.13.0}"
OPENCV_SHA256="${OPENCV_SHA256:?OPENCV_SHA256 is required}"
OPENCV_CONTRIB_SHA256="${OPENCV_CONTRIB_SHA256:?OPENCV_CONTRIB_SHA256 is required}"
OPENCV_PREFIX="${OPENCV_PREFIX:-/opt/opencv}"
OPENCV_PYTHON_EXECUTABLE="${OPENCV_PYTHON_EXECUTABLE:-python3}"
OPENCV_PYTHON3_LIMITED_API="${OPENCV_PYTHON3_LIMITED_API:-OFF}"
OPENCV_BUILD_PYTHON3="${OPENCV_BUILD_PYTHON3:-ON}"
OPENCV_WITH_CUDA="${OPENCV_WITH_CUDA:-OFF}"
OPENCV_CUDA_ARCH_BIN="${OPENCV_CUDA_ARCH_BIN:-}"
OPENCV_CUDA_ARCH_PTX="${OPENCV_CUDA_ARCH_PTX:-}"
OPENCV_NUMPY_TARGET_VERSION="${OPENCV_NUMPY_TARGET_VERSION:-}"
OPENCV_WITH_CUDNN="${OPENCV_WITH_CUDNN:-OFF}"
OPENCV_WITH_CUBLAS="${OPENCV_WITH_CUBLAS:-OFF}"
OPENCV_WITH_CUFFT="${OPENCV_WITH_CUFFT:-OFF}"
OPENCV_DNN_CUDA="${OPENCV_DNN_CUDA:-OFF}"
OPENCV_SOURCE_DIR="${OPENCV_SOURCE_DIR:-/tmp/opencv-src}"
OPENCV_CONTRIB_SOURCE_DIR="${OPENCV_CONTRIB_SOURCE_DIR:-/tmp/opencv-contrib-src}"
OPENCV_BUILD_DIR="${OPENCV_BUILD_DIR:-/tmp/opencv-build}"
OPENCV_PYTHON_INCLUDE_DIR="$(
    "${OPENCV_PYTHON_EXECUTABLE}" -c \
        'import sysconfig; print(sysconfig.get_path("include"))'
)"
OPENCV_NUMPY_INCLUDE_DIR="$(
    "${OPENCV_PYTHON_EXECUTABLE}" -c \
        'import numpy; print(numpy.get_include())'
)"
OPENCV_C_FLAGS="${CFLAGS:-}"
OPENCV_CXX_FLAGS="${CXXFLAGS:-}"

# NumPy 2 defaults extension modules to the NumPy 1.x C API for backwards
# compatibility.  That produces a cv2 module which NumPy 2 refuses to import.
# Jetson images ship NumPy 2, so explicitly opt the Python bindings into its
# 2.0 API when requested by the image Dockerfile.
if [ -n "${OPENCV_NUMPY_TARGET_VERSION}" ]; then
    OPENCV_C_FLAGS="${OPENCV_C_FLAGS} -DNPY_TARGET_VERSION=${OPENCV_NUMPY_TARGET_VERSION}"
    OPENCV_CXX_FLAGS="${OPENCV_CXX_FLAGS} -DNPY_TARGET_VERSION=${OPENCV_NUMPY_TARGET_VERSION}"
fi

case "${OPENCV_WITH_CUDA}" in
    ON|OFF) ;;
    *)
        echo "OPENCV_WITH_CUDA must be ON or OFF" >&2
        exit 2
        ;;
esac

if [ "${OPENCV_WITH_CUDA}" = "ON" ] && [ -z "${OPENCV_CUDA_ARCH_BIN}" ]; then
    echo "OPENCV_CUDA_ARCH_BIN is required when CUDA support is enabled" >&2
    exit 2
fi

mkdir -p "${OPENCV_SOURCE_DIR}" "${OPENCV_CONTRIB_SOURCE_DIR}"
opencv_archive="/tmp/opencv-${OPENCV_VERSION}.tar.gz"
opencv_contrib_archive="/tmp/opencv_contrib-${OPENCV_VERSION}.tar.gz"
curl \
    --fail \
    --location \
    --retry 5 \
    --retry-all-errors \
    --retry-delay 2 \
    "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" \
    -o "${opencv_archive}"
echo "${OPENCV_SHA256}  ${opencv_archive}" | sha256sum -c -
tar -xzf "${opencv_archive}" --strip-components=1 -C "${OPENCV_SOURCE_DIR}"
curl \
    --fail \
    --location \
    --retry 5 \
    --retry-all-errors \
    --retry-delay 2 \
    "https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.tar.gz" \
    -o "${opencv_contrib_archive}"
echo "${OPENCV_CONTRIB_SHA256}  ${opencv_contrib_archive}" | sha256sum -c -
tar -xzf "${opencv_contrib_archive}" \
    --strip-components=1 \
    -C "${OPENCV_CONTRIB_SOURCE_DIR}"

if [ "${OPENCV_WITH_CUDA}" = "ON" ]; then
    cuda_release="$(
        nvcc --version |
            sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' |
            head -n 1
    )"
    set -- ${cuda_release}
    if [ "$#" -ne 2 ]; then
        echo "Could not parse the CUDA compiler version" >&2
        exit 2
    fi
    if [ "$1" -gt 13 ] || { [ "$1" -eq 13 ] && [ "$2" -ge 2 ]; }; then
        cuda_13_2_patch=/tmp/opencv-contrib-cuda-13.2.patch
        curl \
            --fail \
            --location \
            --retry 5 \
            --retry-all-errors \
            --retry-delay 2 \
            https://github.com/opencv/opencv_contrib/commit/f2854f4f5e7b67d4e073ea002ae0174d437e2962.patch \
            -o "${cuda_13_2_patch}"
        echo "d436681d50c61837f82f4f70b835b298e781491beadee43d11914bbe793c983e  ${cuda_13_2_patch}" |
            sha256sum -c -
        patch -d "${OPENCV_CONTRIB_SOURCE_DIR}" -p1 < "${cuda_13_2_patch}"
        rm "${cuda_13_2_patch}"
    fi
fi

cmake \
    -S "${OPENCV_SOURCE_DIR}" \
    -B "${OPENCV_BUILD_DIR}" \
    -GNinja \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="${OPENCV_PREFIX}" \
    -D 'CMAKE_INSTALL_RPATH=/opt/gstreamer/lib;/opt/ffmpeg/lib' \
    -D OPENCV_EXTRA_MODULES_PATH="${OPENCV_CONTRIB_SOURCE_DIR}/modules" \
    -D OPENCV_PYTHON3_INSTALL_PATH="${OPENCV_PREFIX}/python" \
    -D PYTHON3_LIMITED_API="${OPENCV_PYTHON3_LIMITED_API}" \
    -D PYTHON3_EXECUTABLE="${OPENCV_PYTHON_EXECUTABLE}" \
    -D Python3_EXECUTABLE="${OPENCV_PYTHON_EXECUTABLE}" \
    -D PYTHON3_INCLUDE_DIR="${OPENCV_PYTHON_INCLUDE_DIR}" \
    -D PYTHON3_NUMPY_INCLUDE_DIRS="${OPENCV_NUMPY_INCLUDE_DIR}" \
    -D CMAKE_C_FLAGS="${OPENCV_C_FLAGS}" \
    -D CMAKE_CXX_FLAGS="${OPENCV_CXX_FLAGS}" \
    -D WITH_GSTREAMER=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_CUDA="${OPENCV_WITH_CUDA}" \
    -D WITH_CUDNN="${OPENCV_WITH_CUDNN}" \
    -D WITH_CUBLAS="${OPENCV_WITH_CUBLAS}" \
    -D WITH_CUFFT="${OPENCV_WITH_CUFFT}" \
    -D OPENCV_DNN_CUDA="${OPENCV_DNN_CUDA}" \
    -D CUDA_ARCH_BIN="${OPENCV_CUDA_ARCH_BIN}" \
    -D CUDA_ARCH_PTX="${OPENCV_CUDA_ARCH_PTX}" \
    -D BUILD_opencv_cudacodec=OFF \
    -D BUILD_opencv_cudafeatures2d=OFF \
    -D BUILD_opencv_cudalegacy=OFF \
    -D BUILD_opencv_cudaobjdetect=OFF \
    -D BUILD_opencv_cudaoptflow=OFF \
    -D BUILD_opencv_cudastereo=OFF \
    -D WITH_NVCUVID=OFF \
    -D WITH_NVCUVENC=OFF \
    -D WITH_GTK=OFF \
    -D WITH_QT=OFF \
    -D VIDEOIO_ENABLE_PLUGINS=OFF \
    -D BUILD_SHARED_LIBS=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_opencv_apps=OFF \
    -D BUILD_opencv_python3="${OPENCV_BUILD_PYTHON3}" \
    -D BUILD_opencv_videoio=ON \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_DOCS=OFF

cmake --build "${OPENCV_BUILD_DIR}" --parallel "$(nproc)"
cmake --install "${OPENCV_BUILD_DIR}"

license_dir="${OPENCV_PREFIX}/share/licenses/opencv"
mkdir -p "${license_dir}"
cp "${OPENCV_SOURCE_DIR}/LICENSE" "${license_dir}/LICENSE.opencv"
cp "${OPENCV_CONTRIB_SOURCE_DIR}/LICENSE" \
    "${license_dir}/LICENSE.opencv_contrib"

if [ "${OPENCV_BUILD_PYTHON3}" = "ON" ]; then
    PYTHONPATH="${OPENCV_PREFIX}/python" \
        "${OPENCV_PYTHON_EXECUTABLE}" -c \
        "import cv2; info = cv2.getBuildInformation(); assert cv2.videoio_registry.hasBackend(cv2.CAP_GSTREAMER); assert 'YES' in info.split('GStreamer:')[1].split(chr(10))[0]; assert 'YES' in info.split('FFMPEG:')[1].split(chr(10))[0]; print(info)"
fi

runtime_prefix="${OPENCV_PREFIX}-runtime"
cp -a "${OPENCV_PREFIX}" "${runtime_prefix}"
rm -rf \
    "${runtime_prefix}/include" \
    "${runtime_prefix}/lib/cmake" \
    "${runtime_prefix}/lib/pkgconfig"
find "${runtime_prefix}" -type f \( -name '*.a' -o -name '*.la' \) -delete
find "${runtime_prefix}" -type f -name '*.so*' \
    -exec strip --strip-unneeded {} +

rm -rf \
    "${opencv_archive}" \
    "${opencv_contrib_archive}" \
    "${OPENCV_SOURCE_DIR}" \
    "${OPENCV_CONTRIB_SOURCE_DIR}" \
    "${OPENCV_BUILD_DIR}"
