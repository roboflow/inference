#!/bin/sh

set -eu

OPENCV_REQUIRE_CUDA_BUILD="${OPENCV_REQUIRE_CUDA_BUILD:-false}"
OPENCV_REQUIRE_CUDA_RUNTIME="${OPENCV_REQUIRE_CUDA_RUNTIME:-false}"
OPENCV_EXPECTED_CUDA_ARCH_BIN="${OPENCV_EXPECTED_CUDA_ARCH_BIN:-}"
export \
    OPENCV_EXPECTED_CUDA_ARCH_BIN \
    OPENCV_REQUIRE_CUDA_BUILD \
    OPENCV_REQUIRE_CUDA_RUNTIME

python3 - <<'PY'
import os

import cv2
import numpy as np

assert cv2.videoio_registry.hasBackend(cv2.CAP_GSTREAMER)
info = cv2.getBuildInformation()
assert "YES" in info.split("GStreamer:", 1)[1].splitlines()[0]
assert "YES" in info.split("FFMPEG:", 1)[1].splitlines()[0]

require_cuda_build = os.environ["OPENCV_REQUIRE_CUDA_BUILD"].lower() == "true"
require_cuda_runtime = os.environ["OPENCV_REQUIRE_CUDA_RUNTIME"].lower() == "true"
expected_cuda_arches = os.environ["OPENCV_EXPECTED_CUDA_ARCH_BIN"]
cuda_section = info.split("NVIDIA CUDA:", 1)
cuda_built = len(cuda_section) == 2 and "YES" in cuda_section[1].splitlines()[0]
if require_cuda_build:
    assert cuda_built
    assert hasattr(cv2, "cuda_GpuMat")
    assert hasattr(cv2.cuda, "cvtColor")
    assert hasattr(cv2.cuda, "demosaicing")
    assert hasattr(cv2.cuda, "resize")
    assert hasattr(cv2.cuda, "createGaussianFilter")
if expected_cuda_arches:
    built_arches = (
        info.split("NVIDIA GPU arch:", 1)[1].splitlines()[0].split()
    )
    for expected_arch in expected_cuda_arches.replace(",", ";").split(";"):
        expected_arch = expected_arch.strip().replace(".", "")
        if expected_arch:
            assert expected_arch in built_arches

gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
assert cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).shape == (4, 4, 3)
assert cv2.cvtColor(
    np.zeros((4, 4, 4), dtype=np.uint8), cv2.COLOR_BGRA2BGR
).shape == (4, 4, 3)
for dtype in (np.uint8, np.uint16):
    bayer = np.arange(64, dtype=dtype).reshape(8, 8)
    for conversion in (
        cv2.COLOR_BAYER_BG2BGR,
        cv2.COLOR_BAYER_GB2BGR,
        cv2.COLOR_BAYER_RG2BGR,
        cv2.COLOR_BAYER_GR2BGR,
    ):
        converted = cv2.cvtColor(bayer, conversion)
        assert converted.shape == (8, 8, 3)
        assert converted.dtype == dtype
assert cv2.cvtColor(
    np.zeros((2, 4, 2), dtype=np.uint8), cv2.COLOR_YUV2BGR_YUY2
).shape == (2, 4, 3)
assert cv2.resize(gray, (2, 2), interpolation=cv2.INTER_AREA).shape == (2, 2)

frame = np.zeros((4, 4, 3), dtype=np.uint8)
lut = np.arange(256, dtype=np.uint8)
assert cv2.LUT(frame, lut).shape == frame.shape
assert cv2.imencode(".jpg", frame)[0]
assert cv2.SIFT_create() is not None
assert cv2.QRCodeDetector() is not None
assert cv2.createBackgroundSubtractorMOG2() is not None

cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount() if cuda_built else 0
if require_cuda_runtime:
    assert cuda_device_count > 0
if cuda_device_count > 0:
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)
    gray_cuda = cv2.cuda_GpuMat()
    gray_cuda.upload(gray)
    assert np.array_equal(gray_cuda.download(), gray)

    color_cuda = cv2.cuda.cvtColor(gray_cuda, cv2.COLOR_GRAY2BGR)
    color = color_cuda.download()
    assert color.shape == (8, 8, 3)
    assert np.array_equal(color[:, :, 0], gray)
    assert np.array_equal(color[:, :, 1], gray)
    assert np.array_equal(color[:, :, 2], gray)

    for dtype in (np.uint8, np.uint16):
        bayer_cuda = cv2.cuda_GpuMat()
        bayer_cuda.upload(np.arange(64, dtype=dtype).reshape(8, 8))
        for conversion in (
            cv2.COLOR_BAYER_BG2BGR,
            cv2.COLOR_BAYER_GB2BGR,
            cv2.COLOR_BAYER_RG2BGR,
            cv2.COLOR_BAYER_GR2BGR,
        ):
            demosaiced_cuda = cv2.cuda.demosaicing(
                bayer_cuda, conversion
            )
            converted = demosaiced_cuda.download()
            assert converted.shape == (8, 8, 3)
            assert converted.dtype == dtype
            expected = cv2.cvtColor(bayer_cuda.download(), conversion)
            assert np.allclose(converted, expected, atol=1)

    resized_cuda = cv2.cuda.resize(
        gray_cuda, (4, 4), interpolation=cv2.INTER_NEAREST
    )
    resized = resized_cuda.download()
    assert resized.shape == (4, 4)
    assert np.array_equal(
        resized,
        cv2.resize(gray, (4, 4), interpolation=cv2.INTER_NEAREST),
    )

    gaussian = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC1, cv2.CV_8UC1, (3, 3), 0
    )
    filtered = gaussian.apply(gray_cuda).download()
    assert filtered.shape == (8, 8)
    assert np.allclose(
        filtered,
        cv2.GaussianBlur(gray, (3, 3), 0, borderType=cv2.BORDER_REFLECT_101),
        atol=1,
    )

capture = cv2.VideoCapture(
    "videotestsrc num-buffers=1 ! "
    "video/x-raw,format=BGR,width=16,height=16 ! appsink",
    cv2.CAP_GSTREAMER,
)
assert capture.isOpened()
ok, captured_frame = capture.read()
capture.release()
assert ok and captured_frame.shape == (16, 16, 3)
PY
