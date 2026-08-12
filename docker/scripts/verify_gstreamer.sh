#!/bin/sh

set -eu

GSTREAMER_REQUIRE_NVCODEC="${GSTREAMER_REQUIRE_NVCODEC:-false}"
GSTREAMER_REQUIRE_NVCODEC_NVRTC="${GSTREAMER_REQUIRE_NVCODEC_NVRTC:-false}"
GSTREAMER_REQUIRE_NVCODEC_RUNTIME="${GSTREAMER_REQUIRE_NVCODEC_RUNTIME:-false}"

for element in \
    appsink \
    appsrc \
    avdec_h264 \
    avdec_h265 \
    avidemux \
    bayer2rgb \
    capssetter \
    capsfilter \
    curlhttpsrc \
    decodebin \
    decodebin3 \
    dtlssrtpdec \
    dtlssrtpenc \
    fakesink \
    filesrc \
    h264parse \
    h265parse \
    jpegdec \
    jpegenc \
    jpegparse \
    matroskademux \
    nicesink \
    nicesrc \
    opusdec \
    opusenc \
    parsebin \
    qtdemux \
    queue \
    rtpbin \
    rtph264depay \
    rtph264pay \
    rtph265depay \
    rtph265pay \
    rtpjpegdepay \
    rtpjpegpay \
    rtpjitterbuffer \
    rtpopusdepay \
    rtpopuspay \
    rtpvp8depay \
    rtpvp8pay \
    rtpvp9depay \
    rtpvp9pay \
    rtmp2sink \
    rtmp2src \
    rtspclientsink \
    rtspsrc \
    sctpdec \
    sctpenc \
    srtpdec \
    srtpenc \
    tcpclientsrc \
    tcpserversink \
    tee \
    udpsink \
    udpsrc \
    uridecodebin \
    videoconvert \
    videorate \
    videoscale \
    videotestsrc \
    vp8dec \
    vp8enc \
    vp9dec \
    vp9enc \
    webrtcbin
do
    gst-inspect-1.0 "${element}" >/dev/null
done

test -e /opt/gstreamer/lib/libgstrtspserver-1.0.so.0
for typelib in Gst GstAllocators GstApp GstRtp GstRtsp GstSdp GstVideo GstWebRTC; do
    test -s "/opt/gstreamer/lib/girepository-1.0/${typelib}-1.0.typelib"
done
ldd /opt/gstreamer/lib/gstreamer-1.0/libgstlibav.so |
    grep -q '/opt/ffmpeg/lib/libavcodec'
test -s /etc/ssl/certs/ca-certificates.crt
test -e /opt/gstreamer/lib/gio/modules/libgiognutls.so
grep -q 'libgiognutls' /opt/gstreamer/lib/gio/modules/giomodule.cache
ldd /opt/gstreamer/lib/gio/modules/libgiognutls.so |
    grep -q 'libgnutls.so'
gst-inspect-1.0 rtspsrc | grep -Eq '^[[:space:]]+rtsps$'

if [ "${GSTREAMER_REQUIRE_NVCODEC}" = "true" ]; then
    test -e /opt/gstreamer/lib/gstreamer-1.0/libgstnvcodec.so
    if ldd /opt/gstreamer/lib/gstreamer-1.0/libgstnvcodec.so | grep -q 'not found'; then
        exit 1
    fi
fi

# GPU-FREE guard for the nvcodec CUDA converters. cudaconvert / cudascale /
# cudaconvertscale JIT their kernels through NVRTC and gstcudanvrtc.cpp dlopen()s
# the UNVERSIONED soname "libnvrtc.so", which CUDA *runtime* base images do not
# ship (it belongs to cuda-nvrtc-dev). When it is missing those three elements
# are silently not registered while nvh264dec / cudaupload / cudadownload still
# are, so GSTREAMER_REQUIRE_NVCODEC (a link-level check) keeps passing and the
# regression only shows up as a CPU-decode fallback at run time on a real GPU.
# This check needs no GPU: it is exactly the dlopen the plugin performs.
if [ "${GSTREAMER_REQUIRE_NVCODEC_NVRTC}" = "true" ]; then
    if ! python3 -c "import ctypes; ctypes.CDLL('libnvrtc.so')" >/dev/null 2>&1; then
        echo "libnvrtc.so (unversioned soname) is not loadable - GStreamer" >&2
        echo "cudaconvert/cudascale/cudaconvertscale will not register and all" >&2
        echo "hardware decoding will silently fall back to the cv2 CPU decoder." >&2
        echo "Symlink it onto the versioned library shipped by the CUDA runtime." >&2
        exit 1
    fi
fi

# Full element-registration + pipeline smoke test. Requires a real GPU (the
# nvcodec plugin only registers elements after cuInit succeeds and a device is
# enumerated), so it CANNOT run on a GPU-less build machine - it is driven from
# the GPU CI job (.github/workflows/test.nvidia_t4.yml) against the built image.
if [ "${GSTREAMER_REQUIRE_NVCODEC_RUNTIME}" = "true" ]; then
    for element in \
        cudaconvertscale \
        cudadownload \
        cudaupload \
        nvh264dec \
        nvh264enc \
        nvh265dec \
        nvh265enc \
        nvjpegdec \
        nvjpegenc
    do
        gst-inspect-1.0 "${element}" >/dev/null
    done
    gst-launch-1.0 -q \
        videotestsrc num-buffers=1 ! \
        video/x-raw,format=I420,width=64,height=64 ! \
        nvjpegenc ! \
        filesink location=/tmp/nvjpeg-smoke.jpg
    gst-launch-1.0 -q \
        filesrc location=/tmp/nvjpeg-smoke.jpg ! \
        jpegparse ! \
        nvjpegdec ! \
        fakesink
    gst-launch-1.0 -q \
        videotestsrc num-buffers=1 ! \
        video/x-raw,format=I420,width=64,height=64 ! \
        cudaupload ! \
        cudaconvertscale ! \
        'video/x-raw(memory:CUDAMemory),format=RGBP' ! \
        cudadownload ! \
        fakesink
    rm /tmp/nvjpeg-smoke.jpg
fi

gst-launch-1.0 -q \
    videotestsrc num-buffers=1 ! \
    video/x-raw,format=BGR,width=16,height=16 ! \
    appsink max-buffers=1 drop=true sync=false wait-on-eos=false

for pattern in bggr gbrg grbg rggb; do
    gst-launch-1.0 -q \
        filesrc location=/dev/zero blocksize=64 num-buffers=1 ! \
        "video/x-bayer,format=${pattern},width=8,height=8,framerate=1/1" ! \
        bayer2rgb ! \
        fakesink
    gst-launch-1.0 -q \
        filesrc location=/dev/zero blocksize=128 num-buffers=1 ! \
        "video/x-bayer,format=${pattern}16le,width=8,height=8,framerate=1/1" ! \
        bayer2rgb ! \
        fakesink
done
