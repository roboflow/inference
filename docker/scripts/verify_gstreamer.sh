#!/bin/sh

set -eu

GSTREAMER_REQUIRE_NVCODEC="${GSTREAMER_REQUIRE_NVCODEC:-false}"
GSTREAMER_REQUIRE_NVCODEC_RUNTIME="${GSTREAMER_REQUIRE_NVCODEC_RUNTIME:-false}"

for element in \
    appsink \
    appsrc \
    avdec_h264 \
    avdec_h265 \
    avidemux \
    bayer2rgb \
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
