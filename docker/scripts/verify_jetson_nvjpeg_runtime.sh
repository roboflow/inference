#!/bin/sh

set -eu

working_directory="$(mktemp -d)"
trap 'rm -rf "${working_directory}"' 0 HUP INT TERM

jpeg_path="${working_directory}/nvjpeg-smoke.jpg"

timeout --signal=TERM --kill-after=5 30 \
    gst-launch-1.0 -q \
        videotestsrc num-buffers=1 pattern=red ! \
        video/x-raw,format=I420,width=320,height=180,framerate=1/1 ! \
        nvvidconv ! \
        'video/x-raw(memory:NVMM),format=I420' ! \
        nvjpegenc ! \
        filesink location="${jpeg_path}"

test -s "${jpeg_path}"

timeout --signal=TERM --kill-after=5 30 \
    gst-launch-1.0 -q \
        filesrc location="${jpeg_path}" ! \
        jpegparse ! \
        nvjpegdec ! \
        fakesink
