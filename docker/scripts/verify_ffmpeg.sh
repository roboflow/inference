#!/bin/sh

set -eu

decoders="aac av1 h264 hevc mjpeg mpeg4 opus vp8 vp9"
decoder_list="$(ffmpeg -hide_banner -decoders 2>/dev/null)"
for decoder in ${decoders}
do
    printf '%s\n' "${decoder_list}" |
        awk -v name="${decoder}" '$2 == name { found = 1 } END { exit !found }'
done

demuxers="avi flv matroska mov mpegts rtsp"
demuxer_list="$(ffmpeg -hide_banner -demuxers 2>/dev/null)"
for demuxer in ${demuxers}
do
    printf '%s\n' "${demuxer_list}" |
        awk -v name="${demuxer}" '
            {
                count = split($2, names, ",")
                for (i = 1; i <= count; i++) {
                    if (names[i] == name) {
                        found = 1
                    }
                }
            }
            END { exit !found }
        '
done

protocol_list="$(ffmpeg -hide_banner -protocols 2>/dev/null)"
for protocol in file http https rtp tcp tls udp
do
    printf '%s\n' "${protocol_list}" | grep -Eq "^[[:space:]]+${protocol}$"
done

ffmpeg -hide_banner -loglevel error \
    -f lavfi \
    -i testsrc=size=16x16:rate=1 \
    -frames:v 1 \
    -c:v ffv1 \
    -y /tmp/ffmpeg-smoke.mkv
ffprobe -v error \
    -select_streams v:0 \
    -show_entries stream=codec_name,width,height \
    -of default=noprint_wrappers=1 \
    /tmp/ffmpeg-smoke.mkv | grep -q '^codec_name=ffv1$'
rm /tmp/ffmpeg-smoke.mkv
