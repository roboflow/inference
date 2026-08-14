#!/bin/sh

set -eu

# Shared runtime hygiene check for every media-enabled image. Jetson-only
# artifacts are validated when present; set
# MEDIA_RUNTIME_REQUIRE_JETSON_PLUGINS=true to make the baked L4T plugin set
# a hard requirement (only the Jetson media image bakes them: JetPack 6
# injects the BSP plugins at container start, and dGPU images have none).

if [ "${VERIFY_MEDIA_RUNTIME_DEVELOPMENT_TOOLS:-true}" = true ]; then
    for command_name in \
        cc \
        c++ \
        gcc \
        g++ \
        make \
        cmake \
        meson \
        ninja \
        git \
        curl \
        pip \
        pip3
    do
        if command -v "${command_name}" >/dev/null 2>&1; then
            echo "Unexpected build command in media runtime: ${command_name}" >&2
            exit 1
        fi
    done

    development_packages="$(
        dpkg-query -W -f='${binary:Package}\n' |
            grep -E -- '-dev(:[^[:space:]]+)?$' || true
    )"
    if [ -n "${development_packages}" ]; then
        printf 'Unexpected development packages in media runtime:\n%s\n' \
            "${development_packages}" >&2
        exit 1
    fi
fi

set -- /opt/ffmpeg /opt/gstreamer
if [ -d /opt/roboflow ]; then
    set -- "$@" /opt/roboflow
fi
if [ -d /opt/opencv ]; then
    set -- "$@" /opt/opencv
fi
if [ -d /opt/cuda-runtime ]; then
    set -- "$@" /opt/cuda-runtime
fi
if [ -d /usr/lib/aarch64-linux-gnu/gstreamer-1.0 ]; then
    set -- "$@" /usr/lib/aarch64-linux-gnu/gstreamer-1.0
fi
if [ -d /usr/lib/aarch64-linux-gnu/nvidia ]; then
    set -- "$@" /usr/lib/aarch64-linux-gnu/nvidia
fi
if [ -d /usr/lib/aarch64-linux-gnu/tegra ]; then
    set -- "$@" /usr/lib/aarch64-linux-gnu/tegra
fi

development_artifact="$(
    find "$@" -type f \
        \( -name '*.a' -o -name '*.la' -o -name '*.h' -o -name '*.pc' \) \
        -print -quit
)"
if [ -n "${development_artifact}" ]; then
    echo "Unexpected development artifact in media runtime: ${development_artifact}" >&2
    exit 1
fi

set -- /opt/ffmpeg /opt/gstreamer
if [ -d /opt/roboflow ]; then
    set -- "$@" /opt/roboflow
fi
if [ -d /opt/opencv ]; then
    set -- "$@" /opt/opencv
fi
if [ -d /opt/cuda-runtime ]; then
    set -- "$@" /opt/cuda-runtime
fi

required_jetson_plugins="
/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvarguscamerasrc.so
/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvjpeg.so
/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvidconv.so
/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvideo4linux2.so
"
if [ "${MEDIA_RUNTIME_REQUIRE_JETSON_PLUGINS:-false}" = "true" ]; then
    for plugin in ${required_jetson_plugins}; do
        test -s "${plugin}"
    done
fi

if [ -s /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvjpeg.so ]; then
    test -s /opt/cuda-runtime/lib/libnvjpeg.so.13
    command -v lsmod >/dev/null
fi

find "$@" -type f -name '*.so*' -print0 |
    xargs -0 sh -c '
        for library do
            missing="$(
                ldd "${library}" 2>/dev/null |
                    awk '\''/not found/ { print $1 }'\'' |
                    grep -v -x '\''libcuda.so.1'\'' || true
            )"
            if [ -n "${missing}" ]; then
                printf "%s\n%s\n" "${library}" "${missing}" >&2
                exit 1
            fi
        done
    ' sh

for library in ${required_jetson_plugins}; do
    if [ ! -s "${library}" ]; then
        continue
    fi
    missing="$(
        ldd "${library}" 2>/dev/null |
            awk '/not found/ { print $1 }' |
            grep -v -x 'libcuda.so.1' || true
    )"
    if [ -n "${missing}" ]; then
        printf '%s\n%s\n' "${library}" "${missing}" >&2
        exit 1
    fi
done
