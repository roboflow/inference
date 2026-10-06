#!/bin/sh

set -eux

FFMPEG_VERSION="${FFMPEG_VERSION:-7.1.3}"
FFMPEG_SHA256="${FFMPEG_SHA256:?FFMPEG_SHA256 is required}"
FFMPEG_PREFIX="${FFMPEG_PREFIX:-/opt/ffmpeg}"
FFMPEG_SOURCE_DIR="${FFMPEG_SOURCE_DIR:-/tmp/ffmpeg-src}"

mkdir -p "${FFMPEG_SOURCE_DIR}"
ffmpeg_archive="/tmp/ffmpeg-${FFMPEG_VERSION}.tar.xz"
curl \
    --fail \
    --location \
    --retry 5 \
    --retry-all-errors \
    --retry-delay 2 \
    "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
    -o "${ffmpeg_archive}"
echo "${FFMPEG_SHA256}  ${ffmpeg_archive}" | sha256sum -c -
tar -xJf "${ffmpeg_archive}" \
    --strip-components=1 \
    -C "${FFMPEG_SOURCE_DIR}"

cd "${FFMPEG_SOURCE_DIR}"
./configure \
    --prefix="${FFMPEG_PREFIX}" \
    --enable-shared \
    --disable-static \
    --enable-pic \
    --disable-autodetect \
    --enable-bzlib \
    --enable-lzma \
    --enable-openssl \
    --enable-pthreads \
    --enable-zlib \
    --disable-debug \
    --disable-doc \
    --disable-ffplay \
    --extra-cflags=-O3 \
    --extra-ldflags="-Wl,-rpath,${FFMPEG_PREFIX}/lib"

make -j"$(nproc)"
make install

mkdir -p "${FFMPEG_PREFIX}/share/licenses/ffmpeg"
cp COPYING.LGPLv2.1 "${FFMPEG_PREFIX}/share/licenses/ffmpeg/"
find "${FFMPEG_PREFIX}/bin" -type f -exec strip --strip-unneeded {} +
find "${FFMPEG_PREFIX}/lib" -type f -name '*.so*' -exec strip --strip-unneeded {} +

runtime_prefix="${FFMPEG_PREFIX}-runtime"
cp -a "${FFMPEG_PREFIX}" "${runtime_prefix}"
rm -rf \
    "${runtime_prefix}/include" \
    "${runtime_prefix}/lib/pkgconfig" \
    "${runtime_prefix}/share/ffmpeg/examples"
find "${runtime_prefix}" -type f \( -name '*.a' -o -name '*.la' \) -delete

rm -rf "${ffmpeg_archive}" "${FFMPEG_SOURCE_DIR}"
