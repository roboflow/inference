#!/bin/sh

set -eux

GSTREAMER_VERSION="${GSTREAMER_VERSION:-1.24.12}"
GSTREAMER_COMMIT="${GSTREAMER_COMMIT:?GSTREAMER_COMMIT is required}"
GSTREAMER_PREFIX="${GSTREAMER_PREFIX:-/opt/gstreamer}"
GSTREAMER_NVCODEC="${GSTREAMER_NVCODEC:-disabled}"
GSTREAMER_NVRTC_COMPUTE_CAPABILITY="${GSTREAMER_NVRTC_COMPUTE_CAPABILITY:-75}"
GSTREAMER_SOURCE_DIR="${GSTREAMER_SOURCE_DIR:-/tmp/gstreamer-src}"
GSTREAMER_BUILD_DIR="${GSTREAMER_BUILD_DIR:-/tmp/gstreamer-build}"
GLIB_NETWORKING_VERSION="${GLIB_NETWORKING_VERSION:-2.72.2}"
GLIB_NETWORKING_SHA256="${GLIB_NETWORKING_SHA256:?GLIB_NETWORKING_SHA256 is required}"
GLIB_NETWORKING_SERIES="${GLIB_NETWORKING_VERSION%.*}"
GLIB_NETWORKING_SOURCE_DIR="${GLIB_NETWORKING_SOURCE_DIR:-/tmp/glib-networking-src}"
GLIB_NETWORKING_BUILD_DIR="${GLIB_NETWORKING_BUILD_DIR:-/tmp/glib-networking-build}"

case "${GSTREAMER_NVCODEC}" in
    enabled|disabled) ;;
    *)
        echo "GSTREAMER_NVCODEC must be enabled or disabled" >&2
        exit 2
        ;;
esac

git clone \
    --branch "${GSTREAMER_VERSION}" \
    --depth 1 \
    https://gitlab.freedesktop.org/gstreamer/gstreamer.git \
    "${GSTREAMER_SOURCE_DIR}"
test "$(git -C "${GSTREAMER_SOURCE_DIR}" rev-parse HEAD)" = \
    "${GSTREAMER_COMMIT}"

if [ "${GSTREAMER_NVCODEC}" = "enabled" ]; then
    nvrtc_source="${GSTREAMER_SOURCE_DIR}/subprojects/gst-plugins-bad/gst-libs/gst/cuda/gstcudanvrtc.cpp"
    test "$(grep -c -- '--gpu-architecture=compute_30' "${nvrtc_source}")" -eq 1
    test "$(grep -c -- '--gpu-architecture=compute_52' "${nvrtc_source}")" -eq 1
    sed -i \
        -e "s/--gpu-architecture=compute_30/--gpu-architecture=compute_${GSTREAMER_NVRTC_COMPUTE_CAPABILITY}/" \
        -e "s/--gpu-architecture=compute_52/--gpu-architecture=compute_${GSTREAMER_NVRTC_COMPUTE_CAPABILITY}/" \
        "${nvrtc_source}"
fi

meson setup "${GSTREAMER_BUILD_DIR}" "${GSTREAMER_SOURCE_DIR}" \
    --buildtype=release \
    --prefix="${GSTREAMER_PREFIX}" \
    --libdir=lib \
    --strip \
    --wrap-mode=default \
    --force-fallback-for=libnice \
    -Dauto_features=disabled \
    -Db_ndebug=true \
    -Dbase=enabled \
    -Dgood=enabled \
    -Dbad=enabled \
    -Dugly=disabled \
    -Dlibav=enabled \
    -Drtsp_server=enabled \
    -Ddevtools=disabled \
    -Dges=disabled \
    -Drs=disabled \
    -Dvaapi=disabled \
    -Dpython=disabled \
    -Dsharp=disabled \
    -Dlibnice=enabled \
    -Dgst-full=disabled \
    -Dgst-examples=disabled \
    -Dgpl=disabled \
    -Dtests=disabled \
    -Dtools=enabled \
    -Dexamples=disabled \
    -Dintrospection=enabled \
    -Dnls=disabled \
    -Dorc=enabled \
    -Dorc-source=subproject \
    -Dqt5=disabled \
    -Dqt6=disabled \
    -Dwebrtc=enabled \
    -Dgst-plugins-base:app=enabled \
    -Dgst-plugins-base:audioconvert=enabled \
    -Dgst-plugins-base:audioresample=enabled \
    -Dgst-plugins-base:encoding=enabled \
    -Dgst-plugins-base:gio=enabled \
    -Dgst-plugins-base:gio-typefinder=enabled \
    -Dgst-plugins-base:opus=enabled \
    -Dgst-plugins-base:playback=enabled \
    -Dgst-plugins-base:rawparse=enabled \
    -Dgst-plugins-base:tcp=enabled \
    -Dgst-plugins-base:typefind=enabled \
    -Dgst-plugins-base:videoconvertscale=enabled \
    -Dgst-plugins-base:videorate=enabled \
    -Dgst-plugins-base:videotestsrc=enabled \
    -Dgst-plugins-good:autodetect=enabled \
    -Dgst-plugins-good:avi=enabled \
    -Dgst-plugins-good:flv=enabled \
    -Dgst-plugins-good:isomp4=enabled \
    -Dgst-plugins-good:jpeg=enabled \
    -Dgst-plugins-good:matroska=enabled \
    -Dgst-plugins-good:multifile=enabled \
    -Dgst-plugins-good:multipart=enabled \
    -Dgst-plugins-good:rtp=enabled \
    -Dgst-plugins-good:rtpmanager=enabled \
    -Dgst-plugins-good:rtsp=enabled \
    -Dgst-plugins-good:udp=enabled \
    -Dgst-plugins-good:v4l2=enabled \
    -Dgst-plugins-good:vpx=enabled \
    -Dgst-plugins-good:wavparse=enabled \
    -Dgst-plugins-bad:curl=enabled \
    -Dgst-plugins-bad:bayer=enabled \
    -Dgst-plugins-bad:dtls=enabled \
    -Dgst-plugins-bad:jpegformat=enabled \
    -Dgst-plugins-bad:mpegtsdemux=enabled \
    -Dgst-plugins-bad:nvcodec="${GSTREAMER_NVCODEC}" \
    -Dgst-plugins-bad:rtmp2=enabled \
    -Dgst-plugins-bad:rtp=enabled \
    -Dgst-plugins-bad:sctp=enabled \
    -Dgst-plugins-bad:sdp=enabled \
    -Dgst-plugins-bad:srtp=enabled \
    -Dgst-plugins-bad:videoparsers=enabled \
    -Dgst-plugins-bad:webrtc=enabled \
    -Dgst-rtsp-server:rtspclientsink=enabled \
    -Dlibnice:crypto-library=openssl \
    -Dlibnice:gupnp=disabled

meson compile -C "${GSTREAMER_BUILD_DIR}"
meson install -C "${GSTREAMER_BUILD_DIR}"

mkdir -p "${GLIB_NETWORKING_SOURCE_DIR}"
glib_networking_archive="/tmp/glib-networking-${GLIB_NETWORKING_VERSION}.tar.xz"
curl \
    --fail \
    --location \
    --retry 5 \
    --retry-all-errors \
    --retry-delay 2 \
    "https://download.gnome.org/sources/glib-networking/${GLIB_NETWORKING_SERIES}/glib-networking-${GLIB_NETWORKING_VERSION}.tar.xz" \
    -o "${glib_networking_archive}"
echo "${GLIB_NETWORKING_SHA256}  ${glib_networking_archive}" | sha256sum -c -
tar -xJf "${glib_networking_archive}" \
    --strip-components=1 \
    -C "${GLIB_NETWORKING_SOURCE_DIR}"

meson setup "${GLIB_NETWORKING_BUILD_DIR}" "${GLIB_NETWORKING_SOURCE_DIR}" \
    --buildtype=release \
    --prefix="${GSTREAMER_PREFIX}" \
    --libdir=lib \
    --strip \
    -Db_ndebug=true \
    -Dgnome_proxy=disabled \
    -Dgnutls=enabled \
    -Dinstalled_tests=false \
    -Dlibproxy=disabled \
    -Dopenssl=disabled \
    -Dstatic_modules=false
meson compile -C "${GLIB_NETWORKING_BUILD_DIR}"
meson install -C "${GLIB_NETWORKING_BUILD_DIR}"
gio-querymodules "${GSTREAMER_PREFIX}/lib/gio/modules"

license_dir="${GSTREAMER_PREFIX}/share/licenses/gstreamer"
mkdir -p "${license_dir}"
cp "${GSTREAMER_SOURCE_DIR}/LICENSE" "${license_dir}/LICENSE"
for project in \
    gst-libav \
    gst-plugins-bad \
    gst-plugins-base \
    gst-plugins-good \
    gst-rtsp-server \
    gstreamer
do
    cp "${GSTREAMER_SOURCE_DIR}/subprojects/${project}/COPYING" \
        "${license_dir}/COPYING.${project}"
done
cp "${GSTREAMER_SOURCE_DIR}/subprojects/libnice/COPYING" \
    "${license_dir}/COPYING.libnice"
cp "${GSTREAMER_SOURCE_DIR}/subprojects/orc/COPYING" \
    "${license_dir}/COPYING.orc"
cp "${GSTREAMER_SOURCE_DIR}/subprojects/gst-plugins-bad/ext/sctp/usrsctp/LICENSE.md" \
    "${license_dir}/LICENSE.usrsctp"
cp "${GLIB_NETWORKING_SOURCE_DIR}/COPYING" \
    "${license_dir}/COPYING.glib-networking"
cp "${GLIB_NETWORKING_SOURCE_DIR}/LICENSE_EXCEPTION" \
    "${license_dir}/LICENSE_EXCEPTION.glib-networking"

runtime_prefix="${GSTREAMER_PREFIX}-runtime"
cp -a "${GSTREAMER_PREFIX}" "${runtime_prefix}"
rm -rf \
    "${runtime_prefix}/include" \
    "${runtime_prefix}/lib/pkgconfig" \
    "${runtime_prefix}/share/aclocal" \
    "${runtime_prefix}/share/gir-1.0" \
    "${runtime_prefix}/share/gtk-doc" \
    "${runtime_prefix}/share/man"
find "${runtime_prefix}" -type f \( -name '*.a' -o -name '*.la' \) -delete

rm -rf \
    "${GSTREAMER_SOURCE_DIR}" \
    "${GSTREAMER_BUILD_DIR}" \
    "${glib_networking_archive}" \
    "${GLIB_NETWORKING_SOURCE_DIR}" \
    "${GLIB_NETWORKING_BUILD_DIR}"
