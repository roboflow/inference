#!/usr/bin/env bash
# Fetches local dev dependencies for the video POC into ./bin (gitignored):
#   - mediamtx (RTSP ingest + WHEP preview)
#   - ffmpeg (only if not already on PATH; used by the connector)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p bin

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) MTX_ARCH="amd64" ;;
  arm64 | aarch64) [ "$OS" = "darwin" ] && MTX_ARCH="arm64" || MTX_ARCH="arm64v8" ;;
  *) echo "unsupported arch: $ARCH" && exit 1 ;;
esac

if [ ! -x bin/mediamtx ]; then
  echo "fetching latest mediamtx..."
  TAG=$(curl -fsSL https://api.github.com/repos/bluenviron/mediamtx/releases/latest | grep -o '"tag_name": *"[^"]*"' | cut -d'"' -f4)
  URL="https://github.com/bluenviron/mediamtx/releases/download/${TAG}/mediamtx_${TAG}_${OS}_${MTX_ARCH}.tar.gz"
  curl -fsSL "$URL" | tar -xz -C bin mediamtx
  echo "mediamtx ${TAG} -> bin/mediamtx"
else
  echo "mediamtx already present"
fi

if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg found on PATH: $(command -v ffmpeg)"
elif [ -x bin/ffmpeg ]; then
  echo "ffmpeg already present in bin/"
elif [ "$OS" = "darwin" ]; then
  echo "fetching static ffmpeg (evermeet.cx)..."
  curl -fsSL "https://evermeet.cx/ffmpeg/getrelease/zip" -o bin/ffmpeg.zip
  (cd bin && unzip -o -q ffmpeg.zip && rm ffmpeg.zip && chmod +x ffmpeg)
  echo "ffmpeg -> bin/ffmpeg"
else
  echo "fetching static ffmpeg (johnvansickle)..."
  FFARCH=$([ "$ARCH" = "x86_64" ] && echo amd64 || echo arm64)
  curl -fsSL "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-${FFARCH}-static.tar.xz" -o bin/ffmpeg.tar.xz
  tar -xJf bin/ffmpeg.tar.xz -C bin --strip-components=1 --wildcards '*/ffmpeg'
  rm bin/ffmpeg.tar.xz
  echo "ffmpeg -> bin/ffmpeg"
fi

echo "done."
