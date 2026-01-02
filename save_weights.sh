#!/usr/bin/env bash
set -euo pipefail

# ====== CONFIG ======
BUCKET="roboflow-staging-core-models"
PREFIX="lora-bases/qwen/qwen3vl-2b-instruct/main"
OBJ_URI="gs://${BUCKET}/${PREFIX}"
ARCHIVE_NAME="weights.tar.gz"

# Work in a fresh temp dir
WORKDIR="$(mktemp -d)"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

echo "Workdir: $WORKDIR"
cd "$WORKDIR"

# ====== SAFETY SETTINGS ======
# Disable parallel composite uploads (splitting file into chunks) for big archives
# This avoids subtle corruption / incompatibilities for large gzip streams.
GSUTIL_SAFE_OPTS=(
  -o "GSUtil:parallel_composite_upload_threshold=0"
)

# Optional: ensure gsutil uses checksums (it does by default on cp, but keeping verification explicit)
# We'll also compute SHA256 locally to verify round trips.

# ====== 1) DOWNLOAD ALL CONTENTS ======
echo "Downloading: ${OBJ_URI}/ -> $WORKDIR/main"
mkdir -p main
# -m for concurrency, cp -r for recursive
gsutil -m "${GSUTIL_SAFE_OPTS[@]}" cp -r "${OBJ_URI}/*" main/

echo "Downloaded files:"
du -sh main || true

# ====== 2) CREATE weights.tar.gz FROM CONTENTS (no top-level 'main' folder) ======
echo "Creating archive: $ARCHIVE_NAME"
tar -C main -czf "$ARCHIVE_NAME" .

# Verify archive integrity (critical)
echo "Verifying gzip stream + tar listing..."
gzip -t "$ARCHIVE_NAME"
tar -tzf "$ARCHIVE_NAME" >/dev/null

echo "Archive size:"
ls -lh "$ARCHIVE_NAME"

# Create checksum for extra safety
sha256sum "$ARCHIVE_NAME" > "${ARCHIVE_NAME}.sha256"

# ====== 3) UPLOAD weights.tar.gz BACK ======
echo "Uploading archive back to: ${OBJ_URI}/$ARCHIVE_NAME"
gsutil "${GSUTIL_SAFE_OPTS[@]}" cp "$ARCHIVE_NAME" "${OBJ_URI}/${ARCHIVE_NAME}"
gsutil "${GSUTIL_SAFE_OPTS[@]}" cp "${ARCHIVE_NAME}.sha256" "${OBJ_URI}/${ARCHIVE_NAME}.sha256"

# ====== 4) VERIFY REMOTE OBJECT INTEGRITY (download to temp and compare sha256) ======
echo "Verifying by downloading back and comparing sha256..."
mkdir -p verify
gsutil -m "${GSUTIL_SAFE_OPTS[@]}" cp "${OBJ_URI}/${ARCHIVE_NAME}" "verify/${ARCHIVE_NAME}"
sha256sum -c "${ARCHIVE_NAME}.sha256" --ignore-missing 2>/dev/null || true

# Compare local vs re-downloaded checksum explicitly
LOCAL_SHA="$(cut -d' ' -f1 < "${ARCHIVE_NAME}.sha256")"
DOWN_SHA="$(sha256sum "verify/${ARCHIVE_NAME}" | cut -d' ' -f1)"

if [[ "$LOCAL_SHA" != "$DOWN_SHA" ]]; then
  echo "ERROR: Checksum mismatch after upload/download!"
  echo "Local:     $LOCAL_SHA"
  echo "Downloaded:$DOWN_SHA"
  exit 1
fi

# Verify the downloaded copy also extracts cleanly
gzip -t "verify/${ARCHIVE_NAME}"
tar -tzf "verify/${ARCHIVE_NAME}" >/dev/null

echo "SUCCESS: Downloaded -> archived -> uploaded -> verified OK."
echo "Remote archive: ${OBJ_URI}/${ARCHIVE_NAME}"
