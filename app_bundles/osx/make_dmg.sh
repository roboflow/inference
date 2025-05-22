#!/bin/bash
set -euo pipefail

APP_NAME="Roboflow Inference"
APP_BUNDLE="dist/${APP_NAME}.app"
DMG_NAME="${APP_NAME// /-}.dmg"
SPARSE_IMAGE="temp.dmg.sparseimage"
VOLUME_NAME="${APP_NAME}"
MOUNT_POINT="/Volumes/${VOLUME_NAME}"
IDENTITY="Developer ID Application: Roboflow, LLC (7SBQ39NG7G)"

echo "💾 Creating sparse DMG..."
hdiutil create -size 3g -fs HFS+ -volname "$VOLUME_NAME" -type SPARSE -ov "$SPARSE_IMAGE"

echo "🔌 Mounting..."
# Capture full stdout so we can parse the device node
ATTACH_OUT=$(hdiutil attach "$SPARSE_IMAGE" -mountpoint "$MOUNT_POINT" -owners on -nobrowse)
DEVICE=$(echo "$ATTACH_OUT" | awk '/^\/dev\// {print $1; exit}')
echo "    Mounted at $MOUNT_POINT on $DEVICE"

echo "📁 Copying app bundle..."
rsync -a "${APP_BUNDLE}/" "$MOUNT_POINT/${APP_NAME}.app/"

echo "🌄 Adding background image..."
mkdir -p "$MOUNT_POINT/.background"
cp background.png "$MOUNT_POINT/.background/background.png"

echo "🔗 Adding Applications alias..."
ln -s /Applications "$MOUNT_POINT/Applications"

echo "📎 Copying DS_Store..."
cp DMG-DS_Store "$MOUNT_POINT/.DS_Store"

echo "🛑 Detaching volume..."
for i in {1..5}; do
  if hdiutil detach "$DEVICE" -quiet 2>/dev/null; then
      echo "    Detached on try $i"
      break
  fi
  echo "    Volume busy, retrying ($i/5)…"
  sleep 2
  if [[ $i -eq 5 ]]; then
      echo "    Forcing detach"
      hdiutil detach "$DEVICE" -force
  fi
done

echo "📦 Converting to compressed DMG..."
hdiutil convert "$SPARSE_IMAGE" -format UDZO -imagekey zlib-level=9 -o "$DMG_NAME"

echo "🧹 Cleaning up sparse image..."
rm "$SPARSE_IMAGE"

echo "🔐 Signing DMG with identity: $IDENTITY"
codesign --sign "$IDENTITY" --timestamp --options runtime "$DMG_NAME"

echo "✅ Signed DMG created at: $DMG_NAME"
