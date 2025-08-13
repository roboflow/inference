# macOS Inference App Build Script
import os
import shutil
import subprocess
import sys
import glob
import argparse

import importlib.util

ENTRY_SCRIPT = "run_inference.py"
BUILD_NAME = "inference-app"
APP_NAME = "Roboflow Inference"
ICON_PNG = "../osx/app-icon.png"
ICNS_PATH = "./app-icon.icns"
SPEC_FILE = f"{BUILD_NAME}.spec"

# Get version from environment variable, default if not set
VERSION = os.getenv("BUILD_VERSION")
if not VERSION:
    print("Warning: BUILD_VERSION environment variable not set. Defaulting to 0.0.0-dev.")
    VERSION = "0.0.0-dev"
else:
    print(f"Using version: {VERSION}")

BUILD_DIR = os.path.join("dist", BUILD_NAME)
APP_PATH = os.path.join("dist", f"{APP_NAME}.app")
ZIP_PATH = f"{APP_NAME.replace(' ', '-')}-{VERSION}.zip"
DMG_PATH = f"{APP_NAME.replace(' ', '-')}-{VERSION}.dmg"
BACKGROUND_PNG = os.path.abspath("background.png")
DS_STORE_SOURCE = os.path.abspath("DMG-DS_Store")
SOURCE_LANDING_DIR = "../../inference/landing"
DEST_LANDING_DIR = os.path.join(BUILD_DIR, "inference", "landing")
LAUNCHER_BINARY_PATH = os.path.abspath("launcher") #need to compile this first from launcher.c if its not already compiled

CODESIGN_IDENTITY = "Developer ID Application: Roboflow, LLC (7SBQ39NG7G)"
NOTARY_PROFILE = "roboflow-notary"


def parse_args():
    parser = argparse.ArgumentParser(description="Build the macOS Inference app bundle")
    parser.add_argument(
        "--skip-codesign",
        action="store_true",
        help="Skip codesigning and notarization steps",
    )
    parser.add_argument(
        "--skip-sign",
        action="store_true",
        help="Skip codesigning step only (useful for dev testing)",
    )
    parser.add_argument(
        "--skip-notarize",
        action="store_true",
        help="Skip notarization step only (useful for dev testing)",
    )
    parser.add_argument(
        "--skip-dmg",
        action="store_true",
        help="Skip DMG creation step",
    )
    parser.add_argument(
        "--adhoc-sign",
        action="store_true",
        help="Use ad-hoc signing identity ('-') instead of Developer ID (cannot be notarized)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip codesign, notarize, and DMG creation (alias for dev fast path)",
    )
    return parser.parse_args()


def clean():
    print("\U0001f9f9 Cleaning old build...")
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("dist", ignore_errors=True)
    for f in [ICNS_PATH, ZIP_PATH, DMG_PATH]:
        if os.path.exists(f):
            os.remove(f)

def generate_icns_from_png(png_path, icns_path):
    print("\U0001f3a8 Generating .icns from PNG...")
    iconset_dir = "app-icon.iconset"
    os.makedirs(iconset_dir, exist_ok=True)

    sizes = [
        (16, 1), (16, 2), (32, 1), (32, 2),
        (128, 1), (128, 2), (256, 1), (256, 2),
        (512, 1), (512, 2)
    ]

    for size, scale in sizes:
        suffix = "" if scale == 1 else "@2x"
        filename = f"icon_{size}x{size}{suffix}.png"
        output_path = os.path.join(iconset_dir, filename)
        actual_size = size * scale
        subprocess.run(["sips", "-z", str(actual_size), str(actual_size), png_path, "--out", output_path], check=True)

    subprocess.run(["iconutil", "-c", "icns", iconset_dir, "-o", icns_path], check=True)
    shutil.rmtree(iconset_dir)


def build_app(icns_path):
    print("\U0001f680 Building with PyInstaller...")

    pyinstaller_cmd = [
        "pyinstaller",
        "--noconfirm",
        SPEC_FILE
    ]
    print(f"Running command: {' '.join(pyinstaller_cmd)}") # Debug print
    subprocess.run(pyinstaller_cmd, check=True)




def copy_static_files():
    print(f"\U0001f4c1 Copying static files from:\n  {SOURCE_LANDING_DIR}\n➡️  to\n  {DEST_LANDING_DIR}")
    if not os.path.exists(SOURCE_LANDING_DIR):
        raise FileNotFoundError(f"STATIC LANDING FILES SOURCE DIRECTORY NOT FOUND: {SOURCE_LANDING_DIR} not found!")

    os.makedirs(DEST_LANDING_DIR, exist_ok=True)
    shutil.copytree(SOURCE_LANDING_DIR, DEST_LANDING_DIR, dirs_exist_ok=True)



def sign_app_bundle(app_path: str, identity: str):
    print("🔒 Recursively signing all executables and libraries...")
    is_adhoc_identity = identity.strip() == "-"

    # Walk through all files in the .app bundle
    for root, dirs, files in os.walk(app_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Check if file is executable or matches .dylib/.so extension
            is_executable = os.access(file_path, os.X_OK)
            if file.endswith((".so", ".dylib")) or is_executable:
                print(f"   Signing: {file_path}")
                cmd = [
                    "codesign",
                    "--force",
                    "--options", "runtime",
                    "--sign", identity,
                ]
                if not is_adhoc_identity:
                    cmd.insert(3, "--timestamp")
                cmd.append(file_path)
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error signing {file_path}:")
                    print(f"STDOUT:\n{e.stdout}")
                    print(f"STDERR:\n{e.stderr}")
                    raise e

    print("🏷️ Signing the full .app bundle...")
    cmd_bundle = [
        "codesign",
        "--force",
        "--options", "runtime",
        "--sign", identity,
    ]
    if not is_adhoc_identity:
        cmd_bundle.insert(3, "--timestamp")
    cmd_bundle.append(app_path)
    try:
        subprocess.run(cmd_bundle, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error signing app bundle {app_path}:")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise e

    print("✅ Signing complete.")

def zip_app(app_path: str, zip_path: str):
    print(f"📦 Zipping {app_path} to {zip_path}...")
    subprocess.run([
        "ditto",
        "-c", "-k",
        "--sequesterRsrc",
        "--keepParent",
        app_path,
        zip_path
    ], check=True)

def notarize_app(zip_path):
    print("📤 Submitting to Apple for notarization...")
    subprocess.run([
        "xcrun", "notarytool", "submit", zip_path,
        "--keychain-profile", NOTARY_PROFILE,
        "--wait"
    ], check=True)

def staple_app(app_path):
    print("📎 Stapling notarization ticket...")
    subprocess.run(["xcrun", "stapler", "staple", app_path], check=True)

def codesign_and_notarize(app_path, zip_path, identity: str):
    sign_app_bundle(app_path, identity)
    zip_app(app_path, zip_path)
    notarize_app(zip_path)
    staple_app(app_path)

def create_dmg(app_path, dmg_path):
    print("💾 Creating DMG via shell script...")
    script_path = os.path.abspath("make_dmg.sh")

    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"DMG creation script not found at {script_path}")

    # Make sure the script is executable
    subprocess.run(["chmod", "+x", script_path], check=True)

    # Determine the likely unversioned output name from make_dmg.sh
    # This is based on the global APP_NAME constant: "Roboflow Inference"
    unversioned_dmg_filename = f"{APP_NAME.replace(' ', '-')}.dmg" # Should be "Roboflow-Inference.dmg"

    print(f"Running {script_path}. Expecting it to create {unversioned_dmg_filename}, which will then be ensured to be {dmg_path}.")
    subprocess.run([script_path], check=True)

    # Check if make_dmg.sh created the unversioned file
    if os.path.exists(unversioned_dmg_filename) and unversioned_dmg_filename != dmg_path:
        print(f"DMG created as {unversioned_dmg_filename}. Renaming to versioned name: {dmg_path}")
        shutil.move(unversioned_dmg_filename, dmg_path)
    elif os.path.exists(dmg_path):
        print(f"DMG successfully found at versioned path: {dmg_path} (make_dmg.sh might be version-aware or previous rename succeeded).")
    else:
        print(f"Error: DMG not found after running {script_path}. Expected {unversioned_dmg_filename} or {dmg_path}.")
        print("Contents of current directory (app_bundles/osx/):")
        try:
            for item in os.listdir("."):
                print(f"  - {item}")
        except Exception as e:
            print(f"    Could not list directory contents: {e}")
        raise FileNotFoundError(f"DMG file not found. {script_path} did not produce {unversioned_dmg_filename} or {dmg_path}.")

    print(f"✅ DMG preparation complete. Final DMG is at: {dmg_path}")



def sign_dmg(dmg_path):
    print("\U0001f512 Signing the DMG...")
    subprocess.run([
        "codesign",
        "--sign", CODESIGN_IDENTITY,
        "--timestamp",
        "--options", "runtime",
        dmg_path
    ], check=True)
    print("✅ DMG signed successfully.")




def create_app_bundle_with_native_launcher(source_dir, app_bundle_path, launcher_binary_path, icon_path=None):
    app_name = os.path.splitext(os.path.basename(app_bundle_path))[0]
    contents_dir = os.path.join(app_bundle_path, "Contents")
    macos_dir = os.path.join(contents_dir, "MacOS")
    resources_dir = os.path.join(contents_dir, "Resources")

    os.makedirs(macos_dir, exist_ok=True)
    os.makedirs(resources_dir, exist_ok=True)

    # Copy PyInstaller build into Resources
    shutil.copytree(source_dir, resources_dir, dirs_exist_ok=True)

    # Copy native launcher binary into MacOS
    shutil.copy2(launcher_binary_path, os.path.join(macos_dir, "launcher"))
    os.chmod(os.path.join(macos_dir, "launcher"), 0o755)

    # Minimal Info.plist with CFBundleExecutable = launcher
    info_plist_contents = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>            <string>{app_name}</string>
    <key>CFBundleDisplayName</key>     <string>{app_name}</string>
    <key>CFBundleExecutable</key>      <string>launcher</string>
    <key>CFBundleIdentifier</key>      <string>com.example.{app_name.lower().replace(' ','')}</string>
    <key>CFBundleVersion</key>         <string>{VERSION}</string>
    <key>CFBundlePackageType</key>     <string>APPL</string>
    <key>LSMinimumSystemVersion</key>  <string>10.13</string>
    <key>NSHighResolutionCapable</key> <true/>
    <key>CFBundleIconFile</key>        <string>app-icon.icns</string>
</dict>
</plist>
"""
    with open(os.path.join(contents_dir, "Info.plist"), "w") as f:
        f.write(info_plist_contents)

    # Copy icon if provided
    if icon_path and os.path.isfile(icon_path):
        shutil.copy2(icon_path, os.path.join(resources_dir, "app-icon.icns"))

def fix_app_permissions(app_path):
    print("🔧 Ensuring app bundle is readable by user...")
    subprocess.run(["chmod", "-R", "u+rwX", app_path], check=True)




if __name__ == "__main__":
    args = parse_args()

    # Derived flags for convenience/backwards compatibility
    skip_sign = bool(args.skip_sign or args.skip_codesign or args.fast)
    skip_notarize = bool(args.skip_notarize or args.skip_codesign or args.fast)
    skip_dmg = bool(args.skip_dmg or args.fast)

    # Determine signing identity
    adhoc_sign = bool(args.adhoc_sign and not skip_sign)
    identity = "-" if adhoc_sign else CODESIGN_IDENTITY

    # Adjust incompatible combinations
    if skip_sign and not skip_notarize:
        print("⚠️ Notarization requires a signed app. Skipping notarization because signing is skipped.")
        skip_notarize = True
    if adhoc_sign and not skip_notarize:
        print("⚠️ Ad-hoc signed apps cannot be notarized. Skipping notarization.")
        skip_notarize = True

    clean()
    generate_icns_from_png(ICON_PNG, ICNS_PATH)
    build_app(ICNS_PATH)
    copy_static_files()
    create_app_bundle_with_native_launcher(BUILD_DIR, APP_PATH, LAUNCHER_BINARY_PATH, icon_path=ICNS_PATH)
    fix_app_permissions(APP_PATH)

    # Signing and Notarization flow
    if skip_sign and skip_notarize:
        print("⏭️ Skipping codesigning and notarization as requested.")
    elif not skip_sign and not skip_notarize:
        print(f"🔏 Signing with identity: {'ad-hoc (-)' if adhoc_sign else identity}")
        codesign_and_notarize(APP_PATH, ZIP_PATH, identity)
    elif not skip_sign and skip_notarize:
        print(f"🔏 Signing only with identity: {'ad-hoc (-)' if adhoc_sign else identity}")
        sign_app_bundle(APP_PATH, identity)

    # DMG creation
    if skip_dmg:
        print("⏭️ Skipping DMG creation as requested.")
    else:
        create_dmg(APP_PATH, DMG_PATH)
        # sign_dmg(DMG_PATH) # Temporarily disabled DMG signing

    print("\n✅ macOS app build complete.")
    print(f"   App: {APP_PATH}")
    if not skip_dmg:
        print(f"   DMG: {DMG_PATH}")