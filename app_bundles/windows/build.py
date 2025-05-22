# Updated build.py to use a .spec file for PyInstaller config
import os
import shutil
import subprocess
import sys
from PIL import Image

ENTRY_SCRIPT = "run_inference.py"
BUILD_NAME = "inference"
ICON_PNG = "../osx/app-icon.png"
ICON_ICO = "app-icon.ico"
SPEC_FILE = f"{BUILD_NAME}.spec"

# Paths
BUILD_DIR = os.path.join("dist", BUILD_NAME)
SOURCE_LANDING_DIR = "../../inference/landing"
DEST_LANDING_DIR = os.path.join(BUILD_DIR, "inference", "landing")


def clean():
    print("Cleaning old build...")
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("dist", ignore_errors=True)
    if os.path.exists(ICON_ICO):
        os.remove(ICON_ICO)


def convert_icon():
    print(f"Converting {ICON_PNG} to {ICON_ICO}...")
    if not os.path.exists(ICON_PNG):
        raise FileNotFoundError(f"{ICON_PNG} not found!")
    img = Image.open(ICON_PNG)
    img.save(ICON_ICO, format='ICO', sizes=[(256, 256)])


def write_inno_setup_file(version):
    print(f"Writing installer script for version {version}")
    installer_script = f"""
[Setup]
AppName=InferenceA
AppVersion={version}
DefaultDirName={{localappdata}}\\RoboflowInference
DisableProgramGroupPage=yes
OutputDir=installer
OutputBaseFilename=inference-{version}-installer
Compression=lzma
SolidCompression=no

[Files]
Source: "dist\\inference\\*"; DestDir: "{{app}}"; Flags: recursesubdirs

[Icons]
Name: "{{autoprograms}}\\InferenceApp"; Filename: "{{app}}\\inference.exe"

[Run]
Filename: "{{app}}\\inference.exe"; Description: "Launch Inference"; Flags: nowait postinstall skipifsilent
"""
    with open("inference-installer.iss", "w") as f:
        f.write(installer_script)

def run_inno_setup_compiler():
    print("Running Inno Setup Compiler...")
    
    # iscc_path = r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"  # Adjust if different
    # git github actions ISCC.exe should be in the PATH after jrmerino/setup-inno-setup action
    try:
        subprocess.run(["ISCC.exe", "inference-installer.iss"], check=True)
        print("Inno Setup compilation successful.")
    except subprocess.CalledProcessError as e:
        print(f"Inno Setup compilation failed. Error: {e}")
        raise RuntimeError("Inno Setup compilation failed.") from e
    except FileNotFoundError as e:
        print("Error: ISCC.exe not found. Make sure Inno Setup is installed and in PATH.")
        raise RuntimeError("ISCC.exe not found.") from e

def build():
    print("Building from spec file...")
    subprocess.run(["pyinstaller", "--noconfirm", SPEC_FILE], check=True)


def copy_static_files():
    print(f"Copying static files from:\n  {SOURCE_LANDING_DIR}\n  to\n  {DEST_LANDING_DIR}")
    if not os.path.exists(SOURCE_LANDING_DIR):
        raise FileNotFoundError(f"{SOURCE_LANDING_DIR} not found!")

    os.makedirs(DEST_LANDING_DIR, exist_ok=True)
    shutil.copytree(SOURCE_LANDING_DIR, DEST_LANDING_DIR, dirs_exist_ok=True)


if __name__ == "__main__":

    version = os.getenv("BUILD_VERSION")
    if not version:
        print("Error: BUILD_VERSION environment variable is not set or is empty.")
        print("This is required to set AppVersion in the installer.")
        sys.exit(1)

    print(f"Version: {version}")
    clean()
    convert_icon()
    build()
    copy_static_files()
    print("Build complete! EXE located at:")
    print(f"   {os.path.abspath(os.path.join(BUILD_DIR, BUILD_NAME + '.exe'))}")
    
    write_inno_setup_file(version)
    run_inno_setup_compiler()
    print(f"Installer built and ready in: installer/inference-{version}-installer.exe")
