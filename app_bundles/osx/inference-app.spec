# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

# ---------------------------------------------------------------------------
# Wheel collections you already had
rasterio_datas = collect_data_files('rasterio', include_py_files=True)

clip_datas, clip_binaries, clip_hiddenimports                 = collect_all('clip')
transformers_datas, transformers_bins, transformers_hiddenimports = collect_all('transformers', include_py_files=True)
peft_datas, peft_bins, peft_hiddenimports                     = collect_all('peft')
cython_datas, cython_bins, cython_hiddenimports               = collect_all('Cython')
tldextract_datas, tldextract_binaries, tldextract_hidden      = collect_all("tldextract")
inference_datas, inference_bins, inference_hidden = collect_all('inference', include_py_files=True)
scipy_datas, scipy_binaries, scipy_hiddenimports              = collect_all('scipy')


# ---------------------------------------------------------------------------
opensslnames = ("libcrypto.3.dylib", "libssl.3.dylib")
BREW_SSL     = "/opt/homebrew/opt/openssl@3/lib"     # change if you vend your own

# ---------------------------------------------------------------------------
a = Analysis(
    ['run_inference_gui.py'],
    pathex=[],
    binaries=[
        *clip_binaries,
        *transformers_bins,
        *peft_bins,
        *cython_bins,
        *tldextract_binaries,
        *inference_bins,
        *scipy_binaries,
    ],
    datas=[
        *clip_datas,
        *rasterio_datas,
        *transformers_datas,
        *peft_datas,
        *cython_datas,
        *tldextract_datas,
        *inference_datas,
        *scipy_datas,
        # Manually include editor.html for the builder interface
        ('../../inference/core/interfaces/http/builder/editor.html', 'inference/core/interfaces/http/builder'),
    ],
    hiddenimports=[
        *clip_hiddenimports,
        *tldextract_hidden,
        *transformers_hiddenimports,
        *peft_hiddenimports,
        *cython_hiddenimports,
        *scipy_hiddenimports,
        'psutil',
        'rasterio',
        'rasterio.sample',
        'rasterio.vrt',
        'rasterio.features',
        'tldextract',
        'transformers',
        'transformers.models',
        'transformers.models.auto',
        'transformers.models.__init__',
        'peft',
        'Cython',
        'inference',
        'pyvips',
        'scipy',
        'scipy.linalg.cython_lapack',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_overflow',
        'scipy._lib.messagestream',
        *inference_hidden,
    ],
    hookspath=['hooks'],     # place custom hooks here if you like
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    optimize=0,
)

# ---------------------------------------------------------------------------
# 🔐  ONE‑VERSION OPENSSL POLICY
# 1) strip every libcrypto/libssl from BOTH binaries *and* datas
a.binaries = [b for b in a.binaries if os.path.basename(b[0]) not in opensslnames]
a.datas    = [d for d in a.datas    if os.path.basename(d[0]) not in opensslnames]

# 2) inject the canonical 3.4.0 dylibs
a.binaries += [
    ("libcrypto.3.dylib", os.path.join(BREW_SSL, "libcrypto.3.dylib"), "BINARY"),
    ("libssl.3.dylib",    os.path.join(BREW_SSL, "libssl.3.dylib"),    "BINARY"),
]

# ---------------------------------------------------------------------------
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='inference-app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app-icon.icns'],
)

coll = COLLECT(
    exe,
    a.binaries,      # now contains only one OpenSSL pair
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='inference-app',
)

app = BUNDLE(
    coll,
    name='Roboflow Inference.app',
    icon='app-icon.icns',
    bundle_identifier='com.roboflow.inference',
    info_plist={
        'CFBundleShortVersionString': '0.0.0',
        'CFBundleVersion': '0.0.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13',
    },
)
