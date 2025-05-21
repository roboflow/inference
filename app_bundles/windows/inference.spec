# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules

rasterio_datas = collect_data_files('rasterio', include_py_files=False)

clip_datas, clip_binaries, clip_hiddenimports = collect_all('clip')
transformers_datas, transformers_bins, transformers_hiddenimports = collect_all('transformers')
peft_datas, peft_bins, peft_hiddenimports = collect_all('peft')
cython_datas, cython_bins, cython_hiddenimports = collect_all('Cython')
tldextract_datas, tldextract_binaries, tldextract_hidden = collect_all("tldextract")
inference_datas, inference_bins, inference_hidden = collect_all('inference') 

a = Analysis(
    ['run_inference.py'],
    pathex=[],
binaries=[
        *clip_binaries,
        *transformers_bins,
        *peft_bins,
        *cython_bins,
        *tldextract_binaries,
        *inference_bins
    ],
    datas=[
        *clip_datas, 
        *rasterio_datas, 
        *transformers_datas,
        *peft_datas,
        *cython_datas,
        *tldextract_datas,
        *inference_datas
    ],
    hiddenimports=[
        *clip_hiddenimports,
        *tldextract_hidden,
        *transformers_hiddenimports,
        *peft_hiddenimports,
        *cython_hiddenimports,
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
        'Cython'
        'inference'
        inference_hidden,
    ],
    hookspath=['hooks'],     # place custom hooks here if you like
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='inference',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app-icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='inference',
)
