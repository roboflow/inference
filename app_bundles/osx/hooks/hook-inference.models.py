#  ── hooks/hook-inference.models.py ────────────────────────────────
#
#  Dynamically collect all sub‑packages that live directly under
#  `inference/models/`.  This means you can keep adding / removing
#  model back‑ends without ever touching your PyInstaller config again.
#
#  The hook is executed by PyInstaller *while* it is analysing your
#  source tree, so we can inspect the filesystem safely here.

import importlib.util
import pathlib
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

print("PYINSTALLER HOOK (OSX): Running hook-inference.models.py")

# ------------------------------------------------------------------
# Locate the physical directory of `inference.models`
# ------------------------------------------------------------------
print("PYINSTALLER HOOK (OSX): Attempting to find spec for inference.models")
spec = importlib.util.find_spec("inference.models")
if spec is None or not spec.submodule_search_locations:
    print("PYINSTALLER HOOK (OSX): inference.models package NOT FOUND or no submodule_search_locations.")
    hiddenimports = []
    datas = []
else:
    print(f"PYINSTALLER HOOK (OSX): inference.models package FOUND. Locations: {spec.submodule_search_locations}")
    models_dir = pathlib.Path(spec.submodule_search_locations[0])
    print(f"PYINSTALLER HOOK (OSX): models_dir set to: {models_dir}")

    # ----------------------------------------------------------------
    # Identify *direct* sub‑packages / modules that look like models
    # (folder with __init__.py  OR  single .py file)
    # ----------------------------------------------------------------
    model_names = []
    print(f"PYINSTALLER HOOK (OSX): Iterating models_dir: {models_dir}")
    for p in models_dir.iterdir():
        print(f"PYINSTALLER HOOK (OSX): Checking path: {p}")
        if p.name.startswith("_") or p.name.startswith("."):
            print(f"PYINSTALLER HOOK (OSX): Skipping private/hidden: {p.name}")
            continue
        if p.is_dir() and (p / "__init__.py").exists():
            model_names.append(f"inference.models.{p.name}")
            print(f"PYINSTALLER HOOK (OSX): Added directory model: {p.name}")
        elif p.suffix == ".py" and p.name != "__init__.py":
            model_names.append(f"inference.models.{p.stem}")
            print(f"PYINSTALLER HOOK (OSX): Added file model: {p.stem}")

    print(f"PYINSTALLER HOOK (OSX): Identified model_names: {model_names}")
    hiddenimports = []
    datas = []
    for mod in model_names:
        print(f"PYINSTALLER HOOK (OSX): Collecting submodules and data for: {mod}")
        hiddenimports += collect_submodules(mod)
        datas += collect_data_files(mod)
    print(f"PYINSTALLER HOOK (OSX): Final hiddenimports: {hiddenimports}")
    print(f"PYINSTALLER HOOK (OSX): Final datas: {datas}")

print("PYINSTALLER HOOK (OSX): hook-inference.models.py finished.")

# Expose the two variables that PyInstaller expects
#   • hiddenimports : list[str]
#   • datas         : list[tuple[str,str]]
