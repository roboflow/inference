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

print("PYINSTALLER HOOK: Running hook-inference.models.py")

# ------------------------------------------------------------------
# Locate the physical directory of `inference.models`
# ------------------------------------------------------------------
print("PYINSTALLER HOOK: Attempting to find spec for inference.models")
spec = importlib.util.find_spec("inference.models")
if spec is None or not spec.submodule_search_locations:
    print("PYINSTALLER HOOK: inference.models package NOT FOUND or no submodule_search_locations.")
    # Package not found; leave hook empty to avoid build crash
    hiddenimports = []
    datas = []
else:
    print(f"PYINSTALLER HOOK: inference.models package FOUND. Locations: {spec.submodule_search_locations}")
    models_dir = pathlib.Path(spec.submodule_search_locations[0])
    print(f"PYINSTALLER HOOK: models_dir set to: {models_dir}")

    # ----------------------------------------------------------------
    # Identify *direct* sub‑packages / modules that look like models
    # (folder with __init__.py  OR  single .py file)
    # ----------------------------------------------------------------
    model_names = []
    print(f"PYINSTALLER HOOK: Iterating models_dir: {models_dir}")
    for p in models_dir.iterdir():
        print(f"PYINSTALLER HOOK: Checking path: {p}")
        if p.name.startswith("_") or p.name.startswith("."):
            print(f"PYINSTALLER HOOK: Skipping private/hidden: {p.name}")
            continue  # skip private/hidden folders
        if p.is_dir() and (p / "__init__.py").exists():
            model_names.append(f"inference.models.{p.name}")
            print(f"PYINSTALLER HOOK: Added directory model: {p.name}")
        elif p.suffix == ".py" and p.name != "__init__.py": # Ensure __init__.py itself is not treated as a model module
            model_names.append(f"inference.models.{p.stem}")
            print(f"PYINSTALLER HOOK: Added file model: {p.stem}")

    print(f"PYINSTALLER HOOK: Identified model_names: {model_names}")
    # Use PyInstaller helpers to pull everything those sub‑packages need
    hiddenimports = []
    datas = []
    for mod in model_names:
        print(f"PYINSTALLER HOOK: Collecting submodules and data for: {mod}")
        hiddenimports += collect_submodules(mod)
        datas += collect_data_files(mod)
    print(f"PYINSTALLER HOOK: Final hiddenimports: {hiddenimports}")
    print(f"PYINSTALLER HOOK: Final datas: {datas}")

# Expose the two variables that PyInstaller expects
#   • hiddenimports : list[str]
#   • datas         : list[tuple[str,str]]
print("PYINSTALLER HOOK: hook-inference.models.py finished.")
