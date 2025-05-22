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

# ------------------------------------------------------------------
# Locate the physical directory of `inference.models`
# ------------------------------------------------------------------
spec = importlib.util.find_spec("inference.models")
if spec is None or not spec.submodule_search_locations:
    # Package not found; leave hook empty to avoid build crash
    hiddenimports = []
    datas = []
else:
    models_dir = pathlib.Path(spec.submodule_search_locations[0])

    # ----------------------------------------------------------------
    # Identify *direct* sub‑packages / modules that look like models
    # (folder with __init__.py  OR  single .py file)
    # ----------------------------------------------------------------
    model_names = []
    for p in models_dir.iterdir():
        if p.name.startswith("_") or p.name.startswith("."):
            continue  # skip private/hidden folders
        if p.is_dir() and (p / "__init__.py").exists():
            model_names.append(f"inference.models.{p.name}")
        elif p.suffix == ".py":
            model_names.append(f"inference.models.{p.stem}")

    # Use PyInstaller helpers to pull everything those sub‑packages need
    hiddenimports = []
    datas = []
    for mod in model_names:
        hiddenimports += collect_submodules(mod)
        datas += collect_data_files(mod)

# Expose the two variables that PyInstaller expects
#   • hiddenimports : list[str]
#   • datas         : list[tuple[str,str]]
