"""`Image` from a Pillow-SIMD build, loaded beside standard Pillow under its own package name.

Pillow-SIMD tracks an older Pillow release than the wheel the images ship, so
it must not replace `PIL`. The Docker images install it under
`/opt/pillow_simd`; importing this module loads that package as `PILSIMD` and
exposes only its `Image` module. The import raises `ModuleNotFoundError` when
no build is under the path and `ImportError` when the build does not load, so
a caller falls back with `except ImportError`.
"""

import importlib
import importlib.util
import os
import sys

from inference_models.logger import LOGGER

PILLOW_SIMD_PATH_ENV = "INFERENCE_MODELS_PILLOW_SIMD_PATH"
DEFAULT_PILLOW_SIMD_PATH = "/opt/pillow_simd"
PILLOW_SIMD_ALIAS = "PILSIMD"

_root = os.environ.get(PILLOW_SIMD_PATH_ENV, DEFAULT_PILLOW_SIMD_PATH)
_package_dir = os.path.join(_root, "PIL") if _root else ""
_init_path = os.path.join(_package_dir, "__init__.py")
if not _package_dir or not os.path.isfile(_init_path):
    raise ModuleNotFoundError(f"No Pillow-SIMD build under {_root or '<empty path>'}")

try:
    _spec = importlib.util.spec_from_file_location(
        PILLOW_SIMD_ALIAS, _init_path, submodule_search_locations=[_package_dir]
    )
    _package = importlib.util.module_from_spec(_spec)
    sys.modules[PILLOW_SIMD_ALIAS] = _package
    _spec.loader.exec_module(_package)
    Image = importlib.import_module(f"{PILLOW_SIMD_ALIAS}.Image")
except Exception as error:
    for name in [m for m in sys.modules if m.split(".")[0] == PILLOW_SIMD_ALIAS]:
        del sys.modules[name]
    LOGGER.warning("Pillow-SIMD at %s did not load: %s", _root, error)
    raise ImportError(f"Pillow-SIMD at {_root} did not load") from error

LOGGER.info("Pillow-SIMD %s loaded from %s", _package.__version__, _root)
