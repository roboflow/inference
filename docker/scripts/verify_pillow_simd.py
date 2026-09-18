"""Build-time check: standard Pillow is intact and Pillow-SIMD loads beside it.

Standalone on purpose: the CPU image installs `inference-models` from PyPI, so
the loader inside the package is not available at every build.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import PIL
import PIL.Image

ROOT = Path(os.environ.get("INFERENCE_MODELS_PILLOW_SIMD_PATH", "/opt/pillow_simd"))
PACKAGE_DIR = ROOT / "PIL"

major, minor = (int(part) for part in PIL.__version__.split(".")[:2])
assert (major, minor) >= (12, 3), f"standard Pillow replaced: {PIL.__version__}"

spec = importlib.util.spec_from_file_location(
    "PILSIMD",
    PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(PACKAGE_DIR)],
)
module = importlib.util.module_from_spec(spec)
sys.modules["PILSIMD"] = module
spec.loader.exec_module(module)
simd_image = importlib.import_module("PILSIMD.Image")
assert "post" in module.__version__, f"not a Pillow-SIMD build: {module.__version__}"
simd_major, simd_minor = (int(part) for part in module.__version__.split(".")[:2])
assert (simd_major, simd_minor) >= (
    12,
    3,
), f"outdated Pillow-SIMD: {module.__version__}"
assert simd_image.core is not PIL.Image.core, "Pillow-SIMD shares the wheel's extension"

source = np.random.default_rng(0).integers(0, 256, (96, 128, 3), dtype=np.uint8)
simd_out = np.asarray(
    simd_image.fromarray(source).resize((32, 32), simd_image.BILINEAR)
)
std_out = np.asarray(PIL.Image.fromarray(source).resize((32, 32), PIL.Image.BILINEAR))
max_diff = int(np.abs(simd_out.astype(np.int16) - std_out.astype(np.int16)).max())
assert max_diff <= 1, f"Pillow-SIMD resize differs from Pillow by {max_diff}"
print(f"Pillow {PIL.__version__} + Pillow-SIMD {module.__version__} at {ROOT}: ok")
