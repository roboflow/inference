"""Load an optional SIMD resize implementation without replacing standard PIL."""

import hashlib
import importlib
import importlib.util
import os
import platform
import sys
import threading
from functools import lru_cache
from pathlib import Path

from packaging.version import Version

_LOAD_LOCK = threading.Lock()


def load_pillow_simd_image():
    """Return an isolated Image module, or raise ImportError with a fallback reason.

    Only Linux x86-64 SSE4.1 builds are supported. Import is lazy so CPU-only,
    ARM and reference-only installations never need this optional dependency.
    """
    if platform.system() != "Linux" or platform.machine().lower() not in (
        "x86_64",
        "amd64",
    ):
        raise ImportError("Pillow-SIMD requires Linux x86-64 with SSE4.1")
    try:
        flags = Path("/proc/cpuinfo").read_text().lower().split()
    except OSError as error:
        raise ImportError("Cannot verify Pillow-SIMD SSE4.1 support") from error
    if "sse4_1" not in flags:
        raise ImportError("Pillow-SIMD requires SSE4.1 CPU instructions")
    root = os.environ.get("INFERENCE_MODELS_PILLOW_SIMD_PATH", "/opt/pillow_simd")
    if not root:
        raise ImportError(
            "Pillow-SIMD disabled by empty INFERENCE_MODELS_PILLOW_SIMD_PATH"
        )
    with _LOAD_LOCK:
        return _load_image(str(Path(root).resolve()))


@lru_cache(maxsize=4)
def _load_image(root):
    package_dir = Path(root) / "PIL"
    if not (package_dir / "__init__.py").is_file():
        raise ImportError(f"No Pillow-SIMD build under {root}")
    alias = "PILSIMD_" + hashlib.sha256(root.encode()).hexdigest()[:16]
    try:
        spec = importlib.util.spec_from_file_location(
            alias,
            package_dir / "__init__.py",
            submodule_search_locations=[str(package_dir)],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[alias] = package
        spec.loader.exec_module(package)
        version = Version(package.__version__)
        if version < Version("12.3.0") or version.post is None:
            raise ImportError(f"Pillow-SIMD >=12.3.0.post0 required, found {version}")
        image = importlib.import_module(f"{alias}.Image")
        from PIL import Image

        if image.core is Image.core:
            raise ImportError("Pillow-SIMD must have its own extension module")
        return image
    except Exception as error:
        for name in list(sys.modules):
            if name == alias or name.startswith(alias + "."):
                del sys.modules[name]
        raise ImportError(f"Pillow-SIMD at {root} did not load: {error}") from error
