"""Build-time check that isolated Pillow-SIMD resizes without replacing Pillow.

Standalone on purpose: CPU images may install inference-models from PyPI.
This is an isolation and native-load smoke check, not a package provenance audit.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import PIL
import PIL.Image
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SIMD_ALIAS = "PILSIMD_BUILD_CHECK"


def _require(condition, message):
    # Unlike assert, build checks must also execute under python -O.
    if not condition:
        raise RuntimeError(message)


def _read_requirement(path, *, name):
    matches = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue

        requirement = Requirement(line)
        if canonicalize_name(requirement.name) == canonicalize_name(name):
            matches.append(requirement)

    _require(len(matches) == 1, f"Expected one {name} requirement in {path}")
    return matches[0]


def _load_simd(package_dir):
    spec = importlib.util.spec_from_file_location(
        SIMD_ALIAS,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    _require(
        spec is not None and spec.loader is not None, "Cannot load Pillow-SIMD package"
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[SIMD_ALIAS] = package
    spec.loader.exec_module(package)
    image = importlib.import_module(f"{SIMD_ALIAS}.Image")
    return package, image


def _verify_resize(*, standard_image, simd_image):
    """Smoke-test one bilinear resize against standard Pillow."""
    source = np.random.default_rng(0).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    expected = np.asarray(
        standard_image.fromarray(source).resize(
            (32, 32), standard_image.Resampling.BILINEAR
        )
    )
    actual = np.asarray(
        simd_image.fromarray(source).resize((32, 32), simd_image.Resampling.BILINEAR)
    )
    _require(
        actual.shape == expected.shape and actual.dtype == expected.dtype,
        "Pillow-SIMD resize shape/dtype mismatch",
    )
    max_diff = int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max())
    _require(max_diff <= 1, f"Pillow-SIMD resize differs from Pillow by {max_diff}")


def _verify(root):
    root = root.resolve()
    requirement = _read_requirement(
        REPOSITORY_ROOT / "requirements/_requirements.txt", name="Pillow"
    )
    _require(
        Version(PIL.__version__) in requirement.specifier,
        f"Standard Pillow {PIL.__version__} does not satisfy {requirement}",
    )
    standard_image, standard_core = PIL.Image, PIL.Image.core
    for module in (PIL, standard_image, standard_core):
        _require(
            not Path(module.__file__).resolve().is_relative_to(root),
            f"Standard Pillow loads from the SIMD installation: {module.__file__}",
        )

    original_modules = {
        name: sys.modules[name] for name in ("PIL", "PIL.Image", "PIL._imaging")
    }

    def _check_standard_unchanged():
        _require(
            all(
                sys.modules.get(name) is module
                for name, module in original_modules.items()
            )
            and PIL.Image is standard_image
            and standard_image.core is standard_core,
            "Pillow-SIMD replaced standard Pillow modules",
        )

    _require(
        SIMD_ALIAS not in sys.modules, f"Module alias already in use: {SIMD_ALIAS}"
    )
    try:
        package, image = _load_simd(root / "PIL")
        for module in (package, image, image.core):
            _require(
                Path(module.__file__).resolve().is_relative_to(root / "PIL"),
                f"Pillow-SIMD module loads outside its installation: {module.__file__}",
            )
        _require(
            image.core is not standard_core
            and not Path(image.core.__file__).samefile(standard_core.__file__),
            "Pillow-SIMD shares standard Pillow's extension",
        )
        _check_standard_unchanged()

        _verify_resize(standard_image=standard_image, simd_image=image)
        _check_standard_unchanged()
    finally:
        for name in list(sys.modules):
            if name == SIMD_ALIAS or name.startswith(SIMD_ALIAS + "."):
                del sys.modules[name]

    print(
        f"Pillow {PIL.__version__} + Pillow-SIMD {package.__version__} at {root}: isolation and resize smoke check ok"
    )


if __name__ == "__main__":
    try:
        _verify(
            Path(
                os.environ.get("INFERENCE_MODELS_PILLOW_SIMD_PATH", "/opt/pillow_simd")
            )
        )
    except Exception as error:
        raise SystemExit(f"Pillow-SIMD verification failed: {error}") from error
