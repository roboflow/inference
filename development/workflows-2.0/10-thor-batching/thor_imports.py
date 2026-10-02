"""The one place 10 reaches the 08 and 09 code it reuses.

Importing this module appends both directories to ``sys.path``::

    import thor_imports                       # first, before any 08/09 module
    import backends, gpu_blocks, detection_blocks, run_benchmark   # plain names
    thor_run_batched = thor_imports.load_09_module("run_batched")  # shadowed names

10 has files named like 09 ones (``run_batched``, ``check_parity``). A script's
own directory comes first on ``sys.path``, so a plain import of such a name
finds the 10 file; ``load_09_module`` imports the 09 file under another name.
Every other 08/09 module is imported by its plain name only, so each file is
one module object per process (one set of block classes, one configured mode).
Nothing heavy is imported here, so ``backends.configure_mode`` can still run
before any Workflows import.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

WORKFLOWS_2_DIR = Path(__file__).resolve().parent.parent
LIVE_DETECTION_DIR = WORKFLOWS_2_DIR / "08-live-detection"
THOR_MULTISTREAM_DIR = WORKFLOWS_2_DIR / "09-thor-multistream"
SHADOWED_09_MODULES = ("run_batched", "check_parity")


def install_paths() -> None:
    """Append the 08 and 09 directories to ``sys.path`` once; runs on import."""
    for directory in (LIVE_DETECTION_DIR, THOR_MULTISTREAM_DIR):
        if str(directory) not in sys.path:
            sys.path.append(str(directory))


def load_09_module(name: str) -> ModuleType:
    """Import ``09-thor-multistream/<name>.py`` as module ``thor_<name>``.

    Args:
        name: One of ``SHADOWED_09_MODULES``.

    Returns:
        The module, imported once per process.

    Raises:
        ValueError: For any other name; import it by its plain name, or the
            process would hold two copies of that file.
    """
    if name not in SHADOWED_09_MODULES:
        raise ValueError(
            f"import 09 module {name!r} by its plain name; load_09_module is for "
            f"{SHADOWED_09_MODULES} only"
        )

    qualified = f"thor_{name}"
    module = sys.modules.get(qualified)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            qualified, THOR_MULTISTREAM_DIR / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)

    return module


install_paths()
