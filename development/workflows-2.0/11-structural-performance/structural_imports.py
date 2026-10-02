"""The one place 11 reaches the 08, 09 and 10 code it reuses.

Importing this module appends the 10 directory to ``sys.path`` and imports
10's ``thor_imports``, which appends 08 and 09::

    import structural_imports                  # first, before any 08/09/10 module
    import batched_backend, gpu_blocks, diagnose        # plain names
    parity_10 = structural_imports.load_10_module("check_parity")  # shadowed names

11 has a ``check_parity.py`` like 09 and 10. A script's own directory comes
first on ``sys.path``, so a plain import of that name finds the 11 file;
``load_10_module`` imports the 10 file under another name. Nothing heavy is
imported here, so 09 ``backends.configure_mode`` can still run first.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

WORKFLOWS_2_DIR = Path(__file__).resolve().parent.parent
THOR_BATCHING_DIR = WORKFLOWS_2_DIR / "10-thor-batching"
SHADOWED_10_MODULES = ("check_parity", "run_batched")

if str(THOR_BATCHING_DIR) not in sys.path:
    sys.path.append(str(THOR_BATCHING_DIR))

import thor_imports  # noqa: E402,F401 - appends the 08 and 09 directories


def load_10_module(name: str) -> ModuleType:
    """Import ``10-thor-batching/<name>.py`` as module ``thor_batching_<name>``.

    Args:
        name: One of ``SHADOWED_10_MODULES``.

    Returns:
        The module, imported once per process.

    Raises:
        ValueError: For any other name; import it by its plain name, or the
            process would hold two copies of that file.
    """
    if name not in SHADOWED_10_MODULES:
        raise ValueError(
            f"import 10 module {name!r} by its plain name; load_10_module is for "
            f"{SHADOWED_10_MODULES} only"
        )

    qualified = f"thor_batching_{name}"
    module = sys.modules.get(qualified)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            qualified, THOR_BATCHING_DIR / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)

    return module
