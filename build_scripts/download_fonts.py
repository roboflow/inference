"""Compatibility entry point for the standalone project's font downloader."""

import importlib.util
import sys
from pathlib import Path

_canonical_path = (
    Path(__file__).resolve().parent.parent
    / "workflows"
    / "build_scripts"
    / "download_fonts.py"
)
_spec = importlib.util.spec_from_file_location(__name__, _canonical_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
if __name__ == "__main__":
    sys.exit(_module.main())
else:
    # Share constants and private helper monkeypatches as well as functions.
    sys.modules[__name__] = _module
