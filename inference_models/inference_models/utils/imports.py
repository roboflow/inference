import importlib
import os
import sys
from typing import Any, Optional


class LazyClass:

    def __init__(self, module_name: str, class_name: str):
        self._module_name = module_name
        self._class_name = class_name
        self._symbol: Optional[type] = None

    def resolve(self) -> type:
        if self._symbol is None:
            module = importlib.import_module(self._module_name)
            self._symbol = getattr(module, self._class_name)
        return self._symbol


def import_class_from_file(
    file_path: str, class_name: str, alias_name: Optional[str] = None
) -> type:
    """
    Emulates what huggingface transformers does to load remote code with trust_remote_code=True,
    but allows us to use the class directly so that we don't have to load untrusted code.
    """
    file_path = os.path.abspath(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    module_dir = os.path.dirname(file_path)
    parent_dir = os.path.dirname(module_dir)

    sys.path.insert(0, parent_dir)

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)

        # Manually set the __package__ attribute to the parent package
        module.__package__ = os.path.basename(module_dir)

        spec.loader.exec_module(module)
        cls = getattr(module, class_name)
        if alias_name:
            globals()[alias_name] = cls
        return cls
    finally:
        sys.path.pop(0)
