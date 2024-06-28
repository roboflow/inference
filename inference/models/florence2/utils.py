import importlib
import os
import sys


def import_class_from_file(file_path, class_name):
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
        return getattr(module, class_name)
    finally:
        sys.path.pop(0)
