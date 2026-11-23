"""Collect indirectly loaded torchvision operators, including stable-ABI builds."""

from PyInstaller.utils.hooks import collect_dynamic_libs

# torchvision resolves these with FileFinder + torch.ops.load_library, rather
# than importing Python extension modules. The upstream _C hidden import misses
# _C_stable.so / image_stable.so in newer wheels, and the default patterns only
# match lib*.so. Collect them and the .dylibs/ they link against explicitly,
# preserving package paths. app_bundles/windows/hooks has the same hook.
binaries = collect_dynamic_libs("torchvision", search_patterns=["*.so", "*.dylib"])
# This hook replaces the upstream one, so keep its .py collection for TorchScript.
module_collection_mode = "pyz+py"
