"""Collect indirectly loaded torchvision operators, including stable-ABI builds."""

from PyInstaller.utils.hooks import collect_dynamic_libs

# torchvision resolves these with FileFinder + torch.ops.load_library, rather
# than importing Python extension modules. The upstream _C hidden import misses
# _C_stable.pyd / image_stable.pyd in newer wheels. Collect binaries explicitly
# so PyInstaller also analyses their dependent DLLs, preserving package paths.
binaries = collect_dynamic_libs(
    "torchvision", search_patterns=["*.pyd", "*.dll", "*.so", "*.dylib"]
)
module_collection_mode = "pyz+py"
