"""
Custom file matchers for specific dataset naming patterns.
"""

from pathlib import Path
from typing import Optional
from .dataset import FileMatcher


class HeavyOcclusionMatcher(FileMatcher):
    """Matcher for heavy_occlusion dataset where:
    - RGB: fullcontext_metaclip_XXXXXXXX_X.png
    - Mask: mask_metaclip_XXXXXXXX_X.png

    This matcher handles the specific prefix replacement pattern
    where RGB images have 'fullcontext_' prefix and masks have 'mask_' prefix.
    """

    def find_mask(self, rgb_path: Path, mask_dir: Path) -> Optional[Path]:
        """Find corresponding mask by replacing 'fullcontext_' with 'mask_'"""
        # Replace the prefix in the filename
        mask_name = rgb_path.name.replace("fullcontext_", "mask_")
        mask_path = mask_dir / mask_name
        return mask_path if mask_path.exists() else None
