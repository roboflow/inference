"""
Simple folder-based dataset for loading images, masks, and pointmaps.
"""

from dataclasses import dataclass
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union
import hashlib
import re
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from loguru import logger

from lidra.data.dataset.tdfy.trellis.dataset import PreProcessor

# Define FolderSampleID as a simple string alias for now
FolderSampleID = str


def frame_stride_filter(
    df: pd.DataFrame, stride: int = 30, offset: int = 0
) -> pd.DataFrame:
    """
    Filter dataframe to keep only every nth frame.

    Args:
        df: DataFrame with samples
        stride: Keep every nth frame (default: 30)
        offset: Starting offset (default: 0)

    Returns:
        Filtered DataFrame
    """
    import re

    # Extract frame numbers from the rgb_path
    def extract_frame_number(path):
        patterns = [
            r"frame[_\-]?(\d+)",  # frame_00000, frame00000
            r"annotated_frame[_\-]?(\d+)",  # annotated_frame_00000
            r"(\d+)\.(?:jpg|png|jpeg)",  # 00000.jpg
        ]

        for pattern in patterns:
            match = re.search(pattern, str(path), re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    # Add frame_number column if not present
    if "frame_number" not in df.columns:
        df = df.copy()
        df["frame_number"] = df["rgb_path"].apply(extract_frame_number)

    # Filter by stride
    filtered = df[df["frame_number"] % stride == offset]

    logger.info(
        f"Frame stride filter: keeping {len(filtered)} of {len(df)} frames (stride={stride}, offset={offset})"
    )

    return filtered


# Data structures


@dataclass
class Sample:
    """A single dataset sample"""

    rgb_path: Path
    mask_path: Optional[Path] = None
    pointmap_path: Optional[Path] = None

    @property
    def uuid(self) -> str:
        """Generate UUID from paths"""
        paths = f"{self.rgb_path}|{self.mask_path or ''}|{self.pointmap_path or ''}"
        return hashlib.md5(paths.encode()).hexdigest()

    @classmethod
    def from_row(cls, row) -> "Sample":
        """Create Sample from DataFrame row"""
        # Handle different column naming conventions
        rgb_col = "rgb_path" if "rgb_path" in row.index else "image_path"
        rgb_path = row[rgb_col] if rgb_col in row.index else None

        if not rgb_path:
            return None

        return cls(
            rgb_path=Path(rgb_path),
            mask_path=(
                Path(row["mask_path"])
                if pd.notna(row.get("mask_path", None)) and row.get("mask_path", "")
                else None
            ),
            pointmap_path=(
                Path(row["pointmap_path"])
                if pd.notna(row.get("pointmap_path", None))
                and row.get("pointmap_path", "")
                else None
            ),
        )


class FileMatcher:
    """Base matcher - direct path correspondence"""

    def match_files(
        self,
        rgb_dir: Path,
        mask_dir: Optional[Path] = None,
        pointmap_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Match RGB images with their masks and pointmaps, return as DataFrame"""
        rows = []

        # Find all RGB images
        rgb_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            rgb_files.extend(rgb_dir.rglob(ext))
        rgb_files = sorted(rgb_files)

        for rgb_path in rgb_files:
            mask_path = self.find_mask(rgb_path, mask_dir) if mask_dir else None
            pointmap_path = (
                self.find_pointmap(rgb_path, pointmap_dir) if pointmap_dir else None
            )

            # Generate UUID
            paths = f"{rgb_path}|{mask_path or ''}|{pointmap_path or ''}"
            uuid = hashlib.md5(paths.encode()).hexdigest()

            rows.append(
                {
                    "rgb_path": str(rgb_path),
                    "mask_path": str(mask_path) if mask_path else "",
                    "pointmap_path": str(pointmap_path) if pointmap_path else "",
                    "uuid": uuid,
                }
            )

        return pd.DataFrame(rows)

    def find_mask(self, rgb_path: Path, mask_dir: Path) -> Optional[Path]:
        """Override in subclasses for custom matching"""
        # Default: same name with .png extension
        mask_path = mask_dir / rgb_path.name.replace(".jpg", ".png").replace(
            ".jpeg", ".png"
        )
        return mask_path if mask_path.exists() else None

    def find_pointmap(self, rgb_path: Path, pointmap_dir: Path) -> Optional[Path]:
        """Override in subclasses for custom matching"""
        # Default: same name with pointmap extension
        base_name = rgb_path.stem
        for ext in [".npy", ".pt", ".pth", ".ply"]:
            pm_path = pointmap_dir / f"{base_name}{ext}"
            if pm_path.exists():
                return pm_path
        return None


class FrameNumberMatcher(FileMatcher):
    """Match files based on frame numbers (for sequences)"""

    def extract_frame_number(self, path: Path) -> Optional[int]:
        """Extract frame number from filename"""
        patterns = [
            r"frame[_\-]?(\d+)",  # frame_00000, frame00000
            r"annotated_frame[_\-]?(\d+)",  # annotated_frame_00000
            r"(\d+)\.(?:jpg|png|jpeg|ply|npy)",  # 00000.jpg
        ]

        for pattern in patterns:
            match = re.search(pattern, str(path.name), re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def find_mask(self, rgb_path: Path, mask_dir: Path) -> Optional[Path]:
        frame_num = self.extract_frame_number(rgb_path)
        if frame_num is None:
            return super().find_mask(rgb_path, mask_dir)

        # Check for frame_XXXXX/mask_*.png pattern
        frame_dir = mask_dir / f"frame_{frame_num:05d}"
        if frame_dir.exists():
            masks = list(frame_dir.glob("mask_*.png"))
            if masks:
                return masks[0]

        # Try direct match with same name
        mask_path = mask_dir / rgb_path.name.replace(".jpg", ".png").replace(
            ".jpeg", ".png"
        )
        if mask_path.exists():
            return mask_path

        # Fallback to parent implementation
        return super().find_mask(rgb_path, mask_dir)

    def find_pointmap(self, rgb_path: Path, pointmap_dir: Path) -> Optional[Path]:
        frame_num = self.extract_frame_number(rgb_path)
        if frame_num is None:
            return super().find_pointmap(rgb_path, pointmap_dir)

        # Try different naming patterns
        patterns = [
            # Direct file in pointmap_dir
            (f"frame_{frame_num:06d}.ply", ".ply"),
            (f"frame_{frame_num:06d}.npy", ".npy"),
            (f"frame_{frame_num:05d}.npy", ".npy"),
            # Subdirectory structure (frame_XXXXXX/frame_XXXXXX.ply)
            (f"frame_{frame_num:06d}/frame_{frame_num:06d}.ply", ".ply"),
            (f"frame_{frame_num:05d}/frame_{frame_num:05d}.ply", ".ply"),
        ]

        for pattern, ext in patterns:
            pm_path = pointmap_dir / pattern
            if pm_path.exists():
                return pm_path

        # Fallback to parent implementation
        return super().find_pointmap(rgb_path, pointmap_dir)


class MultiInstanceMaskMatcher(FileMatcher):
    """Match files where masks are organized in subdirectories by image name.
    Creates separate samples for each mask instance."""

    def match_files(
        self,
        rgb_dir: Path,
        mask_dir: Optional[Path] = None,
        pointmap_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Match RGB images with masks, creating one row per mask instance"""
        rows = []

        # No mask directory - return empty DataFrame
        if not mask_dir:
            return pd.DataFrame(rows)

        # Find all RGB images
        rgb_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            rgb_files.extend(rgb_dir.rglob(ext))
        rgb_files = sorted(rgb_files)

        for rgb_path in rgb_files:
            # Check for mask subdirectory
            mask_subdir = mask_dir / rgb_path.stem
            if not mask_subdir.exists() or not mask_subdir.is_dir():
                continue  # Skip images without mask subdirectory

            # Get all masks in subdirectory
            masks = []
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                masks.extend(mask_subdir.glob(ext))
            masks = sorted(masks)
            if not masks:
                continue  # Skip images with empty mask subdirectory

            # Create one sample per mask
            for mask_path in masks:
                pointmap_path = (
                    self.find_pointmap(rgb_path, pointmap_dir) if pointmap_dir else None
                )
                paths = f"{rgb_path}|{mask_path}|{pointmap_path or ''}"
                uuid = hashlib.md5(paths.encode()).hexdigest()

                # Extract mask index from filename
                mask_index = mask_path.stem.split("_")[-1]

                rows.append(
                    {
                        "rgb_path": str(rgb_path),
                        "mask_path": str(mask_path),
                        "pointmap_path": str(pointmap_path) if pointmap_path else "",
                        "uuid": uuid,
                        "mask_index": mask_index,
                        "image_stem": rgb_path.stem,
                    }
                )

        df = pd.DataFrame(rows)

        # Log statistics
        if len(df) > 0:
            unique_images = (
                df["image_stem"].nunique() if "image_stem" in df.columns else 0
            )
            total_masks = len(df)
            avg_masks = total_masks / unique_images if unique_images > 0 else 0
            logger.info(
                f"Found {unique_images} images with {total_masks} masks (avg {avg_masks:.1f} masks/image)"
            )

        return df


class FileLoader:
    """Load files from disk"""

    def load_image(self, path: Optional[Path]) -> Optional[Image.Image]:
        return Image.open(path).convert("RGB")

    def load_mask(self, path: Optional[Path]) -> Optional[Image.Image]:
        if path is None:
            return None
        img = Image.open(path)
        # If image has alpha channel, extract it
        if img.mode in ("RGBA", "LA"):
            return img.getchannel("A")
        # For 3-channel images (RGB/JPG), convert to grayscale and handle binary masks
        elif img.mode == "RGB":
            # Convert to grayscale first
            gray_img = img.convert("L")
            # Convert to numpy for binary mask processing
            import numpy as np

            mask_array = np.array(gray_img)
            # Handle binary masks where values are either 0 or 255
            mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
            return Image.fromarray(mask_array, mode="L")
        # Otherwise convert to grayscale
        return img.convert("L")


class FolderDataset(Dataset):
    """Simple folder-based dataset with CSV metadata support"""

    def __init__(
        self,
        image_dir: str,
        pointmap_loader: Optional[Callable] = None,
        mask_dir: Optional[str] = None,
        pointmap_dir: Optional[str] = None,
        preprocessor: Optional[PreProcessor] = None,
        file_matcher: Optional[FileMatcher] = None,
        metadata_csv: Optional[str] = None,
        metadata_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        return_pointmap: bool = True,
        auto_match: bool = True,
        save_metadata_csv: Optional[str] = None,
        csv_base_dir: Optional[str] = None,
    ):
        """
        Args:
            image_dir: Directory containing RGB images
            pointmap_loader: Callable for loading pointmaps (optional)
            mask_dir: Directory containing masks (optional)
            pointmap_dir: Directory containing pointmaps (optional)
            preprocessor: PreProcessor for transforms
            file_matcher: Strategy for matching files across directories
            metadata_csv: Path to CSV file with pre-computed matches
            metadata_filter: Optional function to filter metadata DataFrame
            return_pointmap: Whether to include pointmap in output
            auto_match: If True, try FrameNumberMatcher if FileMatcher finds nothing
            save_metadata_csv: If provided, save generated metadata to this CSV path
            csv_base_dir: Base directory for resolving relative paths in CSV (defaults to parent of metadata_csv)
        """
        # Store paths
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.pointmap_dir = Path(pointmap_dir) if pointmap_dir else None
        self.csv_base_dir = Path(csv_base_dir) if csv_base_dir else None

        # Use provided or default components
        self.matcher = file_matcher or FileMatcher()
        self.loader = FileLoader()
        self.preprocessor = preprocessor or PreProcessor()
        self.pointmap_loader = pointmap_loader
        self.return_pointmap = return_pointmap

        # Build or load metadata
        self.metadata = self.get_metadata(
            metadata_csv=metadata_csv,
            metadata_filter=metadata_filter,
            auto_match=auto_match,
            save_metadata_csv=save_metadata_csv,
        )

        if len(self.metadata) == 0:
            raise ValueError(f"No samples found in {image_dir}")

        logger.info(f"Dataset ready with {len(self.metadata)} samples")

    def _resolve_paths_in_dataframe(
        self, df: pd.DataFrame, metadata_csv: str
    ) -> pd.DataFrame:
        """Resolve relative paths in DataFrame to absolute paths."""
        base_dir = self.csv_base_dir if self.csv_base_dir else Path(metadata_csv).parent

        for col in ["rgb_path", "mask_path", "pointmap_path"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda p: (
                        str(base_dir / p)
                        if pd.notna(p) and p and not Path(p).is_absolute()
                        else p
                    )
                )

        return df

    def get_metadata(
        self,
        metadata_csv: Optional[str] = None,
        metadata_filter: Optional[Callable] = None,
        auto_match: bool = True,
        save_metadata_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get metadata DataFrame"""
        if not metadata_csv or not Path(metadata_csv).exists():
            df = self._build_metadata(metadata_filter, auto_match, save_metadata_csv)
        else:
            df = pd.read_csv(metadata_csv)
            logger.info(f"Loaded {len(df)} samples from {metadata_csv}")

            # Resolve relative paths if csv_base_dir is set or infer from metadata_csv location
            df = self._resolve_paths_in_dataframe(df, metadata_csv)

        if metadata_filter:
            df = metadata_filter(df)
        logger.info(f"Loaded {len(df)} samples from {metadata_csv}")
        return df

    def _build_metadata(
        self,
        metadata_filter: Optional[Callable] = None,
        auto_match: bool = True,
        save_metadata_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """Build or load metadata DataFrame"""
        # Otherwise, build metadata by matching files
        df = self.matcher.match_files(self.image_dir, self.mask_dir, self.pointmap_dir)

        # If no samples found and auto_match is True, try FrameNumberMatcher
        if (
            len(df) == 0
            and auto_match
            and not isinstance(self.matcher, FrameNumberMatcher)
        ):
            logger.info(
                "No samples found with default matcher, trying FrameNumberMatcher"
            )
            frame_matcher = FrameNumberMatcher()
            df = frame_matcher.match_files(
                self.image_dir, self.mask_dir, self.pointmap_dir
            )

        # Apply filter if provided
        if metadata_filter and len(df) > 0:
            df = metadata_filter(df)

        # Save metadata if requested
        if save_metadata_csv and len(df) > 0:
            df.to_csv(save_metadata_csv, index=False)
            logger.info(f"Saved metadata to {save_metadata_csv} with {len(df)} samples")

        return df

    def _load_pointmap(self, sample: Sample, row: pd.Series) -> Optional[torch.Tensor]:
        """Load pointmap using the configured loader"""
        if self.pointmap_loader is None:
            return None

        return self.pointmap_loader(
            base_name=str(sample.rgb_path.stem),
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: Union[int, str]) -> Tuple[FolderSampleID, Dict]:
        if isinstance(idx, int):
            row = self.metadata.iloc[idx]
        elif isinstance(idx, str):
            row = self.metadata[self.metadata["uuid"] == idx]
            if len(row) == 0:
                raise KeyError(f"No sample found with uuid: {idx}")
            row = row.iloc[0]
        else:
            raise TypeError(f"Index must be int or str, got {type(idx)}")
        return self._compute_item(row, idx)

    def _compute_item(self, row: pd.Series, idx: int) -> Tuple[FolderSampleID, Dict]:
        sample = Sample.from_row(row)

        # Load files
        rgb = self.loader.load_image(sample.rgb_path)
        mask = self.loader.load_mask(sample.mask_path)

        # Load pointmap using helper method
        pointmap = self._load_pointmap(sample, row)

        # Process
        data = self.preprocessor._process_image_mask_pointmap_mess(rgb, mask, pointmap)
        data["metadata"] = row.to_dict()
        # Use UUID from row (preserves original UUID even after path resolution)
        uuid = row.get("uuid", sample.uuid)
        return uuid, data
