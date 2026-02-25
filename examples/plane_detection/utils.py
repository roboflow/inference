from pathlib import Path

import click
import numpy as np
import piexif
from pillow_heif import open_heif

# Full-frame 35mm film reference dimensions (mm)
FULL_FRAME_SENSOR_WIDTH_MM = 36.0
FULL_FRAME_SENSOR_DIAGONAL_MM = 43.3
TYPICAL_SMARTPHONE_SENSOR_DIAGONAL_MM = 7.0


def focal_length_from_35mm_equivalent(f35_mm: float, image_width_px: float) -> float:
    """Convert 35mm-equivalent focal length to pixel focal length.

    Full-frame width is 36mm (reference), so f_px ≈ (f35/36) * image_width_px.
    """
    return (f35_mm / FULL_FRAME_SENSOR_WIDTH_MM) * image_width_px


def focal_length_from_physical(f_mm, image_width_px: float, *, from_exif: bool = True) -> float:
    """Estimate pixel focal length from physical focal length.

    Uses heuristic: assume typical smartphone sensor diagonal ≈ 7.0mm
    => crop ≈ 43.3 / 7.0. Less reliable than 35mm equivalent.

    Args:
        f_mm: Focal length in mm. When from_exif=True (default), accepts EXIF rational (num, den).
        image_width_px: Image width in pixels.
        from_exif: If True, convert EXIF rational to float. If False, f_mm must already be float.
    """

    def _rat_to_float(x) -> float:
        # piexif encodes rationals as (num, den)
        if isinstance(x, tuple) and len(x) == 2 and x[1] != 0:
            return float(x[0]) / float(x[1])
        return float(x)

    if from_exif:
        f_mm = _rat_to_float(f_mm)
    crop = FULL_FRAME_SENSOR_DIAGONAL_MM / TYPICAL_SMARTPHONE_SENSOR_DIAGONAL_MM
    f35_est = f_mm * crop
    return focal_length_from_35mm_equivalent(f35_est, image_width_px)


def get_camera_intrinsics_from_exif_in_heic_image(image_path: str) -> tuple[float, float, float, float]:
    """Return approximate pinhole intrinsics (fx, fy, cx, cy) for a HEIC image.

    Notes:
    - Uses EXIF `FocalLengthIn35mmFilm` (recommended) to estimate focal length in pixels.
    - Assumes square pixels => fx == fy.
    - Uses EXIF `SubjectArea` (if present) as a hint for principal point; otherwise uses image center.

    Returns:
        (fx, fy, cx, cy) in pixel units.
    """

    heif = open_heif(image_path)

    exif_bytes = heif.info.get("exif")
    if not exif_bytes:
        raise ValueError("No EXIF data found in the image")

    exif_dict = piexif.load(exif_bytes)
    exif = exif_dict.get("Exif")
    if not exif:
        raise ValueError("No EXIF sub-dictionary found in the image")

    # Prefer EXIF pixel dimensions; fall back to container dimensions.
    W = exif.get(piexif.ExifIFD.PixelXDimension)
    H = exif.get(piexif.ExifIFD.PixelYDimension)

    if W is None or H is None:
        # pillow-heif exposes size on the object
        try:
            W, H = heif.size
        except Exception as e:
            raise ValueError("Could not determine image dimensions from EXIF or HEIF container") from e

    W = int(W)
    H = int(H)

    # Prefer 35mm equivalent focal length if available.
    f35 = exif.get(piexif.ExifIFD.FocalLengthIn35mmFilm)

    if f35 is not None:
        f_px = focal_length_from_35mm_equivalent(float(f35), float(W))
    else:
        f_mm = exif.get(piexif.ExifIFD.FocalLength)
        if f_mm is None:
            raise ValueError(
                "Neither `FocalLengthIn35mmFilm` nor `FocalLength` found in EXIF. Cannot estimate intrinsics."
            )
        f_px = focal_length_from_physical(f_mm, float(W))

    fx = f_px
    fy = f_px  # assume square pixels

    # Principal point: SubjectArea is often (x, y, w, h) where (x, y) is the focus point.
    subject_area = exif.get(piexif.ExifIFD.SubjectArea)
    if isinstance(subject_area, (tuple, list)) and len(subject_area) >= 2:
        cx = float(subject_area[0])
        cy = float(subject_area[1])
    else:
        cx = float(W) / 2.0
        cy = float(H) / 2.0

    return fx, fy, cx, cy


def depth_to_organized_point_cloud(depth, fx, fy, cx, cy):
    H, W = depth.shape

    u, v = np.meshgrid(np.arange(W), np.arange(H))

    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    pc = np.stack((X, Y, Z), axis=-1)  # (H, W, 3)
    return pc


@click.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(image_path: Path):
    print(get_camera_intrinsics_from_exif_in_heic_image(image_path))
    pass

if __name__ == "__main__":
    main()
