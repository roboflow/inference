from pathlib import Path

import click
import numpy as np
import piexif
from pillow_heif import open_heif
from piexif import TAGS



def get_camera_intrinsics_from_exif_in_heic_image(image_path: str) -> tuple[float, float, float, float]:
    """Return approximate pinhole intrinsics (fx, fy, cx, cy) for a HEIC image.

    Notes:
    - Uses EXIF `FocalLengthIn35mmFilm` (recommended) to estimate focal length in pixels.
    - Assumes square pixels => fx == fy.
    - Uses EXIF `SubjectArea` (if present) as a hint for principal point; otherwise uses image center.

    Returns:
        (fx, fy, cx, cy) in pixel units.
    """

    def _rat_to_float(x) -> float:
        # piexif encodes rationals as (num, den)
        if isinstance(x, tuple) and len(x) == 2 and x[1] != 0:
            return float(x[0]) / float(x[1])
        return float(x)

    heif = open_heif(image_path)

    exif_bytes = heif.info.get("exif")
    if not exif_bytes:
        raise ValueError("No EXIF data found in the image")

    exif_dict = piexif.load(exif_bytes)

    # Prefer EXIF pixel dimensions; fall back to container dimensions.
    W = exif_dict.get("Exif", {}).get(piexif.ExifIFD.PixelXDimension)
    H = exif_dict.get("Exif", {}).get(piexif.ExifIFD.PixelYDimension)

    if W is None or H is None:
        # pillow-heif exposes size on the object
        try:
            W, H = heif.size
        except Exception as e:
            raise ValueError("Could not determine image dimensions from EXIF or HEIF container") from e

    W = int(W)
    H = int(H)

    # Prefer 35mm equivalent focal length if available.
    f35 = exif_dict.get("Exif", {}).get(piexif.ExifIFD.FocalLengthIn35mmFilm)

    if f35 is not None:
        f35 = float(f35)
        # Convert 35mm-equivalent focal length to pixel focal length.
        # Full-frame width is 36mm (reference), so f_px ≈ (f35/36) * image_width_px.
        f_px = (f35 / 36.0) * float(W)
    else:
        # Fallback: estimate via physical focal length + sensor crop factor inferred from EXIF LensSpecification.
        # This is less reliable but better than failing.
        f_mm = exif_dict.get("Exif", {}).get(piexif.ExifIFD.FocalLength)
        if f_mm is None:
            raise ValueError(
                "Neither `FocalLengthIn35mmFilm` nor `FocalLength` found in EXIF. Cannot estimate intrinsics."
            )

        f_mm = _rat_to_float(f_mm)

        # Heuristic fallback: assume typical smartphone sensor diagonal ≈ 7.0mm
        # => crop ≈ 43.3 / 7.0
        crop = 43.3 / 7.0
        f35_est = f_mm * crop
        f_px = (f35_est / 36.0) * float(W)

    fx = f_px
    fy = f_px  # assume square pixels

    # Principal point: SubjectArea is often (x, y, w, h) where (x, y) is the focus point.
    subject_area = exif_dict.get("Exif", {}).get(piexif.ExifIFD.SubjectArea)
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
