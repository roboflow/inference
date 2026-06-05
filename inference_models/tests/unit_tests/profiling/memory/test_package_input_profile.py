from __future__ import annotations

import json
from pathlib import Path

import pytest

from profiling.memory.package_input_profile import (
    InputProfileMismatchError,
    shape_spec_from_package_dir,
    validate_profiling_image_shapes,
    validate_profiling_image_shapes_for_package_dir,
)


def _write_inference_config(
    package_dir: Path,
    *,
    static_batch_size: int = 2,
    height: int = 640,
    width: int = 640,
    dynamic_spatial: bool = False,
) -> None:
    config = {
        "network_input": {
            "training_input_size": {"height": height, "width": width},
            "dynamic_spatial_size_supported": dynamic_spatial,
            "color_mode": "rgb",
            "resize_mode": "stretch",
            "input_channels": 3,
        },
        "forward_pass": {"static_batch_size": static_batch_size},
    }
    (package_dir / "inference_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )


def test_validate_accepts_matching_static_shapes(tmp_path: Path) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    _write_inference_config(package_dir)

    batch, height, width = validate_profiling_image_shapes_for_package_dir(
        package_dir,
        batch_size=2,
        height=640,
        width=640,
    )

    assert (batch, height, width) == (2, 640, 640)


def test_validate_rejects_mismatched_batch(tmp_path: Path) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    _write_inference_config(package_dir)

    with pytest.raises(InputProfileMismatchError) as error:
        validate_profiling_image_shapes_for_package_dir(
            package_dir,
            batch_size=4,
            height=640,
            width=640,
        )

    assert "batch" in str(error.value)


def test_dynamic_spatial_does_not_force_resolution(tmp_path: Path) -> None:
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    _write_inference_config(package_dir, dynamic_spatial=True)

    spec = shape_spec_from_package_dir(package_dir)
    assert spec is not None
    assert spec.height.resolution == "dynamic"

    validate_profiling_image_shapes(
        batch_size=2,
        height=1280,
        width=1280,
        spec=spec,
    )
