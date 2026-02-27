import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch
from pillow_heif import open_heif
from transformers import AutoModelForDepthEstimation, DepthAnythingConfig, AutoImageProcessor

from examples.indoor_design.plane_detection.utils import (
    get_camera_intrinsics_from_exif_in_heic_image,
    inverse_depth_to_organized_point_cloud,
)
from examples.indoor_design.plane_detection.visualizations import (
    get_inverse_depth_heatmap_fig,
    get_point_cloud_3d_fig,
)


@click.command()
@click.option("--image-path", type=click.Path(exists=True), required=True, default=Path("../data/image.HEIC"))
@click.option("--output-dir", type=click.Path(exists=False), required=True, default=Path(f"../data/metric_depth_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
def main(image_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps")
    config = DepthAnythingConfig(depth_estimation_type="metric", max_depth=20)

    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf",
        config=config,
    ).to(device)

    image = open_heif(image_path).to_pillow()

    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    inverse_depth = post_processed_output[0]["predicted_depth"]
    inverse_depth = inverse_depth.detach().cpu().numpy()

    fig_depth = get_inverse_depth_heatmap_fig(inverse_depth)
    fig_depth.write_html(output_dir / "inverse_depth_heatmap.html", include_plotlyjs="cdn")

    fx, fy, cx, cy = get_camera_intrinsics_from_exif_in_heic_image(image_path)

    organized_point_cloud = inverse_depth_to_organized_point_cloud(inverse_depth, fx, fy, cx, cy)
    point_cloud = organized_point_cloud.reshape(-1, 3)

    fig = get_point_cloud_3d_fig(point_cloud)
    fig.write_html(output_dir / "point_cloud.html", include_plotlyjs="cdn")

    np.save(output_dir / "organized_point_cloud.npy", organized_point_cloud)

    stats = {
        "image_width": image.width,
        "image_height": image.height,
        "focal_length_x": fx,
        "focal_length_y": fy,
        "principal_point_x": cx,
        "principal_point_y": cy,
        "inverse_depth_min": float(inverse_depth.min()),
        "inverse_depth_max": float(inverse_depth.max()),
        "inverse_depth_mean": float(inverse_depth.mean()),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
