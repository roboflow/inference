import torch
from transformers import AutoModelForDepthEstimation, DepthAnythingConfig, AutoImageProcessor
from PIL import Image
import numpy as np
from pillow_heif import open_heif
from examples.indoor_design.plane_detection.utils import get_camera_intrinsics_from_exif_in_heic_image
from examples.indoor_design.plane_detection.visualizations import get_point_cloud_3d_fig
from examples.indoor_design.plane_detection.utils import depth_to_organized_point_cloud


def depth_to_3d_points(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.stack((X, Y, Z), axis=-1)


if __name__ == "__main__":
    device = torch.device("mps")
    config = DepthAnythingConfig(depth_estimation_type="metric", max_depth=4.7)

    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf",
        config=config,
    ).to(device)

    image_path = "../data/image.HEIC"
    image = open_heif(image_path).to_pillow()
    print(image.width, image.height)

    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    predicted_depth = predicted_depth.detach().cpu().numpy()
    
    print(predicted_depth.min(), predicted_depth.max(), predicted_depth.mean())

    normalized_depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    normalized_depth = normalized_depth * 255
    normalized_depth = Image.fromarray(normalized_depth.astype("uint8"))
    print(normalized_depth.width, normalized_depth.height)
    normalized_depth.show()

    
    fx, fy, cx, cy = get_camera_intrinsics_from_exif_in_heic_image(image_path)

    organized_point_cloud = depth_to_organized_point_cloud(predicted_depth, fx, fy, cx, cy)
    point_cloud = organized_point_cloud.reshape(-1, 3)

    fig = get_point_cloud_3d_fig(point_cloud)
    fig.show()

    np.save("../data/organized_point_cloud.npy", organized_point_cloud)
