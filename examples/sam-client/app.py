import argparse
import os

import cv2
import supervision as sv

from inference.models import SegmentAnything

parser = argparse.ArgumentParser(description="Segment images with SAM.")

parser.add_argument(
    "--image_path", type=str, required=True, help="Path to image to segment"
)
parser.add_argument(
    "--text_prompt", type=str, required=True, help="Text prompt for segmentation"
)
parser.add_argument(
    "--inference_endpoint",
    type=str,
    required=True,
    help="Roboflow Inference endpoint URL",
    default="http://localhost:9001",
)
parser.add_argument(
    "--api_key",
    type=str,
    required=True,
    help="Roboflow API key",
    default=os.environ.get("ROBOFLOW_API_KEY"),
)

args = parser.parse_args()

model = SegmentAnything(api_key=args.api_key)

inference_results = model.infer(args.image_path)

masks = inference_results["masks"]

image = cv2.imread(args.image_path)

mask_annotator = sv.MaskAnnotator()
detections = sv.Detections.from_sam(masks)
annotated_image = mask_annotator.annotate(image, detections)

sv.plot_image(annotated_image, (4, 4))