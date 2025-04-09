import argparse
import base64
import os
from typing import List

import cv2
import requests
import supervision as sv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process image and annotate with detections."
    )

    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image file."
    )
    parser.add_argument(
        "--class_list",
        type=str,
        nargs="+",
        required=True,
        help="List of classes to detect in the image.",
    )
    parser.add_argument(
        "--dataset_id", type=str, required=True, help="Dataset ID for the API request."
    )
    parser.add_argument(
        "--version_id", type=str, required=True, help="Version ID for the API request."
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for the detections.",
    )

    return parser.parse_args()


def annotate_image(
    image_path: str,
    class_list: List[str],
    dataset_id: str,
    version_id: str,
    confidence: float,
    api_key: str,
) -> None:
    url = f"http://localhost:9001/{dataset_id}/{version_id}"
    headers = {"Content-Type": "application/json"}
    params = {
        "api_key": api_key,
        "confidence": confidence,
    }

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    image = cv2.imread(image_path)
    response = requests.post(
        url, headers=headers, params=params, data=encoded_image
    ).json()
    detections = sv.Detections.from_inference(response)

    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{class_list[class_id]} {det_confidence:0.2f}"
        for _, _, det_confidence, class_id, _ in detections
    ]
    annotated_image = box_annotator.annotate(
        image, detections=detections, labels=labels
    )

    cv2.imshow("Annotated image", annotated_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    args = parse_arguments()

    API_KEY = os.environ.get("API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY not found in environment variables.")

    annotate_image(
        args.image_path,
        args.class_list,
        args.dataset_id,
        args.version_id,
        args.confidence,
        API_KEY,
    )
