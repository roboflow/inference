import argparse
import os
import pickle
from typing import List

import cv2
import requests
import supervision as sv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process video and annotate frames with detections."
    )

    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the video file."
    )
    parser.add_argument(
        "--class_list",
        type=str,
        nargs="+",
        required=True,
        help="List of classes to detect in the video.",
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


def process_and_annotate_frames(
    video_path: str,
    class_list: List[str],
    dataset_id: str,
    version_id: str,
    confidence: float,
    api_key: str,
) -> None:
    url = f"http://localhost:9001/{dataset_id}/{version_id}?image_type=numpy"
    headers = {"Content-Type": "application/json"}
    params = {
        "api_key": api_key,
        "confidence": confidence,
    }

    box_annotator = sv.BoxAnnotator()

    for frame in sv.get_video_frames_generator(source_path=video_path):
        numpy_data = pickle.dumps(frame)
        response = requests.post(
            url, headers=headers, params=params, data=numpy_data
        ).json()
        detections = sv.Detections.from_inference(response)
        labels = [
            f"{class_list[class_id]} {confidence_value:0.2f}"
            for _, _, confidence_value, class_id, _ in detections
        ]
        annotated_image = box_annotator.annotate(
            frame, detections=detections, labels=labels
        )
        cv2.imshow("Annotated image", annotated_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    args = parse_arguments()

    API_KEY = os.environ.get("API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY not found in environment variables.")

    process_and_annotate_frames(
        args.video_path,
        args.class_list,
        args.dataset_id,
        args.version_id,
        args.confidence,
        API_KEY,
    )
