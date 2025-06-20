import argparse
import os

from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

from .dataset import download_dataset
from .serialization import serialize_results, dump_json


def main(
    model_id: str,
    output_dir: str,
    image_size: int,
) -> None:
    dataset = download_dataset()
    model = YOLO(f"{model_id}.pt")
    results = []
    for image_id, image in tqdm(dataset, desc="Making predictions..."):
        predictions = model(image, imgsz=image_size, verbose=False)
        if getattr(predictions, "keypoints") is not None:
            predictions = sv.KeyPoints.from_ultralytics(predictions[0])
        else:
            predictions = sv.Detections.from_ultralytics(predictions[0])
        serialized = serialize_results(predictions=predictions)
        results.append((image_id, serialized))
    for image_id, serialized in tqdm(results, desc="Saving results"):
        target_path = os.path.join(output_dir, f"{image_id}.json")
        dump_json(path=target_path, content=serialized)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    args = parser.parse_args()
    main(
        model_id=args.model_id,
        output_dir=args.output_dir,
        image_size=args.image_size,
    )
