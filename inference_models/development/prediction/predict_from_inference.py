import argparse
import os.path
from typing import Optional

from inference_models import AutoModel
from tqdm import tqdm

from .dataset import download_dataset
from .serialization import dump_json, serialize_results


def main(
    model_id: str,
    output_dir: str,
    model_package_id: Optional[str] = None,
) -> None:
    dataset = download_dataset()
    model = AutoModel.from_pretrained(
        model_id_or_path=model_id, model_package_id=model_package_id
    )
    results = []
    for image_id, image in tqdm(dataset, desc="Making predictions..."):
        predictions = model(image)
        serialized = serialize_results(predictions=predictions[0].to_supervision())
        results.append((image_id, serialized))
    for image_id, serialized in tqdm(results, desc="Saving results"):
        target_path = os.path.join(output_dir, f"{image_id}.json")
        dump_json(path=target_path, content=serialized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_package_id", type=str, required=False, default=None)
    args = parser.parse_args()
    main(
        model_id=args.model_id,
        output_dir=args.output_dir,
        model_package_id=args.model_package_id,
    )
