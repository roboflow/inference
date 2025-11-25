import numpy as np
import os
import PIL
from tqdm import tqdm
import json


def save_kubric_example(example: dict, save_dir: str, n_workers: int = 4):
    """Save Kubric example data to the specified directory.

    Args:
        example: Kubric dataset example containing video frames and metadata
        save_dir: Directory path where files should be saved
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Save metadata
    example["metadata"]["video_name"] = example["metadata"]["video_name"].decode(
        "utf-8"
    )
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        converted = convert_arrays_to_lists(example["metadata"])
        json.dump(converted, f)

    # Save instance info
    instance_info = {
        "asset_id": example["instances"]["asset_id"],
        "positions": example["instances"]["positions"],
        "quaternions": example["instances"]["quaternions"],
        "scale": example["instances"]["scale"],
        "visibility": example["instances"]["visibility"],
    }
    np.savez_compressed(os.path.join(save_dir, "instance_info.npz"), **instance_info)

    # Create subdirectories
    rgb_dir = os.path.join(save_dir, "rgb")
    instances_dir = os.path.join(save_dir, "instances")
    depth_dir = os.path.join(save_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(instances_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    from functools import partial
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def save_rgb_frame(args):
        frame_id, rgb, rgb_dir = args
        PIL.Image.fromarray(rgb).save(
            os.path.join(rgb_dir, f"frame_{frame_id:04d}.png")
        )

    def save_segmentation_frame(args):
        frame_id, segmentation, instances_dir = args
        PIL.Image.fromarray(segmentation.squeeze(), mode="L").save(
            os.path.join(instances_dir, f"frame_{frame_id:04d}.png")
        )

    def save_depth_frame(args):
        frame_id, depth, depth_dir = args
        PIL.Image.fromarray(depth.astype(np.uint16).squeeze()).save(
            os.path.join(depth_dir, f"frame_{frame_id:04d}.png")
        )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Save RGB frames
        rgb_futures = [
            executor.submit(save_rgb_frame, (i, frame, rgb_dir))
            for i, frame in enumerate(example["video"])
        ]

        # Save instance segmentation masks
        seg_futures = [
            executor.submit(save_segmentation_frame, (i, seg, instances_dir))
            for i, seg in enumerate(example["segmentations"])
        ]

        # Save depth maps
        depth_futures = [
            executor.submit(save_depth_frame, (i, depth, depth_dir))
            for i, depth in enumerate(example["depth"])
        ]

        # Wait for all futures to complete
        for future in as_completed(rgb_futures + seg_futures + depth_futures):
            future.result()  # This will raise any exceptions that occurred

    # Save camera parameters
    np.savez_compressed(os.path.join(save_dir, "camera.npz"), **example["camera"])


def process_kubric_dataset(
    data_dir: str,
    output_dir: str,
    dataset_name: str = "movi_d",
    split: str = "train",
    n_workers: int = 16,
):
    assert split in [
        "train",
        "validation",
    ], f"split must be either train or validation, not {split}"
    import tensorflow_datasets as tfds

    ds, ds_info = tfds.load(dataset_name, data_dir=data_dir, with_info=True)
    ds_iter = iter(tfds.as_numpy(ds[split]))
    len_ds = len(tfds.as_numpy(ds[split]))

    # def process_single_example(video_idx, example, output_dir, dataset_name, split):
    #     save_dir = os.path.join(output_dir, f"{dataset_name}/{split}/video_{video_idx:04d}")
    #     save_kubric_example(example, save_dir)
    #     return video_idx

    for video_idx, example in tqdm(enumerate(ds_iter), total=len_ds):
        save_dir = os.path.join(
            output_dir, f"{dataset_name}/{split}/video_{video_idx:04d}"
        )
        save_kubric_example(example, save_dir, n_workers=n_workers)


def convert_arrays_to_lists(data: dict) -> dict:
    """
    Converts from np things to json.
    Torch doesn't interop nicely with np.uint16
    """

    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.int32) or isinstance(value, np.uint16):
            return int(value)
        elif isinstance(value, np.float32):
            return float(value)
        elif isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [convert(v) for v in value]
        else:
            return value

    return {k: convert(v) for k, v in data.items()}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Kubric dataset from TensorFlow Datasets."
    )
    parser.add_argument(
        "--tfds_dir",
        type=str,
        required=True,
        help="Directory where TFDS will store/load the dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where processed data will be saved",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="movi_d",
        help="Name of the Kubric dataset to process (default: movi_d)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers to use for processing (default: number of CPUs)",
    )
    args = parser.parse_args()

    print(
        f"Processing {args.dataset_name} dataset split {args.split} with {args.n_workers} workers"
    )
    # exit()
    process_kubric_dataset(
        data_dir=args.tfds_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        split=args.split,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
