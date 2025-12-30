import os
import time

import cv2
import torch
import torchvision
from inference_models.models.yolov8.yolov8_object_detection_trt import (
    YOLOv8ForObjectDetectionTRT,
)
from tqdm import tqdm

IMAGE_PATH = os.environ["IMAGE_PATH"]
MODEL_PACKAGE = os.environ["MODEL_PACKAGE"]
E2E_CYCLES = int(os.getenv("E2E_CYCLES", "1000"))
PRE_PROCESS_CYCLES = int(os.getenv("PRE_PROCESS_CYCLES", "5000"))
PREDICT_PROCESS_CYCLES = int(os.getenv("PREDICT_PROCESS_CYCLES", "500"))
POST_PROCESS_CYCLES = int(os.getenv("POST_PROCESS_CYCLES", "5000"))
DEVICE = os.environ["DEVICE"]
TORCH_INPUT = bool(int(os.getenv("TORCH_INPUT", "0")))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))


def main() -> None:
    if TORCH_INPUT:
        image = torchvision.io.read_image(IMAGE_PATH).to(DEVICE)
    else:
        image = cv2.imread(IMAGE_PATH)
    print(f"Input image shape: {image.shape}")
    if BATCH_SIZE > 1:
        image = [image] * BATCH_SIZE
    print(f"BS={BATCH_SIZE}")
    model = YOLOv8ForObjectDetectionTRT.from_pretrained(
        MODEL_PACKAGE, device=torch.device(DEVICE)
    )
    pre_processed_image, pre_processed_metadata = model.pre_process(image)
    raw_predictions = model.forward(pre_processed_image)
    start = time.monotonic()
    for _ in tqdm(range(E2E_CYCLES), total=E2E_CYCLES, desc="Inference e2e"):
        _ = model(image)
    end = time.monotonic()
    fps = E2E_CYCLES / (end - start)
    print(f"INFERENCE FPS={round(fps * BATCH_SIZE, 2)}")
    start = time.monotonic()
    for _ in tqdm(
        range(PRE_PROCESS_CYCLES), total=PRE_PROCESS_CYCLES, desc="Preprocessing"
    ):
        _ = model.pre_process(image)
    end = time.monotonic()
    fps = PRE_PROCESS_CYCLES / (end - start)
    print(f"PRE PROCESSING FPS={round(fps * BATCH_SIZE, 2)}")
    start = time.monotonic()
    for _ in tqdm(
        range(PREDICT_PROCESS_CYCLES), total=PREDICT_PROCESS_CYCLES, desc="Prediction"
    ):
        _ = model.forward(pre_processed_image)
    end = time.monotonic()
    fps = PREDICT_PROCESS_CYCLES / (end - start)
    print(f"PREDICTION FPS={round(fps * BATCH_SIZE, 2)}")
    start = time.monotonic()
    for _ in tqdm(
        range(POST_PROCESS_CYCLES), total=POST_PROCESS_CYCLES, desc="Post processing"
    ):
        _ = model.post_process(raw_predictions, pre_processed_metadata)
    end = time.monotonic()
    fps = POST_PROCESS_CYCLES / (end - start)
    print(f"POST PROCESSING FPS={round(fps * BATCH_SIZE, 2)}")


if __name__ == "__main__":
    main()
