import os
import time

import cv2
from tqdm import tqdm

from inference import get_model

IMAGE_PATH = os.environ["IMAGE_PATH"]
MODEL = os.environ["MODEL"]
E2E_CYCLES = int(os.getenv("E2E_CYCLES", "1000"))
PRE_PROCESS_CYCLES = int(os.getenv("PRE_PROCESS_CYCLES", "5000"))
PREDICT_PROCESS_CYCLES = int(os.getenv("PREDICT_PROCESS_CYCLES", "500"))
POST_PROCESS_CYCLES = int(os.getenv("POST_PROCESS_CYCLES", "5000"))


def main() -> None:
    image = cv2.imread(IMAGE_PATH)
    print(f"Input image shape: {image.shape}")
    model = get_model(MODEL)
    print(f"Loaded model: {MODEL}")
    pre_processed_image, pre_processed_metadata = model.preprocess(image)
    raw_predictions = model.predict(pre_processed_image)
    start = time.monotonic()
    for _ in tqdm(range(E2E_CYCLES), total=E2E_CYCLES, desc="Inference e2e"):
        _ = model.infer(image)
    end = time.monotonic()
    fps = E2E_CYCLES / (end - start)
    print(f"INFERENCE FPS={round(fps, 2)}")
    start = time.monotonic()
    for _ in tqdm(
        range(PRE_PROCESS_CYCLES), total=PRE_PROCESS_CYCLES, desc="Preprocessing"
    ):
        _ = model.preprocess(image)
    end = time.monotonic()
    fps = PRE_PROCESS_CYCLES / (end - start)
    print(f"PRE PROCESSING FPS={round(fps, 2)}")
    start = time.monotonic()
    for _ in tqdm(
        range(PREDICT_PROCESS_CYCLES), total=PREDICT_PROCESS_CYCLES, desc="Prediction"
    ):
        _ = model.predict(pre_processed_image)
    end = time.monotonic()
    fps = PREDICT_PROCESS_CYCLES / (end - start)
    print(f"PREDICTION FPS={round(fps, 2)}")
    start = time.monotonic()
    for _ in tqdm(
        range(POST_PROCESS_CYCLES), total=POST_PROCESS_CYCLES, desc="Post processing"
    ):
        _ = model.postprocess(raw_predictions, pre_processed_metadata)
    end = time.monotonic()
    fps = POST_PROCESS_CYCLES / (end - start)
    print(f"POST PROCESSING FPS={round(fps, 2)}")


if __name__ == "__main__":
    main()
