import os
import shutil
from glob import glob

from compilation.core import compile_model

MODELS_TO_COMPILE = [
    "yolov8m-pose-640",
    "yolov8m-seg-640",
    "yolov8n-seg-1280",
    "yolov8s-seg-1280",
    "yolov8m-seg-1280",
    "yolov8m-640",
    "yolov8n-1280",
    "yolov8s-1280",
    "yolov8m-1280",
    "yolov10m-640",
]

MODELS_OUTPUT_DIR = "/model-compilation"
WORKSPACE_SIZE_IN_GB = 15

for model_id in MODELS_TO_COMPILE:
    for precision in ["fp16", "fp32"]:
        target_path = os.path.join(MODELS_OUTPUT_DIR, f"{model_id}-{precision}")
        model_input_size = (640, 640) if "-640" in model_id else (1280, 1280)
        try:
            compile_model(
                model_id,
                target_path=target_path,
                precision=precision,
                min_batch_size=1,
                opt_batch_size=8,
                max_batch_size=16,
                workspace_size_gb=WORKSPACE_SIZE_IN_GB,
                model_input_size=model_input_size,
            )
        except Exception as error:
            print(f"Could not finish compilation: {error}")
