import os

from compilation.core import compile_model

MODELS_TO_COMPILE = [
    "rfdetr-base",
    "rfdetr-large",
]

MODELS_OUTPUT_DIR = "/model-compilation"
WORKSPACE_SIZE_IN_GB = 15

for model_id in MODELS_TO_COMPILE:
    for precision in ["fp16", "fp32"]:
        target_path = os.path.join(MODELS_OUTPUT_DIR, f"{model_id}-{precision}")
        try:
            compile_model(
                model_id,
                target_path=target_path,
                precision=precision,
                workspace_size_gb=WORKSPACE_SIZE_IN_GB,
                model_input_size=(560, 560),
            )
        except Exception as error:
            print(f"Could not finish compilation: {error}")
