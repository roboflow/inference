import os

# The unit tests compare image pipelines against standard Pillow, so a
# Pillow-SIMD build on the host must not bind into the modules under test.
os.environ["INFERENCE_MODELS_PILLOW_SIMD_PATH"] = ""
