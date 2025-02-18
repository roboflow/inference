import os

os.environ["TELEMETRY_OPT_OUT"] = "True"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider,CPUExecutionProvider]"
