import os

from compilation.core import compile_model

MODELS_OUTPUT_DIR = os.getenv("MODELS_OUTPUT_DIR", "/model-compilation")
WORKSPACE_SIZE_IN_GB = int(os.getenv("WORKSPACE_SIZE_GB", "15"))

model_id = os.getenv("MODEL_ID") or os.getenv("MODEL")
if not model_id:
    raise RuntimeError("MODEL_ID environment variable is required")

precisions_env = os.getenv("PRECISIONS", "fp16,fp32")
precisions = [p.strip() for p in precisions_env.split(",") if p.strip()]


def parse_int_from_env(var_name: str) -> int:
    value = os.getenv(var_name)
    if value is None or value == "":
        return None
    return int(value)


def parse_model_input_size_from_env() -> tuple:
    value = os.getenv("MODEL_INPUT_SIZE")
    if value is None or value == "":
        return None
    normalized = value.lower().replace(" ", "").replace("x", ",")
    parts = [p for p in normalized.split(",") if p]
    if len(parts) == 1:
        n = int(parts[0])
        return (n, n)
    if len(parts) == 2:
        h = int(parts[0])
        w = int(parts[1])
        return (h, w)
    raise RuntimeError("MODEL_INPUT_SIZE must be like '640', '640x640', or '640,640'")


model_input_size = parse_model_input_size_from_env()

min_batch_size = parse_int_from_env("MIN_BATCH_SIZE")
opt_batch_size = parse_int_from_env("OPT_BATCH_SIZE")
max_batch_size = parse_int_from_env("MAX_BATCH_SIZE")

if any(v is None for v in [min_batch_size, opt_batch_size, max_batch_size]):
    # Fall back to previous defaults when not fully provided
    min_batch_size = 1 if min_batch_size is None else min_batch_size
    opt_batch_size = 8 if opt_batch_size is None else opt_batch_size
    max_batch_size = 16 if max_batch_size is None else max_batch_size

for precision in precisions:
    target_path = os.path.join(MODELS_OUTPUT_DIR, f"{model_id}-{precision}")
    # try:
    compile_model(
        model_id,
        target_path=target_path,
        precision=precision,
        min_batch_size=min_batch_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size,
        workspace_size_gb=WORKSPACE_SIZE_IN_GB,
        model_input_size=model_input_size,
    )
    # except Exception as error:
    #     print(f"Could not finish compilation for {model_id} ({precision}): {error}")


# TODO: register model in registry
