FROM roboflow/l4t-ml:r35.2.1-py3.12-cu118-trt-10-v0.0.1

COPY requirements/requirements.clip.txt \
    requirements/requirements.http.txt \
    requirements/requirements.doctr.txt \
    requirements/requirements.groundingdino.txt \
    requirements/requirements.sdk.http.txt \
    requirements/requirements.yolo_world.txt \
    requirements/_requirements.txt \
    requirements/requirements.easyocr.txt \
    requirements/requirements.gpu.txt \
    ./

RUN python -m pip install \
    -r _requirements.txt \
    -r requirements.clip.txt \
    -r requirements.http.txt \
    -r requirements.doctr.txt \
    -r requirements.groundingdino.txt \
    -r requirements.sdk.http.txt \
    -r requirements.yolo_world.txt \
    -r requirements.easyocr.txt \
    -r requirements.gpu.txt \
    "pycuda>=2025.0.0,<2026.0.0"


WORKDIR /app/
COPY inference inference
COPY inference_cli inference_cli
COPY inference_sdk inference_sdk
COPY docker/config/gpu_http.py gpu_http.py
COPY .release .release
COPY requirements requirements
COPY Makefile Makefile

RUN make create_inference_cli_whl PYTHON=python3.12
RUN python -m pip install dist/inference_cli*.whl

ENV VERSION_CHECK_MODE=continuous \
    PROJECT=roboflow-platform \
    ORT_TENSORRT_FP16_ENABLE=1 \
    ORT_TENSORRT_ENGINE_CACHE_ENABLE=1 \
    CORE_MODEL_SAM_ENABLED=False \
    PROJECT=roboflow-platform \
    NUM_WORKERS=1 \
    HOST=0.0.0.0 \
    PORT=9001 \
    OPENBLAS_CORETYPE=ARMV8 \
    LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 \
    WORKFLOWS_STEP_EXECUTION_MODE=local \
    WORKFLOWS_MAX_CONCURRENT_STEPS=2 \
    API_LOGGING_ENABLED=True \
    CORE_MODEL_TROCR_ENABLED=false \
    RUNS_ON_JETSON=True \
    ENABLE_PROMETHEUS=True \
    ENABLE_STREAM_API=True \
    STREAM_API_PRELOADED_PROCESSES=2 \
    PYTHONPATH=/app:$PYTHONPATH
ENV CORE_MODEL_SAM3_ENABLED=False \
    ALLOW_INFERENCE_EXP_UNTRUSTED_MODELS=True \
    USE_INFERENCE_EXP_MODELS=False

ENTRYPOINT uvicorn gpu_http:app --workers $NUM_WORKERS --host $HOST --port $PORT