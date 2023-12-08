import os
from inference.core.env import (
    REDIS_PORT,
    NUM_WORKERS,
    HOST,
    PORT,
    NUM_CELERY_WORKERS,
    CELERY_LOG_LEVEL,
)

os.system(
    f'redis-server --io-threads 8 --save ""--port {REDIS_PORT} &'
    f"celery -A inference.enterprise.parallel.tasks worker --prefetch-multiplier=4 --concurrency={NUM_CELERY_WORKERS} -Q pre  --loglevel={CELERY_LOG_LEVEL} &"
    f"celery -A inference.enterprise.parallel.tasks worker --prefetch-multiplier=4 --concurrency={NUM_CELERY_WORKERS} -Q post  --loglevel={CELERY_LOG_LEVEL} &"
    f"python3 inference/enterprise/parallel/infer.py &"
    f"gunicorn parallel_http:app --workers={NUM_WORKERS} --bind={HOST}:{PORT} -k uvicorn.workers.UvicornWorker && fg "
)
