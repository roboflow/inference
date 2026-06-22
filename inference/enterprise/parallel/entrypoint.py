import os

from inference.core.env import (
    CELERY_LOG_LEVEL,
    ENABLE_HTTPS,
    HOST,
    NUM_CELERY_WORKERS,
    NUM_WORKERS,
    PORT,
    REDIS_PORT,
    SSL_CA_CERTS,
    SSL_CERTFILE,
    SSL_KEYFILE,
    SSL_KEYFILE_PASSWORD,
)


def _gunicorn_ssl_flags() -> str:
    if not ENABLE_HTTPS:
        return ""
    if not SSL_CERTFILE or not SSL_KEYFILE:
        raise RuntimeError(
            "ENABLE_HTTPS is set but SSL_CERTFILE and SSL_KEYFILE must both be configured."
        )
    flags = f"--certfile={SSL_CERTFILE} --keyfile={SSL_KEYFILE}"
    if SSL_KEYFILE_PASSWORD:
        flags += f" --ssl-keyfile-password={SSL_KEYFILE_PASSWORD}"
    if SSL_CA_CERTS:
        flags += f" --ca-certs={SSL_CA_CERTS}"
    return " " + flags


os.system(
    f'redis-server --io-threads 8 --save ""--port {REDIS_PORT} &'
    f"celery -A inference.enterprise.parallel.tasks worker --prefetch-multiplier=4 --concurrency={NUM_CELERY_WORKERS} -Q pre  --loglevel={CELERY_LOG_LEVEL} &"
    f"celery -A inference.enterprise.parallel.tasks worker --prefetch-multiplier=4 --concurrency={NUM_CELERY_WORKERS} -Q post  --loglevel={CELERY_LOG_LEVEL} &"
    f"python3 inference/enterprise/parallel/infer.py &"
    f"gunicorn parallel_http:app --workers={NUM_WORKERS} --bind={HOST}:{PORT} -k uvicorn.workers.UvicornWorker{_gunicorn_ssl_flags()} && fg "
)
