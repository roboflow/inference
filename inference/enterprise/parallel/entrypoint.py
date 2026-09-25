import os
import shlex

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
    flags = (
        f"--certfile={shlex.quote(SSL_CERTFILE)} --keyfile={shlex.quote(SSL_KEYFILE)}"
    )
    if SSL_KEYFILE_PASSWORD:
        raise RuntimeError(
            "The parallel Gunicorn launcher does not support SSL_KEYFILE_PASSWORD. "
            "Use the Uvicorn launcher for encrypted TLS keys; no key password is ignored."
        )
    if SSL_CA_CERTS:
        flags += f" --ca-certs={shlex.quote(SSL_CA_CERTS)} --cert-reqs=2"
    return " " + flags


os.system(
    f'redis-server --io-threads 8 --save ""--port {REDIS_PORT} &'
    f"celery -A inference.enterprise.parallel.tasks worker --prefetch-multiplier=4 --concurrency={NUM_CELERY_WORKERS} -Q pre  --loglevel={CELERY_LOG_LEVEL} &"
    f"celery -A inference.enterprise.parallel.tasks worker --prefetch-multiplier=4 --concurrency={NUM_CELERY_WORKERS} -Q post  --loglevel={CELERY_LOG_LEVEL} &"
    f"python3 inference/enterprise/parallel/infer.py &"
    f"gunicorn parallel_http:app --workers={NUM_WORKERS} --bind={HOST}:{PORT} -k uvicorn.workers.UvicornWorker{_gunicorn_ssl_flags()} && fg "
)
