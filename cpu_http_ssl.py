"""SSL-enabled HTTP server for Inference

This module extends the standard HTTP server with SSL support when enabled.
"""

import os
import sys
import multiprocessing
from multiprocessing import Process
from functools import partial
from typing import Optional, Tuple

from inference.core.cache import cache
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.interfaces.stream_manager.manager_app.app import start
from inference.core.managers.active_learning import (
    ActiveLearningManager,
    BackgroundTaskActiveLearningManager,
)
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference.core import logger
from inference.core.env import (
    MAX_ACTIVE_MODELS,
    ACTIVE_LEARNING_ENABLED,
    LAMBDA,
    ENABLE_STREAM_API,
    STREAM_API_PRELOADED_PROCESSES,
    ENABLE_SSL,
    SSL_PORT,
    SSL_CERTIFICATE,
    HOST,
    PORT,
    NUM_WORKERS,
)

# Create the app instance (shared between HTTP and HTTPS)
if ENABLE_STREAM_API:
    stream_manager_process = Process(
        target=partial(start, expected_warmed_up_pipelines=STREAM_API_PRELOADED_PROCESSES),
    )
    stream_manager_process.start()

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)

if ACTIVE_LEARNING_ENABLED:
    if LAMBDA:
        model_manager = ActiveLearningManager(
            model_registry=model_registry, cache=cache
        )
    else:
        model_manager = BackgroundTaskActiveLearningManager(
            model_registry=model_registry, cache=cache
        )
else:
    model_manager = ModelManager(model_registry=model_registry)

model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)
model_manager.init_pingback()
interface = HttpInterface(model_manager)
app = interface.app


def get_ssl_config() -> Optional[Tuple[str, str]]:
    """Get SSL certificate configuration if SSL is enabled"""
    if not ENABLE_SSL:
        return None
        
    from inference.core.interfaces.http.ssl import SSLCertificateManager
    
    cert_manager = SSLCertificateManager()
    cert_path, key_path = cert_manager.get_certificate_paths(SSL_CERTIFICATE)
    
    return cert_path, key_path


def start_https_server():
    """Start the HTTPS server"""
    import uvicorn
    
    ssl_config = get_ssl_config()
    if not ssl_config:
        logger.error("SSL is enabled but no certificate configuration found")
        return
        
    cert_path, key_path = ssl_config
    
    # Determine SSL port
    ssl_port = SSL_PORT
    if ssl_port == 9002 and PORT != 9001:  # Use default logic
        ssl_port = PORT + 1 if PORT != 80 else 443
    
    logger.info(f"Starting HTTPS server on {HOST}:{ssl_port}")
    
    try:
        uvicorn.run(
            app,
            host=HOST,
            port=ssl_port,
            workers=1,  # SSL server uses single worker to simplify cert management
            ssl_keyfile=key_path,
            ssl_certfile=cert_path,
            log_level="warning",
        )
    except Exception as e:
        logger.error(f"Failed to start HTTPS server: {e}")
        raise


# Start HTTPS server in a separate process if SSL is enabled
if __name__ == "__main__" and ENABLE_SSL:
    https_process = Process(target=start_https_server)
    https_process.start()
    logger.info("Started HTTPS server process")
