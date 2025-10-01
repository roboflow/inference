
# Divert the program flow in worker sub-process as soon as possible,
# before importing heavy-weight modules.
import multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()    


import logging
import os
import sys

# Set up logging configuration for bundled app
# Enable in-memory logging for the FastAPI server to use
os.environ.setdefault("ENABLE_IN_MEMORY_LOGS", "True")
os.environ.setdefault("ENABLE_DASHBOARD", "True")

# Set up minimal console logging (only warnings and errors)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)

logger = logging.getLogger("inference.app")
import certifi
from platformdirs import user_cache_dir, user_data_dir


def setup_runtime_cache_env(app_name="roboflow-inference"):
    """
    Sets environment variables for all common runtime cache/data usage,
    so bundled apps (like PyInstaller builds) can write and reuse data
    without errors.
    """
    cache_dir = user_cache_dir(app_name)
    data_dir = user_data_dir(app_name)

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Common Python libs that rely on cached data
    os.environ.setdefault("TLD_EXTRACT_CACHE", os.path.join(cache_dir, "tldextract"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_dir, "torch"))
    os.environ.setdefault("HF_HOME", os.path.join(data_dir, "huggingface"))
    os.environ.setdefault("MATPLOTLIBCONFIGDIR", os.path.join(cache_dir, "matplotlib"))
    os.environ.setdefault("MODEL_CACHE_DIR", os.path.join(cache_dir, "models"))

    logger.info("🧠 Runtime cache environment configured:")
    logger.info(f" - TLD_EXTRACT_CACHE: {os.environ['TLD_EXTRACT_CACHE']}")
    logger.info(f" - MATPLOTLIBCONFIGDIR: {os.environ['MATPLOTLIBCONFIGDIR']}")
    logger.info(f" - TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
    logger.info(f" - TORCH_HOME: {os.environ['TORCH_HOME']}")
    logger.info(f" - HF_HOME: {os.environ['HF_HOME']}")
    logger.info(f" - MODEL_CACHE_DIR: {os.environ['MODEL_CACHE_DIR']}")

    return {
        "cache_dir": cache_dir,
        "data_dir": data_dir
    }


# Determine app_dir
if getattr(sys, 'frozen', False):
    logger.info("Launching Roboflow Inference (bundle)")

    app_dir = os.path.dirname(sys.executable)

    bundled_site_packages = os.path.join(os.path.dirname(sys.executable), 'site-packages')
    sys.path.insert(0, bundled_site_packages)

    # Set GDAL_DATA environment variable
    import rasterio
    gdal_data = os.path.join(os.path.dirname(rasterio.__file__), 'gdal_data')
    os.environ['GDAL_DATA'] = gdal_data

    #setup global cache env needed for tldexract and other packages
    setup_runtime_cache_env()

else:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info("Launching Roboflow Inference (source)")


logger.info("Initializing services")
os.chdir(app_dir)


logger.info("Configuring environment")
# Fix for SSL certs in PyInstaller bundle
os.environ["SSL_CERT_FILE"] = certifi.where()

# Set default env vars
os.environ.setdefault("VERSION_CHECK_MODE", "continuous")
os.environ.setdefault("PROJECT", "roboflow-platform")
os.environ.setdefault("NUM_WORKERS", "1")
os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "9001")
os.environ.setdefault("WORKFLOWS_STEP_EXECUTION_MODE", "local")
os.environ.setdefault("WORKFLOWS_MAX_CONCURRENT_STEPS", "4")
os.environ.setdefault("API_LOGGING_ENABLED", "True")
os.environ.setdefault("CORE_MODEL_SAM2_ENABLED", "True")
os.environ.setdefault("CORE_MODEL_OWLV2_ENABLED", "True")
os.environ.setdefault("ENABLE_STREAM_API", "True")
os.environ.setdefault("ENABLE_WORKFLOWS_PROFILING", "False")
os.environ.setdefault("ENABLE_PROMETHEUS", "True")
os.environ.setdefault("ENABLE_BUILDER", "True")


# # Force all foundational model imports
# can uncomment this to to debug missing dependencies or hidden
# modules needed for pyinstaller (the improt errors otherewise get
# swallowed in inference )

# # Modules behind conditionals (force them in unconditionally)
# import inference.models.clip as _clip
# import inference.models.gaze as _gaze
# import inference.models.sam as _sam
# import inference.models.sam2 as _sam2
# # import inference.models.doctr as _doctr
# import inference.models.grounding_dino as _grounding_dino
# import inference.models.yolo_world as _yolo_world

# import inference.models.paligemma as _paligemma
# import inference.models.florence2 as _florence2
# import inference.models.qwen25vl as _qwen25vl
# import inference.models.trocr as _trocr

# # Models that are always imported
# import inference.models.resnet as _resnet
# import inference.models.vit as _vit
# import inference.models.yolact as _yolact
# import inference.models.yolonas as _yolonas
# import inference.models.yolov5 as _yolov5
# import inference.models.yolov7 as _yolov7
# import inference.models.yolov8 as _yolov8
# import inference.models.yolov9 as _yolov9
# import inference.models.yolov10 as _yolov10
# import inference.models.yolov11 as _yolov11
# import inference.models.yolov12 as _yolov12
# import inference.models.rfdetr as _rfdetr



if __name__ == "__main__":
    logger.info("Starting server")
    # Import the FastAPI app
    from cpu_http import app
    import uvicorn
    import asyncio

    class FilteredAccessLogConfig(logging.Filter):
        """Filter out static file requests from access logs"""
        def filter(self, record):
            message = record.getMessage()
            # Filter out static paths and root requests (any HTTP method)
            if '/static' in message or '/_next/static' in message or ' / HTTP' in message:
                return False
            return True

    async def _serve_with_banner():
        port = int(os.environ.get("PORT", "9001"))
        url = f"http://localhost:{port}/"
        
        # Configure access log filtering
        access_logger = logging.getLogger("uvicorn.access")
        access_logger.addFilter(FilteredAccessLogConfig())
        
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True,
        )
        server = uvicorn.Server(config)
        serve_task = asyncio.create_task(server.serve())
        try:
            await server.started.wait()
        except Exception:
            # Fallback if the readiness event is unavailable
            await asyncio.sleep(0.5)

        # After startup: remove console stream handlers (keeps non-stream handlers like memory handlers)
        def _remove_console_handlers(logger_name: str):
            lg = logging.getLogger(logger_name)
            for handler in list(lg.handlers):
                if isinstance(handler, logging.StreamHandler):
                    lg.removeHandler(handler)

        for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access", "inference", "inference.app"):
            _remove_console_handlers(name)
        banner = (
            "\n\n\n\n\n\n\n\n\n"
            "─────────────────────────────────────────────────────────── \n"
            "                                                           \n"
            "  Roboflow Inference is ready                              \n"
            f"  Dashboard: {url:<44} │\n"
            "                                                           \n"
            "───────────────────────────────────────────────────────────\n"
        )
        print(banner, flush=True)
        await serve_task

    try:
        asyncio.run(_serve_with_banner())
    except Exception as e:
        logger.exception("Error starting server: %s", e)
        sys.exit(1)
