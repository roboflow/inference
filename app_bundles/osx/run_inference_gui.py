import multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()

import logging
import threading
import webview
import json
import os
import sys
import time
import urllib.request

# Setup logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("inference.app")

# HTML for the webview window
HTML = """
<!doctype html>
<html>
  <body style="font-family: -apple-system, Menlo; margin:0">
    <div style="display:flex;align-items:center;gap:8px;padding:8px;border-bottom:1px solid #eee">
      <span id="status">Starting…</span>
      <button onclick="openDash()">Open Dashboard</button>
    </div>
    <pre id="log" style="margin:0;padding:8px;height:calc(100vh - 42px);overflow:auto;background:#111;color:#ddd"></pre>
    <script>
      function append(line){ 
        const el = document.getElementById('log'); 
        el.textContent += line; 
        el.scrollTop = el.scrollHeight; 
      }
      function setStatus(s){ 
        document.getElementById('status').textContent = s; 
      }
      function openDash(){ 
        window.pywebview.api.open_dashboard(); 
      }
    </script>
  </body>
</html>
"""

class Api:
    def open_dashboard(self):
        import webbrowser
        webbrowser.open(f"http://localhost:{os.environ.get('PORT','9001')}")

class GuiLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record) + "\n"
        except Exception:
            msg = record.getMessage() + "\n"
        # Push to the webview via JS
        try:
            if webview.windows:
                webview.windows[0].evaluate_js(f"append({json.dumps(msg)})")
        except Exception:
            pass

# Global flag for clean shutdown
shutdown_event = threading.Event()
server_thread = None

def run_server():
    """Run the inference server in a background thread"""
    try:
        # Import all the setup code from run_inference.py
        import certifi
        from platformdirs import user_cache_dir, user_data_dir
        
        def setup_runtime_cache_env(app_name="roboflow-inference"):
            cache_dir = user_cache_dir(app_name)
            data_dir = user_data_dir(app_name)
            
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            
            os.environ.setdefault("TLD_EXTRACT_CACHE", os.path.join(cache_dir, "tldextract"))
            os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_dir, "transformers"))
            os.environ.setdefault("TORCH_HOME", os.path.join(cache_dir, "torch"))
            os.environ.setdefault("HF_HOME", os.path.join(data_dir, "huggingface"))
            os.environ.setdefault("MATPLOTLIBCONFIGDIR", os.path.join(cache_dir, "matplotlib"))
            os.environ.setdefault("MODEL_CACHE_DIR", os.path.join(cache_dir, "models"))
            
            logger.info("Runtime cache configured")
            return {"cache_dir": cache_dir, "data_dir": data_dir}
        
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
            
            setup_runtime_cache_env()
            
            # Force load the correct openssl libs
            import ctypes
            crypto_path = os.path.join(app_dir, "_internal", "libcrypto.3.dylib")
            ssl_path = os.path.join(app_dir, "_internal", "libssl.3.dylib")
            ctypes.CDLL(crypto_path, mode=ctypes.RTLD_GLOBAL)
            ctypes.CDLL(ssl_path, mode=ctypes.RTLD_GLOBAL)
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__))
            logger.info("Launching Roboflow Inference (source)")
        
        logger.info("Initializing services")
        os.chdir(app_dir)
        
        logger.info("Configuring environment")
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
        
        logger.info("Starting server")
        from cpu_http import app
        import uvicorn
        
        # Run uvicorn with proper shutdown handling
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", "9001")),
            log_level="info",
            access_log=False,
        )
        server = uvicorn.Server(config)
        
        # Run the server
        server.run()
        
    except Exception as e:
        logger.exception("Error starting server: %s", e)
        if webview.windows:
            webview.windows[0].evaluate_js(f"setStatus('Error: {str(e)}')")

def start_window():
    """Start the webview window"""
    global server_thread
    
    # Setup GUI logging handler
    gui_handler = GuiLogHandler()
    gui_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(gui_handler)
    
    # Create window
    window = webview.create_window(
        'Roboflow Inference Console', 
        html=HTML, 
        width=880, 
        height=560, 
        js_api=Api()
    )
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Start readiness check in background
    def check_readiness():
        url = f"http://localhost:{os.environ.get('PORT','9001')}/readiness"
        for _ in range(120):
            try:
                with urllib.request.urlopen(url, timeout=1.5) as r:
                    if r.status == 200:
                        logger.info("Server ready")
                        window.evaluate_js("setStatus('Running')")
                        Api().open_dashboard()
                        break
            except Exception:
                time.sleep(0.5)
    
    threading.Thread(target=check_readiness, daemon=True).start()
    
    # Start webview (blocks until window is closed)
    webview.start()
    
    # Clean shutdown when window is closed
    logger.info("Window closed, shutting down...")
    shutdown_event.set()
    sys.exit(0)

if __name__ == '__main__':
    start_window()