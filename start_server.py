#!/usr/bin/env python3
"""
Startup script for Inference Server with optional SSL support

This script handles starting both HTTP and HTTPS servers when SSL is enabled.
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path
from multiprocessing import Process
from functools import partial

from inference.core.env import (
    ENABLE_SSL,
    SSL_PORT,
    SSL_CERTIFICATE,
    HOST,
    PORT,
    NUM_WORKERS,
    ENABLE_STREAM_API,
    STREAM_API_PRELOADED_PROCESSES,
)
from inference.core import logger


def start_stream_manager():
    """Start the stream manager if enabled"""
    if ENABLE_STREAM_API:
        from inference.core.interfaces.stream_manager.manager_app.app import start
        
        logger.info("Starting Stream Manager...")
        stream_manager_process = Process(
            target=partial(start, expected_warmed_up_pipelines=STREAM_API_PRELOADED_PROCESSES),
        )
        stream_manager_process.start()
        return stream_manager_process
    return None


def get_ssl_config():
    """Get SSL certificate configuration if SSL is enabled"""
    if not ENABLE_SSL:
        return None
        
    from inference.core.interfaces.http.ssl import SSLCertificateManager
    
    cert_manager = SSLCertificateManager()
    cert_path, key_path = cert_manager.get_certificate_paths(SSL_CERTIFICATE)
    
    return cert_path, key_path


def start_http_server():
    """Start the HTTP server"""
    logger.info(f"Starting HTTP server on {HOST}:{PORT}")
    
    # Set environment to prevent duplicate stream manager
    env = os.environ.copy()
    env['STREAM_MANAGER_STARTED_BY_PARENT'] = 'true'
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "cpu_http:app",
        "--workers", str(NUM_WORKERS),
        "--host", HOST,
        "--port", str(PORT)
    ]
    subprocess.run(cmd, env=env)


def start_https_server():
    """Start the HTTPS server"""
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
    
    # Set environment to prevent duplicate stream manager
    env = os.environ.copy()
    env['STREAM_MANAGER_STARTED_BY_PARENT'] = 'true'
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "cpu_http:app",
        "--workers", "1",  # Single worker for SSL to simplify cert management
        "--host", HOST,
        "--port", str(ssl_port),
        "--ssl-keyfile", key_path,
        "--ssl-certfile", cert_path,
        "--log-level", "warning"
    ]
    subprocess.run(cmd, env=env)


def certificate_refresh_loop():
    """Periodically check for certificate updates"""
    if not ENABLE_SSL or SSL_CERTIFICATE != "INDIRECT":
        return
        
    from inference.core.interfaces.http.ssl import SSLCertificateManager
    cert_manager = SSLCertificateManager()
    
    while True:
        time.sleep(3600)  # Check every hour
        try:
            # This will trigger a refresh if needed
            cert_manager.get_certificate_paths(SSL_CERTIFICATE)
        except Exception as e:
            logger.error(f"Certificate refresh check failed: {e}")


def main():
    """Main entry point"""
    # Start stream manager first (only once)
    stream_manager_process = start_stream_manager()
    
    # Give stream manager time to start
    if stream_manager_process:
        time.sleep(2)
    
    if ENABLE_SSL:
        # Start certificate refresh thread
        refresh_thread = threading.Thread(target=certificate_refresh_loop, daemon=True)
        refresh_thread.start()
        
        # Start HTTPS server in a thread
        https_thread = threading.Thread(target=start_https_server)
        https_thread.start()
        
        # Small delay to ensure HTTPS starts
        time.sleep(2)
    
    # Start HTTP server in main thread
    start_http_server()


if __name__ == "__main__":
    main()
