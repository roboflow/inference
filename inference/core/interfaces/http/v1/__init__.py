"""
V1 HTTP API for Roboflow Inference Server.

This module provides the new v1 API with:
- Header-based authentication (no API keys in request bodies)
- Multipart form data for efficient image uploads (no base64 JSON bottleneck)
- Clean RESTful interfaces with explicit versioning
"""

from inference.core.interfaces.http.v1.router import v1_router

__all__ = ["v1_router"]
