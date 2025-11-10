"""
Header-based authentication for v1 API.

This module provides authentication that extracts API keys from headers or query
parameters only, never from request bodies. This allows auth to happen before body
parsing and is compatible with standard reverse proxy authentication.
"""

from typing import Optional
import time

from fastapi import Header, Query, HTTPException, Request

from inference.core.exceptions import (
    RoboflowAPINotAuthorizedError,
    WorkspaceLoadError,
)
from inference.core.roboflow_api import get_roboflow_workspace_async
from inference.core import logger

# Cache for validated API keys (key -> expiration timestamp)
# Shared across all v1 endpoints
_api_key_cache: dict = {}


async def get_api_key_from_header_or_query(
    authorization: Optional[str] = Header(None, description="Authorization header with Bearer token"),
    x_roboflow_api_key: Optional[str] = Header(None, description="Roboflow API key header"),
    api_key: Optional[str] = Query(None, description="API key as query parameter (fallback)"),
) -> Optional[str]:
    """
    Extract API key from headers or query parameters.

    Supports multiple authentication methods (in order of precedence):
    1. Authorization header: "Bearer <api_key>"
    2. X-Roboflow-Api-Key header: "<api_key>"
    3. Query parameter: ?api_key=<api_key>

    Args:
        authorization: Authorization header value
        x_roboflow_api_key: Custom Roboflow API key header
        api_key: API key from query parameter

    Returns:
        The extracted API key, or None if not found
    """
    # Try Authorization header (Bearer token)
    if authorization:
        if authorization.startswith("Bearer "):
            return authorization[7:]  # Remove "Bearer " prefix
        elif authorization.startswith("bearer "):
            return authorization[7:]  # Handle lowercase
        else:
            # If Authorization header is present but not Bearer format, use it as-is
            return authorization

    # Try custom header
    if x_roboflow_api_key:
        return x_roboflow_api_key

    # Fall back to query parameter
    if api_key:
        return api_key

    return None


async def validate_api_key(
    api_key: Optional[str],
    required: bool = True,
    cache_ttl_seconds: int = 3600,
) -> Optional[str]:
    """
    Validate an API key against Roboflow API.

    Uses caching to avoid repeated API calls for the same key.

    Args:
        api_key: The API key to validate
        required: If True, raises HTTPException when key is missing or invalid
        cache_ttl_seconds: How long to cache valid API keys (default 1 hour)

    Returns:
        The validated API key

    Raises:
        HTTPException: If key is required but missing or invalid
    """
    if api_key is None:
        if required:
            raise HTTPException(
                status_code=401,
                detail="API key is required. Provide it via Authorization header, X-Roboflow-Api-Key header, or api_key query parameter.",
            )
        return None

    # Check cache
    current_time = time.time()
    cached_expiration = _api_key_cache.get(api_key, 0)

    if cached_expiration > current_time:
        # Cache hit - key is valid
        logger.debug(f"API key cache hit (expires in {int(cached_expiration - current_time)}s)")
        return api_key

    # Cache miss or expired - validate against Roboflow API
    try:
        logger.debug("Validating API key against Roboflow API")
        await get_roboflow_workspace_async(api_key=api_key)

        # Cache the valid key
        _api_key_cache[api_key] = current_time + cache_ttl_seconds
        logger.debug(f"API key validated and cached for {cache_ttl_seconds}s")

        return api_key

    except (RoboflowAPINotAuthorizedError, WorkspaceLoadError) as e:
        logger.warning(f"API key validation failed: {e}")
        if required:
            raise HTTPException(
                status_code=401,
                detail="Invalid or unauthorized API key",
            )
        return None


async def get_validated_api_key(
    authorization: Optional[str] = Header(None),
    x_roboflow_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Query(None),
) -> str:
    """
    FastAPI dependency for extracting and validating API keys.

    This is the main dependency to use in v1 endpoints that require authentication.
    It extracts the key from headers/query params and validates it.

    Returns:
        The validated API key

    Raises:
        HTTPException: If no valid API key is provided

    Example:
        @router.post("/v1/object-detection/{model_id}")
        async def detect_objects(
            model_id: str,
            api_key: str = Depends(get_validated_api_key),
        ):
            # api_key is guaranteed to be valid here
            ...
    """
    extracted_key = await get_api_key_from_header_or_query(
        authorization=authorization,
        x_roboflow_api_key=x_roboflow_api_key,
        api_key=api_key,
    )

    return await validate_api_key(extracted_key, required=True)


async def get_optional_api_key(
    authorization: Optional[str] = Header(None),
    x_roboflow_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Query(None),
) -> Optional[str]:
    """
    FastAPI dependency for extracting and validating optional API keys.

    Similar to get_validated_api_key but doesn't raise an error if no key is provided.
    Use for endpoints where authentication is optional.

    Returns:
        The validated API key, or None if not provided
    """
    extracted_key = await get_api_key_from_header_or_query(
        authorization=authorization,
        x_roboflow_api_key=x_roboflow_api_key,
        api_key=api_key,
    )

    return await validate_api_key(extracted_key, required=False)
