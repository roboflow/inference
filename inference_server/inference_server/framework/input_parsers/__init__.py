"""Body / URL extraction shared across model handlers.

Public entry: `extract_images_and_params(request)` dispatches by Content-Type.
URL-based input: `fetch_image_from_url(url)`.
"""

from inference_server.framework.input_parsers.dispatch import (
    extract_images_and_params,
)
from inference_server.framework.input_parsers.url_fetch import (
    fetch_image_from_url,
)

__all__ = ["extract_images_and_params", "fetch_image_from_url"]
