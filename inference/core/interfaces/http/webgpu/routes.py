"""On-device WebGPU playground: an opt-in page for phones, tablets and laptops.

Mounted by ``inference/core/interfaces/http/http_api.py`` under ``/webgpu`` only
when ``ENABLE_WEBGPU_PLAYGROUND`` is truthy (default: False). The page measures
the visitor's GPU with WebGPU compute shaders and runs a small language model in
the browser through WebLLM, so prompts and outputs stay on the device: the
server only serves one static HTML file.
"""

from pathlib import Path

from fastapi import FastAPI
from starlette.responses import HTMLResponse

PLAYGROUND_PAGE = Path(__file__).parent / "playground.html"


async def webgpu_playground() -> HTMLResponse:
    """Serve the on-device WebGPU playground page.

    Returns:
        HTMLResponse: the self-contained playground page. It loads WebLLM from
        jsDelivr and model weights from Hugging Face in the browser, only when
        the visitor starts a model.
    """
    return HTMLResponse(
        PLAYGROUND_PAGE.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache"},
    )


def mount_webgpu_playground(app: FastAPI, prefix: str = "/webgpu") -> None:
    """Register the playground page at ``prefix`` and ``prefix + "/"``.

    The routes go on the app directly rather than through ``include_router``:
    FastAPI 0.141 wraps an included router in an object without ``.path``, and
    prometheus-fastapi-instrumentator 6.0.0 reads ``route.path`` on every route
    while it names a request, so a page behind ``include_router`` answers 500
    while metrics are on (the default).

    Args:
        app: the inference server's FastAPI app.
        prefix: the URL path of the page.
    """
    for path in (prefix, f"{prefix}/"):
        app.add_api_route(
            path,
            webgpu_playground,
            methods=["GET"],
            include_in_schema=False,
            tags=["webgpu"],
        )
