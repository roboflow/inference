"""On-device WebGPU playground: an opt-in page for phones, tablets and laptops.

Mounted by ``inference/core/interfaces/http/http_api.py`` under ``/webgpu`` only
when ``ENABLE_WEBGPU_PLAYGROUND`` is truthy (default: False). The page measures
the visitor's GPU with WebGPU compute shaders and runs a small language model in
the browser through WebLLM, so prompts and outputs stay on the device: the
router only serves one static HTML file.
"""

from pathlib import Path

from fastapi import APIRouter
from starlette.responses import HTMLResponse

PLAYGROUND_PAGE = Path(__file__).parent / "playground.html"

router = APIRouter()


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
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
