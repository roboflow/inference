import os
import re
import json
import logging
from pathlib import Path
from fastapi import APIRouter, Body, HTTPException
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response, HTMLResponse
from starlette.status import HTTP_201_CREATED, HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST
from inference.core.interfaces.http.http_api import with_route_exceptions

from inference.core.env import (
    MODEL_CACHE_DIR,
    BUILDER_ORIGIN
)

logger = logging.getLogger(__name__)

workflow_local_dir = Path(MODEL_CACHE_DIR) / "workflow" / "local"
workflow_local_dir.mkdir(parents=True, exist_ok=True)

router = APIRouter()

# ---------------------
# FRONTEND HTML ROUTES
# ---------------------

@router.get(
    "",
    summary="Workflow Builder List",
    description="Loads the list of Workflows available for editing",
)
@with_route_exceptions
async def builder_browse():
    """
    Loads the list of Workflows available for editing.

    Returns:
        FileResponse: The HTML file containing the list of workflows.
    """
    base_path = Path(__file__).parent
    file_path = base_path / "editor.html"
    content = file_path.read_text(encoding="utf-8")
    content = content.replace("{{BUILDER_ORIGIN}}", BUILDER_ORIGIN)

    return HTMLResponse(content)

@router.get("/", include_in_schema=False)
async def builder_redirect():
    return RedirectResponse(url="/build", status_code=302)

@router.get(
    "/edit/{workflow_id}",
    summary="Workflow Builder",
    description="Loads a specific workflow for editing",
)
@with_route_exceptions
async def builder_edit(workflow_id: str):
    """
    Loads a specific workflow for editing.

    Args:
        workflow_id (str): The ID of the workflow to be edited.

    Returns:
        FileResponse: The HTML file containing the workflow editor.
    """
    base_path = Path(__file__).parent
    file_path = base_path / "editor.html"
    content = file_path.read_text(encoding="utf-8")
    content = content.replace("{{BUILDER_ORIGIN}}", BUILDER_ORIGIN)

    return HTMLResponse(content)

# ----------------------
# BACKEND JSON API ROUTES
# ----------------------

@router.get("/api")
@with_route_exceptions
async def get_all_workflows():
    """
    Returns JSON info about all .json files in {MODEL_CACHE_DIR}/workflow/local.
    """
    data = {}
    for json_file in workflow_local_dir.glob("*.json"):
        stat_info = json_file.stat()
        try:
            with json_file.open("r", encoding="utf-8") as f:
                config_contents = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {json_file}: {e}")
            continue

        data[json_file.stem] = {
            "createTime": int(stat_info.st_ctime),
            "updateTime": int(stat_info.st_mtime),
            "config": config_contents,
        }

    return Response(
        content=json.dumps(
            {
                "data": data
            },
            indent=4
        ),
        media_type="application/json",
        status_code=200
    )

@router.get("/api/{workflow_id}")
@with_route_exceptions
async def get_workflow(workflow_id: str):
    """
    Return JSON for workflow_id.json, or { "error": "not found" } with 404 if missing.
    """
    if not re.match(r'^[\w\-]+$', workflow_id):
        return JSONResponse({"error": "invalid id"}, status_code=HTTP_400_BAD_REQUEST)

    file_path = workflow_local_dir / f"{workflow_id}.json"
    if not file_path.exists():
        # Return the structure you specifically asked for
        return JSONResponse({"error": "not found"}, status_code=HTTP_404_NOT_FOUND)

    stat_info = file_path.stat()
    try:
        with file_path.open("r", encoding="utf-8") as f:
            config_contents = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error reading JSON from {file_path}: {e}")
        # You can also do a 400 or 500 if you prefer:
        return JSONResponse({"error": "invalid JSON"}, status_code=500)

    return Response(
        content=json.dumps(
            {
                "data": {
                    "createTime": int(stat_info.st_ctime),
                    "updateTime": int(stat_info.st_mtime),
                    "config": config_contents
                }
            },
            indent=4
        ),
        media_type="application/json",
        status_code=200
    )

@router.post("/api/{workflow_id}")
@with_route_exceptions
async def create_or_overwrite_workflow(workflow_id: str, request_body: dict = Body(...)):
    if not re.match(r'^[\w\-]+$', workflow_id):
        return JSONResponse({"error": "invalid id"}, status_code=HTTP_400_BAD_REQUEST)

    file_path = workflow_local_dir / f"{workflow_id}.json"
    workflow_local_dir.mkdir(parents=True, exist_ok=True)

    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(request_body, f, indent=2)
    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}")
        return JSONResponse({"error": "unable to write file"}, status_code=500)

    return JSONResponse({"message": f"Workflow '{workflow_id}' created/updated successfully."}, status_code=HTTP_201_CREATED)

# ------------------------
# FALLBACK REDIRECT HELPER
# ------------------------

@router.get("/{workflow_id}")
@with_route_exceptions
async def builder_maybe_redirect(workflow_id: str):
    """
    If the workflow_id.json file exists, redirect to /build/edit/{workflow_id}.
    Otherwise, redirect back to /build.
    """
    # Sanitize workflow_id to prevent path traversal
    if not re.match(r'^[\w\-]+$', workflow_id):
        # If it's invalid, just redirect home (or raise 400)
        return RedirectResponse(url="/build", status_code=302)

    file_path = workflow_local_dir / f"{workflow_id}.json"
    if file_path.exists():
        return RedirectResponse(url=f"/build/edit/{workflow_id}", status_code=302)
    else:
        return RedirectResponse(url="/build", status_code=302)