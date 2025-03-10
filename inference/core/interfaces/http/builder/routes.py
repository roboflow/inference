import json
import logging
import os
import re
from pathlib import Path

from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.status import HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from inference.core.env import BUILDER_ORIGIN, MODEL_CACHE_DIR
from inference.core.interfaces.http.http_api import with_route_exceptions

logger = logging.getLogger(__name__)

workflow_local_dir = Path(MODEL_CACHE_DIR) / "workflow" / "local"
workflow_local_dir.mkdir(parents=True, exist_ok=True)

router = APIRouter()

# ----------------------------------------------------------------
# Generate or read the "csrf" token from disk once per server run
# ----------------------------------------------------------------
csrf_file = workflow_local_dir / ".csrf"
if csrf_file.exists():
    csrf = csrf_file.read_text()
else:
    csrf = os.urandom(16).hex()
    csrf_file.write_text(csrf)


# ----------------------------------------------------------------
# Dependency to verify the X-CSRF header on any protected route
# ----------------------------------------------------------------
def verify_csrf_token(x_csrf: str = Header(None)):
    if x_csrf != csrf:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token"
        )


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
    Loads the main builder UI (editor.html).
    Injects the CSRF token and BUILDER_ORIGIN
    so the client can parse them on page load.
    """
    base_path = Path(__file__).parent
    file_path = base_path / "editor.html"
    content = file_path.read_text(encoding="utf-8")
    content = content.replace("{{BUILDER_ORIGIN}}", BUILDER_ORIGIN)
    content = content.replace("{{CSRF}}", csrf)

    return HTMLResponse(content)


@router.get("/", include_in_schema=False)
async def builder_redirect():
    """
    If user hits /build/ with trailing slash, redirect to /build
    """
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
    """
    base_path = Path(__file__).parent
    file_path = base_path / "editor.html"
    content = file_path.read_text(encoding="utf-8")
    content = content.replace("{{BUILDER_ORIGIN}}", BUILDER_ORIGIN)
    content = content.replace("{{CSRF}}", csrf)

    return HTMLResponse(content)


# ----------------------
# BACKEND JSON API ROUTES
# ----------------------


@router.get("/api", dependencies=[Depends(verify_csrf_token)])
@with_route_exceptions
async def get_all_workflows():
    """
    Returns JSON info about all .json files in {MODEL_CACHE_DIR}/workflow/local.
    Protected by CSRF token check.
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
        content=json.dumps({"data": data}, indent=4),
        media_type="application/json",
        status_code=200,
    )


@router.get("/api/{workflow_id}", dependencies=[Depends(verify_csrf_token)])
@with_route_exceptions
async def get_workflow(workflow_id: str):
    """
    Return JSON for workflow_id.json, or 404 if missing.
    """
    if not re.match(r"^[\w\-]+$", workflow_id):
        return JSONResponse({"error": "invalid id"}, status_code=HTTP_400_BAD_REQUEST)

    file_path = workflow_local_dir / f"{workflow_id}.json"
    if not file_path.exists():
        return JSONResponse({"error": "not found"}, status_code=HTTP_404_NOT_FOUND)

    stat_info = file_path.stat()
    try:
        with file_path.open("r", encoding="utf-8") as f:
            config_contents = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error reading JSON from {file_path}: {e}")
        return JSONResponse({"error": "invalid JSON"}, status_code=500)

    return Response(
        content=json.dumps(
            {
                "data": {
                    "createTime": int(stat_info.st_ctime),
                    "updateTime": int(stat_info.st_mtime),
                    "config": config_contents,
                }
            },
            indent=4,
        ),
        media_type="application/json",
        status_code=200,
    )


@router.post("/api/{workflow_id}", dependencies=[Depends(verify_csrf_token)])
@with_route_exceptions
async def create_or_overwrite_workflow(
    workflow_id: str, request_body: dict = Body(...)
):
    """
    Create or overwrite a workflow's JSON file on disk.
    Protected by CSRF token check.
    """
    if not re.match(r"^[\w\-]+$", workflow_id):
        return JSONResponse({"error": "invalid id"}, status_code=HTTP_400_BAD_REQUEST)

    file_path = workflow_local_dir / f"{workflow_id}.json"
    workflow_local_dir.mkdir(parents=True, exist_ok=True)

    # If the body claims a different ID, treat that as a "rename".
    if request_body.get("id") and request_body.get("id") != workflow_id:
        old_id = request_body["id"]
        if not re.match(r"^[\w\-]+$", old_id):
            return JSONResponse(
                {"error": "invalid id"}, status_code=HTTP_400_BAD_REQUEST
            )

        old_file_path = workflow_local_dir / f"{old_id}.json"
        if old_file_path.exists():
            try:
                old_file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting {old_file_path}: {e}")
                return JSONResponse({"error": "unable to delete file"}, status_code=500)

        request_body["id"] = workflow_id

    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(request_body, f, indent=2)
    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}")
        return JSONResponse({"error": "unable to write file"}, status_code=500)

    return JSONResponse(
        {"message": f"Workflow '{workflow_id}' created/updated successfully."},
        status_code=HTTP_201_CREATED,
    )


@router.delete("/api/{workflow_id}", dependencies=[Depends(verify_csrf_token)])
@with_route_exceptions
async def delete_workflow(workflow_id: str):
    """
    Delete a workflow's JSON file from disk.
    Protected by CSRF token check.
    """
    if not re.match(r"^[\w\-]+$", workflow_id):
        return JSONResponse({"error": "invalid id"}, status_code=HTTP_400_BAD_REQUEST)

    file_path = workflow_local_dir / f"{workflow_id}.json"
    if not file_path.exists():
        return JSONResponse({"error": "not found"}, status_code=HTTP_404_NOT_FOUND)

    try:
        file_path.unlink()
    except Exception as e:
        logger.error(f"Error deleting {file_path}: {e}")
        return JSONResponse({"error": "unable to delete file"}, status_code=500)

    return JSONResponse(
        {"message": f"Workflow '{workflow_id}' deleted successfully."}, status_code=200
    )


# ------------------------
# FALLBACK REDIRECT HELPER
# ------------------------


@router.get("/{workflow_id}", include_in_schema=False)
@with_route_exceptions
async def builder_maybe_redirect(workflow_id: str):
    """
    If the workflow_id.json file exists, redirect to /build/edit/{workflow_id}.
    Otherwise, redirect back to /build.
    """
    if not re.match(r"^[\w\-]+$", workflow_id):
        return RedirectResponse(url="/build", status_code=302)

    file_path = workflow_local_dir / f"{workflow_id}.json"
    if file_path.exists():
        return RedirectResponse(url=f"/build/edit/{workflow_id}", status_code=302)
    else:
        return RedirectResponse(url="/build", status_code=302)
