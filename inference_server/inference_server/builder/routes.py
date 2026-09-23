import asyncio
import json
import logging
import os
import re
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Body, Depends, Header, HTTPException, status
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.status import HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from inference_server import configuration
from inference_server.builder import models
from inference_server.legacy.bridge import LegacyModelBridge
from inference_server.legacy.errors import with_legacy_errors
from inference_server.legacy.router import get_bridge

logger = logging.getLogger(__name__)


def _path_is_strict_descendant(path: str, parent: str) -> bool:
    resolved_path = os.path.realpath(path)
    resolved_parent = os.path.realpath(parent)
    try:
        return (
            resolved_path != resolved_parent
            and os.path.commonpath([resolved_parent, resolved_path]) == resolved_parent
        )
    except ValueError:
        return False


def _workflow_local_path_is_safe(path: Path) -> bool:
    """Reject child symlinks below the configured model-cache volume."""

    workflow_cache_root = Path(configuration.MODEL_CACHE_DIR, "workflow").absolute()
    candidate = path.absolute()
    if workflow_cache_root.is_symlink():
        return False
    try:
        relative_path = candidate.relative_to(workflow_cache_root)
    except ValueError:
        return False
    if not relative_path.parts:
        return False
    current_path = workflow_cache_root
    for path_part in relative_path.parts:
        current_path = current_path / path_part
        if current_path.is_symlink():
            return False
    return _path_is_strict_descendant(
        path=str(candidate),
        parent=str(workflow_cache_root),
    )


workflow_local_dir = Path(configuration.MODEL_CACHE_DIR) / "workflow" / "local"
if not _workflow_local_path_is_safe(workflow_local_dir):
    raise RuntimeError("Refusing to use an unsafe local Workflow cache directory")
workflow_local_dir.mkdir(parents=True, exist_ok=True)
if not _workflow_local_path_is_safe(workflow_local_dir):
    raise RuntimeError("Refusing to use an unsafe local Workflow cache directory")

router = APIRouter()

csrf_file = workflow_local_dir / ".csrf"
if not _workflow_local_path_is_safe(csrf_file):
    raise RuntimeError("Refusing to use an unsafe Workflow Builder CSRF file")


def _read_csrf_file() -> str:
    descriptor = os.open(
        csrf_file,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
    )
    with os.fdopen(descriptor, "r") as file_handle:
        return file_handle.read()


if csrf_file.exists():
    csrf = _read_csrf_file()
else:
    candidate_csrf = os.urandom(16).hex()
    try:
        descriptor = os.open(
            csrf_file,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        if not _workflow_local_path_is_safe(csrf_file):
            raise RuntimeError("Refusing to use an unsafe Workflow Builder CSRF file")
        csrf = _read_csrf_file()
    else:
        with os.fdopen(descriptor, "w") as file_handle:
            file_handle.write(candidate_csrf)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        csrf = candidate_csrf


def verify_csrf_token(x_csrf: str = Header(None)):
    if x_csrf != csrf:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token"
        )


@router.get(
    "",
    summary="Workflow Builder List",
    description="Loads the list of Workflows available for editing",
)
@with_legacy_errors
async def builder_browse():
    """Loads the main builder UI (editor.html)."""
    base_path = Path(__file__).parent
    file_path = base_path / "editor.html"
    content = file_path.read_text(encoding="utf-8")
    content = content.replace("{{BUILDER_ORIGIN}}", configuration.BUILDER_ORIGIN)
    content = content.replace("{{CSRF}}", csrf)

    return HTMLResponse(content)


@router.get("/", include_in_schema=False)
async def builder_redirect():
    """If user hits /build/ with trailing slash, redirect to /build"""
    return RedirectResponse(url="/build", status_code=302)


@router.get(
    "/edit/{workflow_id}",
    summary="Workflow Builder",
    description="Loads a specific workflow for editing",
)
@with_legacy_errors
async def builder_edit(workflow_id: str):
    """Loads a specific workflow for editing."""
    base_path = Path(__file__).parent
    file_path = base_path / "editor.html"
    content = file_path.read_text(encoding="utf-8")
    content = content.replace("{{BUILDER_ORIGIN}}", configuration.BUILDER_ORIGIN)
    content = content.replace("{{CSRF}}", csrf)

    return HTMLResponse(content)


@router.get("/api", dependencies=[Depends(verify_csrf_token)])
@with_legacy_errors
async def get_all_workflows():
    """Returns JSON info about all .json files in {MODEL_CACHE_DIR}/workflow/local."""
    data = {}
    for json_file in workflow_local_dir.glob("*.json"):
        if not _workflow_local_path_is_safe(json_file):
            logger.warning("Skipping unsafe local Workflow file: %s", json_file)
            continue
        stat_info = json_file.stat()
        try:
            with json_file.open("r", encoding="utf-8") as f:
                config_contents: Dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {json_file}: {e}")
            continue

        data[config_contents.get("id", json_file.stem)] = {
            "createTime": {"_seconds": int(stat_info.st_ctime)},
            "updateTime": {"_seconds": int(stat_info.st_mtime)},
            "config": config_contents,
        }

    return Response(
        content=json.dumps({"data": data}, indent=4),
        media_type="application/json",
        status_code=200,
    )


_RESERVED_WORKFLOW_IDS = {"models"}
_models_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_MODELS_CACHE_TTL = 30.0
_models_lock = asyncio.Lock()


@router.get("/api/models", dependencies=[Depends(verify_csrf_token)])
@with_legacy_errors
async def get_cached_models(bridge: LegacyModelBridge = Depends(get_bridge)):
    """Return all models available in the local cache."""
    global _models_cache

    async with _models_lock:
        now = time.time()
        if _models_cache is not None:
            cached_at, cached_result = _models_cache
            if now - cached_at < _MODELS_CACHE_TTL:
                return JSONResponse(content={"models": cached_result})

        listed_models = await models.list_models(bridge)

        _models_cache = (now, listed_models)

    return JSONResponse(content={"models": listed_models})


@router.get("/api/{workflow_id}", dependencies=[Depends(verify_csrf_token)])
@with_legacy_errors
async def get_workflow(workflow_id: str):
    """Return JSON for workflow_id.json, or 404 if missing."""
    if not re.match(r"^[\w\-]+$", workflow_id):
        return JSONResponse({"error": "invalid id"}, status_code=HTTP_400_BAD_REQUEST)
    if workflow_id in _RESERVED_WORKFLOW_IDS:
        return JSONResponse(
            {"error": f"'{workflow_id}' is a reserved identifier"},
            status_code=HTTP_400_BAD_REQUEST,
        )

    workflow_hash = sha256(workflow_id.encode()).hexdigest()
    file_path = workflow_local_dir / f"{workflow_hash}.json"
    if not _workflow_local_path_is_safe(file_path) or not file_path.exists():
        return JSONResponse({"error": "not found"}, status_code=HTTP_404_NOT_FOUND)

    stat_info = file_path.stat()
    try:
        with file_path.open("r", encoding="utf-8") as f:
            config_contents = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error reading JSON for {workflow_id} from '{file_path}': {e}")
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
@with_legacy_errors
async def create_or_overwrite_workflow(
    workflow_id: str, request_body: dict = Body(...)
):
    """Create or overwrite a workflow's JSON file on disk."""
    if not re.match(r"^[\w\-]+$", workflow_id):
        return JSONResponse({"error": "invalid id"}, status_code=HTTP_400_BAD_REQUEST)
    if workflow_id in _RESERVED_WORKFLOW_IDS:
        return JSONResponse(
            {"error": f"'{workflow_id}' is a reserved identifier"},
            status_code=HTTP_400_BAD_REQUEST,
        )

    if not _workflow_local_path_is_safe(workflow_local_dir):
        return JSONResponse({"error": "unsafe cache path"}, status_code=500)
    workflow_local_dir.mkdir(parents=True, exist_ok=True)
    if not _workflow_local_path_is_safe(workflow_local_dir):
        return JSONResponse({"error": "unsafe cache path"}, status_code=500)

    if request_body.get("id") and request_body.get("id") != workflow_id:
        old_id: str = request_body["id"]
        if not re.match(r"^[\w\-]+$", old_id):
            return JSONResponse(
                {"error": "invalid id"}, status_code=HTTP_400_BAD_REQUEST
            )

        old_workflow_hash = sha256(old_id.encode()).hexdigest()
        old_file_path = workflow_local_dir / f"{old_workflow_hash}.json"
        if old_file_path.exists():
            if not _workflow_local_path_is_safe(old_file_path):
                return JSONResponse({"error": "unsafe cache path"}, status_code=500)
            try:
                old_file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting {old_id} from {old_file_path}: {e}")
                return JSONResponse({"error": "unable to delete file"}, status_code=500)

    request_body["id"] = workflow_id

    workflow_hash = sha256(workflow_id.encode()).hexdigest()
    file_path = workflow_local_dir / f"{workflow_hash}.json"
    if not _workflow_local_path_is_safe(file_path):
        return JSONResponse({"error": "unsafe cache path"}, status_code=500)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=workflow_local_dir,
            prefix=".local-workflow.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary_path = f.name
            json.dump(request_body, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if not _workflow_local_path_is_safe(file_path):
            raise ValueError("unsafe cache path")
        os.replace(temporary_path, file_path)
        temporary_path = None
    except Exception as e:
        logger.error(f"Error writing JSON for {workflow_id} to {file_path}: {e}")
        return JSONResponse({"error": "unable to write file"}, status_code=500)
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass

    return JSONResponse(
        {"message": f"Workflow '{workflow_id}' created/updated successfully."},
        status_code=HTTP_201_CREATED,
    )


@router.delete("/api/{workflow_id}", dependencies=[Depends(verify_csrf_token)])
@with_legacy_errors
async def delete_workflow(workflow_id: str):
    """Delete a workflow's JSON file from disk."""
    if not re.match(r"^[\w\-]+$", workflow_id):
        return JSONResponse({"error": "invalid id"}, status_code=HTTP_400_BAD_REQUEST)
    if workflow_id in _RESERVED_WORKFLOW_IDS:
        return JSONResponse(
            {"error": f"'{workflow_id}' is a reserved identifier"},
            status_code=HTTP_400_BAD_REQUEST,
        )

    workflow_hash = sha256(workflow_id.encode()).hexdigest()
    file_path = workflow_local_dir / f"{workflow_hash}.json"
    if not _workflow_local_path_is_safe(file_path) or not file_path.exists():
        return JSONResponse({"error": "not found"}, status_code=HTTP_404_NOT_FOUND)

    try:
        file_path.unlink()
    except Exception as e:
        logger.error(f"Error deleting {workflow_id} from {file_path}: {e}")
        return JSONResponse({"error": "unable to delete file"}, status_code=500)

    return JSONResponse(
        {"message": f"Workflow '{workflow_id}' deleted successfully."}, status_code=200
    )


@router.get("/{workflow_id}", include_in_schema=False)
@with_legacy_errors
async def builder_maybe_redirect(workflow_id: str):
    """Redirect to the editor when the workflow exists, otherwise to /build."""
    if not re.match(r"^[\w\-]+$", workflow_id):
        return RedirectResponse(url="/build", status_code=302)

    workflow_hash = sha256(workflow_id.encode()).hexdigest()
    file_path = workflow_local_dir / f"{workflow_hash}.json"
    if _workflow_local_path_is_safe(file_path) and file_path.exists():
        return RedirectResponse(url=f"/build/edit/{workflow_id}", status_code=302)
    else:
        return RedirectResponse(url="/build", status_code=302)
