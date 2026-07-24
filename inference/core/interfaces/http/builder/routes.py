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

from inference.core.cache.air_gapped import (
    get_cached_foundation_models,
    get_configured_model_cache_roots,
    get_task_type_to_block_mapping,
    scan_cached_models,
)
from inference.core.env import BUILDER_ORIGIN, MODEL_CACHE_DIR
from inference.core.interfaces.http.error_handlers import with_route_exceptions_async
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)

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
        # Different drives on Windows cannot share a common path.
        return False


def _workflow_local_path_is_safe(path: Path) -> bool:
    """Reject child symlinks below the configured model-cache volume."""

    workflow_cache_root = Path(MODEL_CACHE_DIR, "workflow").absolute()
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


def _collect_unambiguous_user_models(
    user_models: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """De-duplicate identical entries and omit IDs with conflicting metadata."""

    models_by_id: Dict[str, Dict[str, Any]] = {}
    conflicting_model_ids = set()
    for model in user_models:
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            logger.warning("Skipping cached model metadata without a valid model_id")
            continue
        if model_id in conflicting_model_ids:
            continue
        existing_model = models_by_id.get(model_id)
        if existing_model is None:
            models_by_id[model_id] = model
            continue
        if existing_model == model:
            continue
        models_by_id.pop(model_id)
        conflicting_model_ids.add(model_id)
        logger.warning(
            "Excluding cached model %s because configured cache roots contain "
            "conflicting metadata for that model ID",
            model_id,
        )
    return models_by_id


workflow_local_dir = Path(MODEL_CACHE_DIR) / "workflow" / "local"
if not _workflow_local_path_is_safe(workflow_local_dir):
    raise RuntimeError("Refusing to use an unsafe local Workflow cache directory")
workflow_local_dir.mkdir(parents=True, exist_ok=True)
if not _workflow_local_path_is_safe(workflow_local_dir):
    raise RuntimeError("Refusing to use an unsafe local Workflow cache directory")

router = APIRouter()

# ----------------------------------------------------------------
# Generate or read the "csrf" token from disk once per server run
# ----------------------------------------------------------------
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
@with_route_exceptions_async
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
@with_route_exceptions_async
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
@with_route_exceptions_async
async def get_all_workflows():
    """
    Returns JSON info about all .json files in {MODEL_CACHE_DIR}/workflow/local.
    Protected by CSRF token check.
    """
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


# ----------------------------------------------------------------
# Models cache for /build/api/models (TTL-based)
# IMPORTANT: This route MUST be defined BEFORE /api/{workflow_id}
# otherwise FastAPI will match "models" as a workflow_id.
# ----------------------------------------------------------------
# IDs that are claimed by explicit sub-routes and therefore cannot
# be used as workflow identifiers.
_RESERVED_WORKFLOW_IDS = {"models"}
_models_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_MODELS_CACHE_TTL = 30.0  # seconds
_models_lock = asyncio.Lock()


@router.get("/api/models", dependencies=[Depends(verify_csrf_token)])
@with_route_exceptions_async
async def get_cached_models():
    """Return all models available in the local cache.

    Combines user-trained models discovered via ``model_type.json`` markers
    with foundation-model blocks whose weights are fully cached.
    Results are cached for 30 seconds to avoid repeated filesystem scans.
    """
    global _models_cache  # noqa: PLW0603

    async with _models_lock:
        now = time.time()
        if _models_cache is not None:
            cached_at, cached_result = _models_cache
            if now - cached_at < _MODELS_CACHE_TTL:
                return JSONResponse(content={"models": cached_result})

        # Inline import: inference.models.aliases transitively imports the
        # inference_models package which may not be installed when
        # ENABLE_BUILDER=False.  Keeping the import lazy avoids breaking
        # the server for non-builder users.
        from inference.models.aliases import REGISTERED_ALIASES

        # Build reverse alias map: canonical_id → [alias1, alias2, ...]
        reverse_aliases: Dict[str, List[str]] = {}
        for alias, canonical in REGISTERED_ALIASES.items():
            reverse_aliases.setdefault(canonical, []).append(alias)

        # Load blocks once and pass to both helpers to avoid double-loading.
        try:
            blocks = load_workflow_blocks()
        except Exception:
            logger.warning(
                "Failed to load workflow blocks — foundation model data will "
                "be unavailable. This may indicate a broken build or missing "
                "dependencies.",
                exc_info=True,
            )
            blocks = []

        # Scan the filesystem for cached models.
        user_models = []
        cache_roots = get_configured_model_cache_roots()
        for cache_root in cache_roots:
            resolved_cache_root = os.path.realpath(cache_root)
            nested_cache_roots = [
                other_root
                for other_root in cache_roots
                if other_root != cache_root
                and _path_is_strict_descendant(
                    path=os.path.realpath(other_root),
                    parent=resolved_cache_root,
                )
            ]
            user_models.extend(
                scan_cached_models(
                    cache_root,
                    excluded_cache_roots=nested_cache_roots,
                )
            )
        foundation_models = get_cached_foundation_models(blocks=blocks)

        # De-duplicate by model_id (foundation models take precedence).
        seen = _collect_unambiguous_user_models(user_models=user_models)
        for m in foundation_models:
            seen[m["model_id"]] = m

        # Enrich each model with compatible block types and aliases.
        task_to_blocks = get_task_type_to_block_mapping(blocks=blocks)
        models = []
        for m in seen.values():
            entry = dict(m)
            # For foundation models, use block_type for compatible_block_types
            # since they have empty task_type.
            block_type = entry.get("block_type")
            if block_type:
                entry.setdefault("compatible_block_types", [block_type])
            else:
                entry.setdefault(
                    "compatible_block_types",
                    task_to_blocks.get(m.get("task_type", ""), []),
                )
            # Add known aliases for this model
            model_id = m.get("model_id", "")
            aliases = reverse_aliases.get(model_id, [])
            entry["aliases"] = aliases
            # Use the shortest alias as display name if available
            if aliases and (entry.get("name") == model_id or not entry.get("name")):
                entry["name"] = min(aliases, key=len)
            # Remove internal-only keys.
            entry.pop("block_type", None)
            models.append(entry)

        _models_cache = (now, models)

    return JSONResponse(content={"models": models})


@router.get("/api/{workflow_id}", dependencies=[Depends(verify_csrf_token)])
@with_route_exceptions_async
async def get_workflow(workflow_id: str):
    """
    Return JSON for workflow_id.json, or 404 if missing.
    IDs in ``_RESERVED_WORKFLOW_IDS`` are rejected to avoid shadowing
    explicit sub-routes like ``/api/models``.
    """
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
@with_route_exceptions_async
async def create_or_overwrite_workflow(
    workflow_id: str, request_body: dict = Body(...)
):
    """
    Create or overwrite a workflow's JSON file on disk.
    Protected by CSRF token check.
    """
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

    # If the body claims a different ID, treat that as a "rename".
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
@with_route_exceptions_async
async def delete_workflow(workflow_id: str):
    """
    Delete a workflow's JSON file from disk.
    Protected by CSRF token check.
    """
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


# ------------------------
# FALLBACK REDIRECT HELPER
# ------------------------


@router.get("/{workflow_id}", include_in_schema=False)
@with_route_exceptions_async
async def builder_maybe_redirect(workflow_id: str):
    """
    If the workflow_id.json file exists, redirect to /build/edit/{workflow_id}.
    Otherwise, redirect back to /build.
    """
    if not re.match(r"^[\w\-]+$", workflow_id):
        return RedirectResponse(url="/build", status_code=302)

    workflow_hash = sha256(workflow_id.encode()).hexdigest()
    file_path = workflow_local_dir / f"{workflow_hash}.json"
    if _workflow_local_path_is_safe(file_path) and file_path.exists():
        return RedirectResponse(url=f"/build/edit/{workflow_id}", status_code=302)
    else:
        return RedirectResponse(url="/build", status_code=302)
