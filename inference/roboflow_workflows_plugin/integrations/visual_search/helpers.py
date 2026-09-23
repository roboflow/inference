from typing import Any, Dict, List, Optional

from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def format_visual_search_candidate(
    candidate: Dict[str, Any],
    extra_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    filename = candidate.get("filename") or candidate.get("name")
    formatted_candidate = {
        "image_id": candidate.get("id"),
        "image_url": candidate.get("url"),
        "name": candidate.get("name") or filename,
        "filename": filename,
        "metadata": candidate.get("user_metadata") or {},
        "tags": candidate.get("tags") or [],
        "width": candidate.get("width"),
        "height": candidate.get("height"),
        "aspect_ratio": candidate.get("aspectRatio"),
    }
    for field in extra_fields or []:
        formatted_candidate[field] = candidate.get(field)
    return formatted_candidate


def build_visual_search_candidate_image(
    candidate: Dict[str, Any],
    fallback_parent_id: str,
) -> Optional[WorkflowImageData]:
    image_url = candidate.get("image_url")
    if not image_url:
        return None
    parent_id = (
        candidate.get("image_id") or candidate.get("filename") or fallback_parent_id
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=str(parent_id)),
        image_reference=image_url,
    )
