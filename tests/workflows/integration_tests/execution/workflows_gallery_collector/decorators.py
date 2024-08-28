import functools
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class WorkflowGalleryEntry:
    category: str
    use_case_title: str
    use_case_description: str
    workflow_definition: dict


GALLERY_ENTRIES: List[WorkflowGalleryEntry] = []


def add_to_workflows_gallery(
    category: str,
    use_case_title: str,
    use_case_description: str,
    workflow_definition: dict,
):
    global GALLERY_ENTRIES
    gallery_entry = WorkflowGalleryEntry(
        category=category,
        use_case_title=use_case_title,
        use_case_description=use_case_description.strip(),
        workflow_definition=workflow_definition,
    )
    GALLERY_ENTRIES.append(gallery_entry)

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    return decorator
