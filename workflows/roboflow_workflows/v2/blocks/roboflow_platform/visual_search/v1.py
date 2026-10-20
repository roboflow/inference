"""Find the images of a project that look most like the query image.

Canonical type: ``roboflow_core/visual_search@v1``.
"""

from typing import Annotated, Any, Dict, List, Optional, Union
from pydantic import Field
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.ports import (
    OFFLINE_VISUAL_SEARCH_PORT,
    VisualSearchPort,
    api_key_for_dispatch,
)
from roboflow_workflows.v2.blocks.roboflow_platform._shared.visual_search import (
    _candidate_image,
    _format_candidate,
    _query_base64,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)


class VisualSearchBlock(Block):
    """Find the images of a project that look most like the query image.

    Resources:
        visual_search_port: Host adapter for the search; the default refuses
            with ``PlatformUnavailableError``.
        api_key: Roboflow API key, a ``str`` or a ``Secret``. Only the port
            reveals it.
    """

    type = "roboflow_core/visual_search@v1"
    metadata = {
        "name": "Roboflow Visual Search",
        "port_of": "roboflow_core/visual_search@v1",
        "section": "advanced",
        "requires_rf_key": True,
    }
    outputs = {
        "candidate_found": Output(
            BOOLEAN_KIND, description="Whether the search returned any candidate."
        ),
        "best_candidate": Output(
            DICTIONARY_KIND, description="Nearest candidate; {} when none."
        ),
        "candidates": Output(
            LIST_OF_VALUES_KIND, description="Candidates, nearest first; [] when none."
        ),
        "best_candidate_image": Output(
            IMAGE_KIND,
            description=(
                "Nearest candidate's image, downloaded when demanded; None without "
                "a candidate or image_url."
            ),
        ),
        "error_status": Output(
            BOOLEAN_KIND, description="True when the search or its result failed."
        ),
        "message": Output(STRING_KIND, description="Human-readable outcome."),
        "best_candidate_metadata": Output(
            DICTIONARY_KIND, description="user_metadata of the nearest candidate."
        ),
        "best_candidate_tags": Output(
            LIST_OF_VALUES_KIND, description="Tags of the nearest candidate."
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Query image.",
            examples=["$inputs.image", "$steps.crop.crops"],
        )
        target_project: str | Ref(STRING_KIND) = Field(
            description="Roboflow project URL slug to search.",
            examples=["reference-images", "$inputs.target_project"],
        )
        top_k: Annotated[int, Field(ge=1, le=50)] | Ref(INTEGER_KIND) = Field(
            default=1,
            description="Number of candidates to request, 1 to 50.",
            examples=[1, 5, "$inputs.top_k"],
        )
        workspace: str | Ref(STRING_KIND) = Field(
            description="Roboflow workspace URL slug that owns the project.",
            examples=["my-workspace", "$inputs.workspace"],
        )

    def __init__(
        self,
        *,
        visual_search_port: VisualSearchPort = OFFLINE_VISUAL_SEARCH_PORT,
        api_key: Optional[Union[str, Secret]] = None,
    ) -> None:
        self._port = visual_search_port
        self._api_key = api_key

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the searched project."""
        resources = [
            DependentResource(
                resource_type="roboflow_platform_project",
                identifier=params.target_project,
            )
        ]

        return resources

    def run(
        self, *, image: ImageData, target_project: str, top_k: int, workspace: str
    ) -> Dict[str, Any]:
        """Search the project with one image.

        Args:
            image: Query image.
            target_project: Project URL slug.
            top_k: Number of candidates to request.
            workspace: Workspace URL slug.

        Returns:
            V1's outputs for one image.

        Raises:
            ValueError: When ``top_k`` is outside 1..50 or the host gave no
                API key.
            SecretUnavailableError: When the API key is a replayed secret.
            PlatformUnavailableError: When the host gave no port.
        """
        # Literals are checked at compile time; V1 also rejected out-of-range
        # workflow inputs before running. V2 checks only their kind.
        if not 1 <= top_k <= 50:
            raise ValueError(f"top_k must be in [1, 50], got {top_k}")

        api_key = api_key_for_dispatch(
            self._api_key, block_name="Roboflow Visual Search"
        )
        try:
            response = self._port.search_project_images(
                api_key=api_key,
                workspace=workspace,
                project=target_project,
                image_base64=_query_base64(image, max_side=None),
                limit=top_k,
            )
            candidates = [
                _format_candidate(candidate)
                for candidate in response.get("results", [])
            ]
            if not candidates:
                return self._search_result(
                    error_status=False, message="No visually similar images found."
                )

            best_candidate = candidates[0]
            result = {
                **self._search_result(
                    error_status=False, message="Visual search completed."
                ),
                "candidate_found": True,
                "best_candidate": best_candidate,
                "candidates": candidates,
                "best_candidate_metadata": best_candidate["metadata"],
                "best_candidate_tags": best_candidate["tags"],
            }
            if self.wants("best_candidate_image"):
                result["best_candidate_image"] = _candidate_image(
                    self._port, best_candidate, fallback_id="visual_search_candidate"
                )
        except PlatformUnavailableError:
            raise
        except Exception as error:
            return self._search_result(
                error_status=True, message=f"Visual search failed: {error}"
            )

        return result

    @staticmethod
    def _search_result(*, error_status: bool, message: str) -> Dict[str, Any]:
        result = {
            "candidate_found": False,
            "best_candidate": {},
            "candidates": [],
            "error_status": error_status,
            "message": message,
            "best_candidate_image": None,
            "best_candidate_metadata": {},
            "best_candidate_tags": [],
        }

        return result
