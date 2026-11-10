"""Classify an image with the annotation of the nearest project image.

Canonical type: ``roboflow_core/visual_search_classifier@v1``.
"""

import uuid
from typing import Annotated, Any, Dict, List, Optional, Union
from pydantic import Field
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.ports import (
    OFFLINE_VISUAL_SEARCH_PORT,
    VisualSearchPort,
    api_key_for_dispatch,
)
from roboflow_workflows.v2.blocks.roboflow_platform._shared.visual_search import (
    CLASSIFIER_FIELDS,
    _candidate_image,
    _classification_prediction,
    _confidence,
    _finite_score,
    _format_classifier_candidate,
    _query_base64,
    parse_visual_search_classification,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.predictions import CLASSIFICATION_PREDICTION_KIND
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)


class VisualSearchClassifierBlock(Block):
    """Classify an image with the annotation of the nearest project image.

    A single-label annotation gives a ``ClassificationPrediction``; several
    labels give a ``MultiLabelClassificationPrediction``. Confidence is the
    best candidate's ``score`` divided by 2, clipped to ``[0, 1]``, and 0.0
    without a usable score.

    Resources:
        visual_search_port: Host adapter for the workspace lookup and search;
            the default refuses with ``PlatformUnavailableError``.
        api_key: Roboflow API key, a ``str`` or a ``Secret``. Only the port
            reveals it.
    """

    type = "roboflow_core/visual_search_classifier@v1"
    metadata = {
        "name": "Roboflow Visual Search Classifier",
        "port_of": "roboflow_core/visual_search_classifier@v1",
        "section": "model",
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
        "predictions": Output(
            CLASSIFICATION_PREDICTION_KIND,
            source="image",
            description="Classification from the nearest candidate; None without one.",
        ),
        "inference_id": Output(
            STRING_KIND, source="image", description="Fresh id of this result."
        ),
        "class_found": Output(
            BOOLEAN_KIND,
            description="Whether the nearest candidate carried a classification.",
        ),
        "visual_search_score": Output(
            FLOAT_KIND,
            description="The nearest candidate's finite score; None without one.",
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
        workspace: Optional[str] | Ref(STRING_KIND) = Field(
            default=None,
            description=(
                "Workspace URL slug that owns the project; None resolves it "
                "from the API key."
            ),
            examples=["my-workspace", "$inputs.workspace"],
        )
        max_image_size: Annotated[int, Field(ge=1)] | Ref(INTEGER_KIND) = Field(
            default=224,
            description=(
                "Longest query side in pixels; larger images are downscaled "
                "keeping their aspect ratio before the search."
            ),
            examples=[224, 640, "$inputs.max_image_size"],
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
        self,
        *,
        image: ImageData,
        target_project: str,
        top_k: int,
        workspace: Optional[str],
        max_image_size: int,
    ) -> Dict[str, Any]:
        """Classify one image by project search.

        Args:
            image: Query image.
            target_project: Classification project URL slug.
            top_k: Number of candidates to request.
            workspace: Workspace URL slug, or ``None`` to look it up.
            max_image_size: Longest query side in pixels.

        Returns:
            V1's outputs for one image.

        Raises:
            ValueError: When ``top_k`` is outside 1..50, ``max_image_size`` is
                below 1 or the host gave no API key.
            SecretUnavailableError: When the API key is a replayed secret.
            PlatformUnavailableError: When the host gave no port.
        """
        # Literals are checked at compile time; V1 also rejected out-of-range
        # workflow inputs before running. V2 checks only their kind.
        if not 1 <= top_k <= 50:
            raise ValueError(f"top_k must be in [1, 50], got {top_k}")
        if not 1 <= max_image_size:
            raise ValueError(
                f"max_image_size must be in [1, inf], got {max_image_size}"
            )

        api_key = api_key_for_dispatch(
            self._api_key, block_name="Roboflow Visual Search Classifier"
        )
        inference_id = str(uuid.uuid4())
        try:
            if not workspace:
                workspace = self._port.get_workspace(api_key=api_key)
            response = self._port.search_project_images(
                api_key=api_key,
                workspace=workspace,
                project=target_project,
                image_base64=_query_base64(image, max_side=max_image_size),
                limit=top_k,
                fields=CLASSIFIER_FIELDS,
            )
            candidates = [
                _format_classifier_candidate(candidate)
                for candidate in response.get("results", [])
            ]
            if not candidates:
                return self._search_result(
                    inference_id,
                    error_status=False,
                    message="No visually similar images found.",
                )

            best_candidate = candidates[0]
            result = self._search_result(
                inference_id,
                error_status=True,
                message=(
                    "Best visual search candidate does not include classification "
                    "labels or annotations."
                ),
            )
            result.update(
                candidate_found=True,
                best_candidate=best_candidate,
                candidates=candidates,
                visual_search_score=_finite_score(best_candidate.get("score")),
            )
            if self.wants("best_candidate_image"):
                result["best_candidate_image"] = _candidate_image(
                    self._port,
                    best_candidate,
                    fallback_id="visual_search_classifier_candidate",
                )

            classification = parse_visual_search_classification(best_candidate)
            if classification is not None:
                result.update(
                    predictions=_classification_prediction(
                        image,
                        classification=classification,
                        inference_id=inference_id,
                        confidence=_confidence(best_candidate.get("score")),
                    ),
                    class_found=True,
                    error_status=False,
                    message="Visual search classification completed.",
                )
        except PlatformUnavailableError:
            raise
        except Exception as error:
            return self._search_result(
                inference_id,
                error_status=True,
                message=f"Visual search classification failed: {error}",
            )

        return result

    @staticmethod
    def _search_result(
        inference_id: str, *, error_status: bool, message: str
    ) -> Dict[str, Any]:
        result = {
            "predictions": None,
            "inference_id": inference_id,
            "candidate_found": False,
            "class_found": False,
            "best_candidate": {},
            "candidates": [],
            "best_candidate_image": None,
            "visual_search_score": None,
            "error_status": error_status,
            "message": message,
        }

        return result
