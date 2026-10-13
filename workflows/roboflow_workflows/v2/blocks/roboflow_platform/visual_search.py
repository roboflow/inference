"""Roboflow project image search, and classification by the nearest image.

Ports of V1 ``roboflow_core/visual_search@v1`` and
``roboflow_core/visual_search_classifier@v1`` (latest tensor sibling). Both
call the platform's project search; neither runs a model, locally or
remotely::

    image ─▶ JPEG q95 (classifier: downscaled to max_image_size first)
          ─▶ port.search_project_images ─▶ candidates, best first
                                            ├─ none ─▶ candidate_found=False
                                            └─ best ─▶ outputs
    classifier: best candidate's classification / labels / annotations
          ─▶ ClassificationPrediction or MultiLabelClassificationPrediction
    best_candidate_image: port.load_candidate_image(image_url), only when
          some reader wants it
    any failure ─▶ error_status=True and V1's "... failed: <error>" message,
          except a missing port or API key, which fails the step

Kept from V1: parameters, outputs, candidate fields, messages, score and
confidence rules, the classification parsing rules and the class id layout of
the tensor sibling (ids below the matched one are filled with ``str(id)``
names and confidence 0.0).

Differences from V1:

* The query is always re-encoded as JPEG quality 95; V1 sent a base64 input
  image's original bytes unchanged.
* ``best_candidate_image`` is downloaded in the step, and only when demanded.
  V1 built a lazy image that downloaded when a later step read its pixels.
  A failed download fails the search result, like any other failure.
* The classifier searches the images of a batch one after another. V1 sent up
  to eight requests at once.
* ``target_project`` is a ``string``; V2 has no ``roboflow_project`` kind.
* A ``classification`` with a class name but no usable ``class_id`` gets class
  id 0. V1 reported -1, and its tensor sibling failed on it.
* Classification tensors live on the input image's device; their metadata
  holds the image's frame record, ``class_names``, ``prediction_type`` and
  ``inference_id``. V1's serializer-only keys are not written: V2 serializes
  native predictions as they are.
"""

import base64
import math
import uuid
from typing import Annotated, Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.ports import (
    OFFLINE_VISUAL_SEARCH_PORT,
    VisualSearchPort,
    api_key_for_dispatch,
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
from roboflow_workflows.v2.data.row_identity import with_classification_parent
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)

TopK = Annotated[int, Field(ge=1, le=50)]
MaxImageSize = Annotated[int, Field(ge=1)]

JPEG_QUALITY = 95
CANDIDATE_IMAGE_OUTPUT = "best_candidate_image"
CLASSIFIER_FIELDS = (
    "id",
    "name",
    "filename",
    "url",
    "user_metadata",
    "tags",
    "width",
    "height",
    "aspectRatio",
    "score",
    "labels",
    "annotations",
)
"""Search result fields the classifier requests, as V1."""

__all__ = ["VisualSearchBlock", "VisualSearchClassifierBlock"]

_CANDIDATE_OUTPUTS = {
    "candidate_found": Output(
        BOOLEAN_KIND, description="Whether the search returned any candidate."
    ),
    "best_candidate": Output(
        DICTIONARY_KIND, description="Nearest candidate; {} when none."
    ),
    "candidates": Output(
        LIST_OF_VALUES_KIND, description="Candidates, nearest first; [] when none."
    ),
    CANDIDATE_IMAGE_OUTPUT: Output(
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
}


class _SearchParams(BlockParams):
    image: Ref(IMAGE_KIND) = Field(
        description="Query image.",
        examples=["$inputs.image", "$steps.crop.crops"],
    )
    target_project: str | Ref(STRING_KIND) = Field(
        description="Roboflow project URL slug to search.",
        examples=["reference-images", "$inputs.target_project"],
    )
    top_k: TopK | Ref(INTEGER_KIND) = Field(
        default=1,
        description="Number of candidates to request, 1 to 50.",
        examples=[1, 5, "$inputs.top_k"],
    )


class _ProjectSearchBlock(Block):
    """Shared resources and request path of both search blocks."""

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

    @staticmethod
    def _check_range(name: str, value: int, *, low: int, high: float) -> None:
        # Literals are checked at compile time; V1 also rejected out-of-range
        # workflow inputs before running. V2 checks only their kind.
        if not low <= value <= high:
            raise ValueError(f"{name} must be in [{low}, {high}], got {value}")

    def _candidate_image(
        self, candidate: Dict[str, Any], *, fallback_id: str
    ) -> Optional[ImageData]:
        image_url = candidate.get("image_url")
        if not image_url:
            return None

        pixels = self._port.load_candidate_image(image_url)
        image_id = candidate.get("image_id") or candidate.get("filename") or fallback_id
        image = ImageData.from_numpy_rgb(pixels, image_id=str(image_id))

        return image


class VisualSearchBlock(_ProjectSearchBlock):
    """Find the images of a project that look most like the query image.

    Resources:
        visual_search_port: Host adapter for the search; the default refuses
            with ``PlatformUnavailableError``.
        api_key: Roboflow API key, a ``str`` or a ``Secret``. Only the port
            reveals it.
    """

    type = "v2/visual_search@v1"
    metadata = {
        "name": "Roboflow Visual Search",
        "port_of": "roboflow_core/visual_search@v1",
        "section": "advanced",
        "requires_rf_key": True,
    }
    outputs = {
        **_CANDIDATE_OUTPUTS,
        "best_candidate_metadata": Output(
            DICTIONARY_KIND, description="user_metadata of the nearest candidate."
        ),
        "best_candidate_tags": Output(
            LIST_OF_VALUES_KIND, description="Tags of the nearest candidate."
        ),
    }

    class Params(_SearchParams):
        workspace: str | Ref(STRING_KIND) = Field(
            description="Roboflow workspace URL slug that owns the project.",
            examples=["my-workspace", "$inputs.workspace"],
        )

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
        self._check_range("top_k", top_k, low=1, high=50)
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
            if self.wants(CANDIDATE_IMAGE_OUTPUT):
                result[CANDIDATE_IMAGE_OUTPUT] = self._candidate_image(
                    best_candidate, fallback_id="visual_search_candidate"
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
            CANDIDATE_IMAGE_OUTPUT: None,
            "best_candidate_metadata": {},
            "best_candidate_tags": [],
        }

        return result


class VisualSearchClassifierBlock(_ProjectSearchBlock):
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

    type = "v2/visual_search_classifier@v1"
    metadata = {
        "name": "Roboflow Visual Search Classifier",
        "port_of": "roboflow_core/visual_search_classifier@v1",
        "section": "model",
        "requires_rf_key": True,
    }
    outputs = {
        **_CANDIDATE_OUTPUTS,
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

    class Params(_SearchParams):
        workspace: Optional[str] | Ref(STRING_KIND) = Field(
            default=None,
            description=(
                "Workspace URL slug that owns the project; None resolves it "
                "from the API key."
            ),
            examples=["my-workspace", "$inputs.workspace"],
        )
        max_image_size: MaxImageSize | Ref(INTEGER_KIND) = Field(
            default=224,
            description=(
                "Longest query side in pixels; larger images are downscaled "
                "keeping their aspect ratio before the search."
            ),
            examples=[224, 640, "$inputs.max_image_size"],
        )

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
        self._check_range("top_k", top_k, low=1, high=50)
        self._check_range("max_image_size", max_image_size, low=1, high=math.inf)
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
                return _classifier_result(
                    inference_id,
                    error_status=False,
                    message="No visually similar images found.",
                )

            best_candidate = candidates[0]
            result = _classifier_result(
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
            if self.wants(CANDIDATE_IMAGE_OUTPUT):
                result[CANDIDATE_IMAGE_OUTPUT] = self._candidate_image(
                    best_candidate, fallback_id="visual_search_classifier_candidate"
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
            return _classifier_result(
                inference_id,
                error_status=True,
                message=f"Visual search classification failed: {error}",
            )

        return result


def _classifier_result(
    inference_id: str, *, error_status: bool, message: str
) -> Dict[str, Any]:
    result = {
        "predictions": None,
        "inference_id": inference_id,
        "candidate_found": False,
        "class_found": False,
        "best_candidate": {},
        "candidates": [],
        CANDIDATE_IMAGE_OUTPUT: None,
        "visual_search_score": None,
        "error_status": error_status,
        "message": message,
    }

    return result


def _query_base64(image: ImageData, *, max_side: Optional[int]) -> str:
    # V1 bytes: BGR host pixels, cv2 bilinear downscale, JPEG quality 95.
    pixels = image.tensor_image.permute(1, 2, 0).cpu().numpy()
    if image.channels == 3:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    else:
        pixels = pixels[:, :, 0]
    if max_side is not None:
        pixels = _downscale(pixels, max_side=max_side)

    encoded, jpeg = cv2.imencode(
        ".jpg", pixels, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not encoded:
        raise ValueError("the query image could not be encoded as JPEG")

    query = base64.b64encode(jpeg.tobytes()).decode("ascii")

    return query


def _downscale(pixels: np.ndarray, *, max_side: int) -> np.ndarray:
    # V1 downscale_image_keeping_aspect_ratio with a square (max_side, max_side).
    height, width = pixels.shape[:2]
    if height <= max_side and width <= max_side:
        return pixels

    if width / height >= 1:
        new_width, new_height = max_side, int(max_side / (width / height))
    else:
        new_width, new_height = int(max_side * (width / height)), max_side
    resized = cv2.resize(pixels, (new_width, new_height))

    return resized


def _format_candidate(
    candidate: Dict[str, Any], extra_fields: Sequence[str] = ()
) -> Dict[str, Any]:
    filename = candidate.get("filename") or candidate.get("name")
    formatted = {
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
    for field in extra_fields:
        formatted[field] = candidate.get(field)

    return formatted


def _format_classifier_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    extra_fields = ["score", "labels", "annotations"]
    if "classification" in candidate:
        extra_fields.append("classification")

    formatted = _format_candidate(candidate, extra_fields=extra_fields)
    formatted["score"] = _finite_score(candidate.get("score"))

    return formatted


def _finite_score(score: Any) -> Optional[float]:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def _confidence(score: Any) -> float:
    # V1: project search scores map to [0, 1] as score / 2, clipped.
    value = _finite_score(score)
    if value is None:
        return 0.0

    return max(0.0, min(1.0, value / 2.0))


def _classification_prediction(
    image: ImageData,
    *,
    classification: Dict[str, Any],
    inference_id: str,
    confidence: float,
) -> Union[ClassificationPrediction, MultiLabelClassificationPrediction]:
    # Dense vectors indexed by class id, as the V1 tensor sibling: ids below the
    # largest matched one are named str(id) with confidence 0.0, because the
    # project's full class list is not part of a search result. float64 keeps
    # the API-provided confidence exact.
    classes = classification["classes"]
    class_count = max(int(entry["class_id"]) for entry in classes) + 1
    class_names = {class_id: str(class_id) for class_id in range(class_count)}
    confidences = [0.0] * class_count
    for entry in classes:
        class_names[int(entry["class_id"])] = str(entry["class"])
        confidences[int(entry["class_id"])] = confidence

    metadata = {
        **image.prediction_metadata(),
        "class_names": class_names,
        "prediction_type": "classification",
        "inference_id": inference_id,
    }
    device = image.device
    if classification["type"] == "multi_label":
        prediction = MultiLabelClassificationPrediction(
            class_ids=torch.tensor(
                [int(entry["class_id"]) for entry in classes],
                dtype=torch.long,
                device=device,
            ),
            confidence=torch.tensor(confidences, dtype=torch.float64, device=device),
            image_metadata=metadata,
        )
    else:
        prediction = ClassificationPrediction(
            class_id=torch.tensor(
                [int(classes[0]["class_id"])], dtype=torch.long, device=device
            ),
            confidence=torch.tensor([confidences], dtype=torch.float64, device=device),
            images_metadata=[metadata],
        )

    parented = with_classification_parent(prediction, image.parent_row)

    return parented


def parse_visual_search_classification(
    candidate: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Read a candidate's classification, as V1's annotation parser.

    Sources in order: ``classification`` (``class`` or ``predicted_classes``),
    then ``labels``, then the ``classes`` of ``annotations``. Repeated names
    are dropped. In ``labels``, ``annotations`` and ``predicted_classes``, a
    missing, negative, non-integer or repeated class id is replaced by the
    entry's position, or the next unused id after it. A single ``class``
    without a usable id gets id 0.

    Args:
        candidate: A formatted search candidate.

    Returns:
        ``{"type": "single_label" | "multi_label", "classes": [{"class":
        name, "class_id": id}, ...]}``, or ``None`` when no source names a
        class.
    """
    classification = candidate.get("classification")
    if isinstance(classification, dict):
        if isinstance(classification.get("class"), str) and classification["class"]:
            # V1 kept a missing id as -1, which its tensor sibling could not
            # index; class 0 is the one id every dense vector has.
            class_id = max(0, _class_id(classification.get("class_id")))
            single = {
                "type": "single_label",
                "classes": [{"class": classification["class"], "class_id": class_id}],
            }
            return single

        parsed = _multi_label_classification(classification)
        if parsed is not None:
            return parsed

    labels = candidate.get("labels")
    if isinstance(labels, list):
        entries = [
            {"class": label} if isinstance(label, str) else label
            for label in labels
            if isinstance(label, (str, dict))
        ]
        parsed = _classes_from_entries(entries)
        if parsed is not None:
            return parsed

    annotations = candidate.get("annotations")
    if not isinstance(annotations, list):
        return None

    names = []
    for annotation in annotations:
        if isinstance(annotation, dict) and isinstance(annotation.get("classes"), list):
            names.extend(annotation["classes"])
    parsed = _classes_from_entries([{"class": name} for name in names])

    return parsed


def _multi_label_classification(
    classification: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    predicted_classes = classification.get("predicted_classes")
    if not isinstance(predicted_classes, list) or not predicted_classes:
        return None

    predictions = classification.get("predictions")
    if not isinstance(predictions, dict):
        predictions = {}
    entries = []
    for name in predicted_classes:
        prediction = predictions.get(name) if isinstance(name, str) else None
        class_id = prediction.get("class_id") if isinstance(prediction, dict) else None
        entries.append({"class": name, "class_id": class_id})
    classes = _unique_classes(entries)
    if not classes:
        return None

    # V1: predicted_classes are multi-label even when one class survives.
    parsed = {"type": "multi_label", "classes": classes}

    return parsed


def _classes_from_entries(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    classes = _unique_classes(entries)
    if not classes:
        return None

    label_type = "single_label" if len(classes) == 1 else "multi_label"
    parsed = {"type": label_type, "classes": classes}

    return parsed


def _unique_classes(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    classes = []
    seen_names = set()
    used_ids = set()
    for position, entry in enumerate(entries):
        name = entry.get("class")
        if not isinstance(name, str) or not name or name in seen_names:
            continue

        class_id = _class_id(entry.get("class_id"))
        if class_id < 0 or class_id in used_ids:
            class_id = position
            while class_id in used_ids:
                class_id += 1
        classes.append({"class": name, "class_id": class_id})
        seen_names.add(name)
        used_ids.add(class_id)

    return classes


def _class_id(value: Any) -> int:
    if value is None:
        return -1

    try:
        class_id = int(value)
    except (TypeError, ValueError):
        return -1

    return class_id
