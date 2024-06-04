import base64
from abc import abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar, Union

import cv2
import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from inference.core.utils.image_utils import (
    attempt_loading_image_from_string,
    encode_image_to_jpeg_bytes,
    load_image_from_url,
)
from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGES_KIND,
    WILDCARD_KIND,
    Kind,
)


class StepExecutionMode(Enum):
    LOCAL = "local"
    REMOTE = "remote"


class OutputDefinition(BaseModel):
    name: str
    kind: List[Kind] = Field(default_factory=lambda: [WILDCARD_KIND])


class CoordinatesSystem(Enum):
    OWN = "own"
    PARENT = "parent"


class JsonField(BaseModel):
    type: Literal["JsonField"]
    name: str
    selector: str
    coordinates_system: CoordinatesSystem = Field(default=CoordinatesSystem.PARENT)

    def get_type(self) -> str:
        return self.type


class WorkflowInput(BaseModel):

    @classmethod
    def is_batch_oriented(cls) -> bool:
        return False


class WorkflowImage(WorkflowInput):
    type: Literal["WorkflowImage", "InferenceImage"]
    name: str
    kind: List[Kind] = Field(default=[BATCH_OF_IMAGES_KIND])

    @classmethod
    def is_batch_oriented(cls) -> bool:
        return True


class WorkflowParameter(WorkflowInput):
    type: Literal["WorkflowParameter", "InferenceParameter"]
    name: str
    kind: List[Kind] = Field(default_factory=lambda: [WILDCARD_KIND])
    default_value: Optional[Union[float, int, str, bool, list, set]] = Field(
        default=None
    )


InputType = Annotated[
    Union[WorkflowImage, WorkflowParameter], Field(discriminator="type")
]

B = TypeVar("B")


class Batch(Generic[B]):

    @classmethod
    def zip_nonempty(cls, batches: List["Batch"]) -> Iterator[tuple]:
        mask = cls.mask_common_empty_elements(batches=batches)
        for zipped in zip(
            *(batch.iter_selected(mask=mask, return_index=False) for batch in batches)
        ):
            yield zipped

    @classmethod
    def align_batches_results(
        cls,
        batches: List["Batch"],
        results: List[Any],
        null_element: Any = None,
    ) -> List[Any]:
        mask = cls.mask_common_empty_elements(batches=batches)
        non_empty_batches_elements = sum(mask)
        if non_empty_batches_elements != len(results):
            raise ValueError(
                "Attempted to align batches results in original batch dimensions, but size of "
                f"batch results ({len(results)}) does not match the number of non-empty "
                f"batches elements: {non_empty_batches_elements}."
            )
        return align_results(
            results=results,
            mask=mask,
            null_element=null_element,
        )

    @classmethod
    def mask_common_empty_elements(cls, batches: List["Batch"]) -> List[bool]:
        if not batches:
            return []
        try:
            all_masks = np.array([batch.mask_empty_elements() for batch in batches])
            return np.logical_and.reduce(all_masks).tolist()
        except ValueError as e:
            raise ValueError(
                f"Could not create common masks for batches of not matching size"
            ) from e

    def __init__(self, content: List[B]):
        self._content = content

    def __getitem__(
        self, index: Union[int, List[bool], np.ndarray]
    ) -> Union[B, List[B]]:
        if isinstance(index, int):
            return self._content[index]
        if len(index) != len(self._content):
            raise ValueError(
                f"Mask provided to select batch element has length {len(index)} which does "
                f"not match batch length: {len(self._content)}"
            )
        return list(self.iter_selected(mask=index, return_index=False))

    def __len__(self):
        return len(self._content)

    def __iter__(self) -> Iterator[B]:
        yield from self._content

    def iter_nonempty(
        self,
        return_index: bool = False,
    ) -> Iterator[B]:
        mask = self.mask_empty_elements()
        yield from self.iter_selected(mask=mask, return_index=return_index)

    def iter_selected(
        self,
        mask: Optional[List[bool]] = None,
        return_index: bool = False,
    ) -> Iterator[B]:
        yield from (
            batch_element if not return_index else (idx, batch_element)
            for idx, (batch_element, mask_element) in enumerate(
                zip(self._content, mask)
            )
            if mask_element
        )

    def align_batch_results(
        self,
        results: List[Any],
        null_element: Any = None,
        mask: Optional[List[bool]] = None,
    ) -> List[Any]:
        if mask is None:
            mask = self.mask_empty_elements()
        non_empty_batches_elements = sum(mask)
        if non_empty_batches_elements != len(results):
            raise ValueError(
                "Attempted to align batches results in original batch dimensions, but size of "
                f"batch results ({len(results)}) does not match the number of non-empty "
                f"batches elements: {non_empty_batches_elements}."
            )
        return align_results(
            results=results,
            mask=mask,
            null_element=null_element,
        )

    def mask_empty_elements(self) -> List[bool]:
        return [batch_element is not None for batch_element in self._content]

    def broadcast(self, n: int) -> List[B]:
        if n <= 0:
            raise ValueError(
                f"Broadcast to size {n} requested which is invalid operation."
            )
        if len(self._content) == n:
            return self._content
        if len(self._content) == 1:
            return [self._content[0]] * n
        raise ValueError(
            f"Could not broadcast batch of size {len(self._content)} to size {n}"
        )


def align_results(results: List[Any], mask: List[bool], null_element: Any) -> List[Any]:
    results_index = 0
    aligned_results = []
    for mask_element in mask:
        if mask_element:
            aligned_results.append(results[results_index])
            results_index += 1
        else:
            aligned_results.append(null_element)
    return aligned_results


@dataclass(frozen=True)
class OriginCoordinatesSystem:
    left_top_x: int
    left_top_y: int
    origin_width: int
    origin_height: int


@dataclass(frozen=True)
class ImageParentMetadata:
    parent_id: str
    origin_coordinates: Optional[OriginCoordinatesSystem] = None


class WorkflowImageData:

    def __init__(
        self,
        parent_metadata: ImageParentMetadata,
        workflow_root_ancestor_metadata: Optional[ImageParentMetadata] = None,
        image_reference: Optional[str] = None,
        base64_image: Optional[str] = None,
        numpy_image: Optional[np.ndarray] = None,
    ):
        if not base64_image and numpy_image is None and not image_reference:
            raise ValueError("Could not initialise empty `WorkflowImageData`.")
        self._parent_metadata = parent_metadata
        self._workflow_root_ancestor_metadata = (
            workflow_root_ancestor_metadata
            if workflow_root_ancestor_metadata
            else self._parent_metadata
        )
        self._image_reference = image_reference
        self._base64_image = base64_image
        self._numpy_image = numpy_image

    @property
    def parent_metadata(self) -> ImageParentMetadata:
        if self._parent_metadata.origin_coordinates is None:
            numpy_image = self.numpy_image
            origin_coordinates = OriginCoordinatesSystem(
                left_top_y=0,
                left_top_x=0,
                origin_width=numpy_image.shape[1],
                origin_height=numpy_image.shape[0],
            )
            self._parent_metadata = replace(
                self._parent_metadata, origin_coordinates=origin_coordinates
            )
        return self._parent_metadata

    @property
    def workflow_root_ancestor_metadata(self) -> ImageParentMetadata:
        if self._workflow_root_ancestor_metadata.origin_coordinates is None:
            numpy_image = self.numpy_image
            origin_coordinates = OriginCoordinatesSystem(
                left_top_y=0,
                left_top_x=0,
                origin_width=numpy_image.shape[1],
                origin_height=numpy_image.shape[0],
            )
            self._workflow_root_ancestor_metadata = replace(
                self._workflow_root_ancestor_metadata,
                origin_coordinates=origin_coordinates,
            )
        return self._workflow_root_ancestor_metadata

    @property
    def numpy_image(self) -> np.ndarray:
        if self._numpy_image is not None:
            return self._numpy_image
        if self._base64_image:
            self._numpy_image = attempt_loading_image_from_string(self._base64_image)[0]
            return self._numpy_image
        if self._image_reference.startswith(
            "http://"
        ) or self._image_reference.startswith("https://"):
            self._numpy_image = load_image_from_url(value=self._image_reference)
        else:
            self._numpy_image = cv2.imread(self._image_reference)
        return self._numpy_image

    @property
    def base64_image(self) -> str:
        if self._base64_image is not None:
            return self._base64_image
        numpy_image = self.numpy_image
        self._base64_image = base64.b64encode(
            encode_image_to_jpeg_bytes(numpy_image)
        ).decode("ascii")
        return self._base64_image

    def to_inference_format(self, numpy_preferred: bool = False) -> Dict[str, Any]:
        if numpy_preferred:
            return {"type": "numpy_object", "value": self.numpy_image}
        if self._image_reference:
            if self._image_reference.startswith(
                "http://"
            ) or self._image_reference.startswith("https://"):
                return {"type": "url", "value": self._image_reference}
            return {"type": "file", "value": self._image_reference}
        if self._base64_image:
            return {"type": "base64", "value": self.base64_image}
        return {"type": "numpy_object", "value": self.numpy_image}
