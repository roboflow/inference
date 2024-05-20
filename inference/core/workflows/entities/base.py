from enum import Enum
from typing import Any, Generic, Iterator, List, Optional, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

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


class WorkflowImage(BaseModel):
    type: Literal["WorkflowImage", "InferenceImage"]
    name: str
    kind: List[Kind] = Field(default=[BATCH_OF_IMAGES_KIND])


class WorkflowParameter(BaseModel):
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
        for zipped in zip(*(batch.iter_nonempty(masks=mask) for batch in batches)):
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
        results_index = 0
        aligned_results = []
        for mask_element in mask:
            if mask_element:
                aligned_results.append(results[results_index])
                results_index += 1
            else:
                aligned_results.append(null_element)
        return aligned_results

    @classmethod
    def mask_common_empty_elements(cls, batches: List["Batch"]) -> List[bool]:
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
        return list(self.iter_nonempty(masks=index))

    def __len__(self):
        return len(self._content)

    def __iter__(self) -> Iterator[B]:
        yield from self._content

    def iter_nonempty(self, masks: Optional[List[int]] = None) -> Iterator[B]:
        if masks is None:
            masks = self.mask_empty_elements()
        yield from (
            batch_element
            for (batch_element, mask_element) in zip(self._content, masks)
            if mask_element is True
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
