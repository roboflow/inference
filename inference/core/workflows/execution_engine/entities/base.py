import base64
from copy import copy
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import cv2
import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from inference.core.utils.image_utils import (
    attempt_loading_image_from_string,
    encode_image_to_jpeg_bytes,
    load_image_from_url,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    VIDEO_METADATA_KIND,
    WILDCARD_KIND,
    Kind,
)


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
    type: str
    name: str
    kind: List[Union[str, Kind]]
    dimensionality: int

    @classmethod
    def is_batch_oriented(cls) -> bool:
        return False


class WorkflowImage(WorkflowInput):
    type: Literal["WorkflowImage", "InferenceImage"]
    name: str
    kind: List[Union[str, Kind]] = Field(default=[IMAGE_KIND])
    dimensionality: int = Field(default=1, ge=1, le=1)

    @classmethod
    def is_batch_oriented(cls) -> bool:
        return True


class WorkflowVideoMetadata(WorkflowInput):
    type: Literal["WorkflowVideoMetadata"]
    name: str
    kind: List[Union[str, Kind]] = Field(default=[VIDEO_METADATA_KIND])
    dimensionality: int = Field(default=1, ge=1, le=1)

    @classmethod
    def is_batch_oriented(cls) -> bool:
        return True


class WorkflowBatchInput(WorkflowInput):
    type: Literal["WorkflowBatchInput"]
    name: str
    kind: List[Union[str, Kind]] = Field(default_factory=lambda: [WILDCARD_KIND])
    dimensionality: int = Field(default=1)

    @classmethod
    def is_batch_oriented(cls) -> bool:
        return True


class WorkflowParameter(WorkflowInput):
    type: Literal["WorkflowParameter", "InferenceParameter"]
    name: str
    kind: List[Union[str, Kind]] = Field(default_factory=lambda: [WILDCARD_KIND])
    default_value: Optional[Union[float, int, str, bool, list, set]] = Field(
        default=None
    )
    dimensionality: int = Field(default=0, ge=0, le=0)


InputType = Annotated[
    Union[WorkflowImage, WorkflowVideoMetadata, WorkflowParameter, WorkflowBatchInput],
    Field(discriminator="type"),
]

B = TypeVar("B")


class Batch(Generic[B]):

    @classmethod
    def init(
        cls,
        content: List[B],
        indices: List[Tuple[int, ...]],
    ) -> "Batch":
        if len(content) != len(indices):
            raise ValueError(
                "Attempted to initialise Batch object providing batch indices of size differing "
                "from size of the data."
            )

        return cls(content=content, indices=indices)

    def __init__(
        self,
        content: List[B],
        indices: Optional[List[Tuple[int, ...]]],
    ):
        self._content = content
        self._indices = indices

    @property
    def indices(self) -> List[Tuple[int, ...]]:
        return copy(self._indices)

    def __getitem__(
        self,
        index: int,
    ) -> B:
        return self._content[index]

    def __len__(self):
        return len(self._content)

    def __iter__(self) -> Iterator[B]:
        yield from self._content

    def remove_by_indices(self, indices_to_remove: Set[tuple]) -> "Batch":
        content, new_indices = [], []
        for index, element in self.iter_with_indices():
            if index in indices_to_remove:
                continue
            content.append(element)
            new_indices.append(index)
        return Batch(
            content=content,
            indices=new_indices,
        )

    def iter_with_indices(self) -> Iterator[Tuple[Tuple[int, ...], B]]:
        for index, element in zip(self._indices, self._content):
            yield index, element

    def broadcast(self, n: int) -> "Batch":
        if n <= 0:
            raise ValueError(
                f"Broadcast to size {n} requested which is invalid operation."
            )
        if len(self._content) == n:
            return self
        if len(self._content) == 1:
            return Batch(content=[self._content[0]] * n, indices=[self._indices[0]] * n)
        raise ValueError(
            f"Could not broadcast batch of size {len(self._content)} to size {n}"
        )


class VideoMetadata(BaseModel):
    video_identifier: str = Field(
        description="Identifier string for video. To be treated as opaque."
    )
    frame_number: int
    frame_timestamp: datetime = Field(
        description="The timestamp of video frame. When processing video it is suggested that "
        "blocks rely on `fps` and `frame_number`, as real-world time-elapse will not "
        "match time-elapse in video file",
    )
    fps: Optional[float] = Field(
        description="Field represents FPS value (if possible to be retrieved)",
        default=None,
    )
    measured_fps: Optional[float] = Field(
        description="Field represents measured FPS of live stream",
        default=None,
    )
    comes_from_video_file: Optional[bool] = Field(
        description="Field is a flag telling if frame comes from video file or stream - "
        "if not possible to be determined - pass None",
        default=None,
    )


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
        video_metadata: Optional[VideoMetadata] = None,
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
        self._video_metadata = video_metadata

    @classmethod
    def copy_and_replace(
        cls, origin_image_data: "WorkflowImageData", **kwargs
    ) -> "WorkflowImageData":
        """
        Creates new instance of `WorkflowImageData` with updated property.

        Properties are passed by kwargs, supported properties are:
        * parent_metadata
        * workflow_root_ancestor_metadata
        * image_reference
        * base64_image
        * numpy_image
        * video_metadata

        When more than one from ["numpy_image", "base64_image", "image_reference"] args are
        given, they MUST be compliant.
        """
        parent_metadata = origin_image_data._parent_metadata
        workflow_root_ancestor_metadata = (
            origin_image_data._workflow_root_ancestor_metadata
        )
        image_reference = origin_image_data._image_reference
        base64_image = origin_image_data._base64_image
        numpy_image = origin_image_data._numpy_image
        video_metadata = origin_image_data._video_metadata
        if any(k in kwargs for k in ["numpy_image", "base64_image", "image_reference"]):
            numpy_image = kwargs.get("numpy_image")
            base64_image = kwargs.get("base64_image")
            image_reference = kwargs.get("image_reference")
        if "parent_metadata" in kwargs:
            if workflow_root_ancestor_metadata is parent_metadata:
                workflow_root_ancestor_metadata = kwargs["parent_metadata"]
            parent_metadata = kwargs["parent_metadata"]
        if "workflow_root_ancestor_metadata" in kwargs:
            if parent_metadata is workflow_root_ancestor_metadata:
                parent_metadata = kwargs["workflow_root_ancestor_metadata"]
            workflow_root_ancestor_metadata = kwargs["workflow_root_ancestor_metadata"]
        if "video_metadata" in kwargs:
            video_metadata = kwargs["video_metadata"]
        return cls(
            parent_metadata=parent_metadata,
            workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
            image_reference=image_reference,
            base64_image=base64_image,
            numpy_image=numpy_image,
            video_metadata=video_metadata,
        )

    @classmethod
    def create_crop(
        cls,
        origin_image_data: "WorkflowImageData",
        crop_identifier: str,
        cropped_image: np.ndarray,
        offset_x: int,
        offset_y: int,
        preserve_video_metadata: bool = False,
    ) -> "WorkflowImageData":
        """
        Creates new instance of `WorkflowImageData` being a crop of original image,
        making adjustment to all metadata.
        """
        parent_metadata = ImageParentMetadata(
            parent_id=crop_identifier,
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=offset_x,
                left_top_y=offset_y,
                origin_width=origin_image_data.numpy_image.shape[1],
                origin_height=origin_image_data.numpy_image.shape[0],
            ),
        )
        workflow_root_ancestor_coordinates = replace(
            origin_image_data.workflow_root_ancestor_metadata.origin_coordinates,
            left_top_x=origin_image_data.workflow_root_ancestor_metadata.origin_coordinates.left_top_x
            + offset_x,
            left_top_y=origin_image_data.workflow_root_ancestor_metadata.origin_coordinates.left_top_y
            + offset_y,
        )
        workflow_root_ancestor_metadata = ImageParentMetadata(
            parent_id=origin_image_data.workflow_root_ancestor_metadata.parent_id,
            origin_coordinates=workflow_root_ancestor_coordinates,
        )
        video_metadata = None
        if preserve_video_metadata and origin_image_data._video_metadata is not None:
            video_metadata = copy(origin_image_data._video_metadata)
            video_metadata.video_identifier = (
                f"{video_metadata.video_identifier} | crop: {crop_identifier}"
            )
        return WorkflowImageData(
            parent_metadata=parent_metadata,
            workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
            numpy_image=cropped_image,
            video_metadata=video_metadata,
        )

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
            encode_image_to_jpeg_bytes(numpy_image, jpeg_quality=95)
        ).decode("ascii")
        return self._base64_image

    @property
    def video_metadata(self) -> VideoMetadata:
        if self._video_metadata is not None:
            return self._video_metadata
        return VideoMetadata(
            video_identifier=self.parent_metadata.parent_id,
            frame_number=0,
            frame_timestamp=datetime.now(),
            fps=30,
            comes_from_video_file=None,
        )

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
