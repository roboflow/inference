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
import torch
from pydantic import BaseModel, Field
from torchvision.io import ImageReadMode, decode_image, read_file
from typing_extensions import Annotated, Literal

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
)
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
    default_value: Optional[Union[float, int, str, bool, list, set, dict]] = Field(
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

    def __init__(self, content: List[B], indices: Optional[List[Tuple[int, ...]]]):
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
        filtered_content = [
            (
                element
                if not isinstance(element, Batch)
                else element.remove_by_indices(indices_to_remove=indices_to_remove)
            )
            for index, element in zip(self._indices, self._content)
            if index not in indices_to_remove
        ]
        filtered_indices = [
            index for index in self._indices if index not in indices_to_remove
        ]

        return Batch(content=filtered_content, indices=filtered_indices)

    def iter_with_indices(self) -> Iterator[Tuple[Tuple[int, ...], B]]:
        return zip(self._indices, self._content)

    def broadcast(self, n: int) -> "Batch":
        if n <= 0:
            raise ValueError(
                f"Broadcast to size {n} requested which is an invalid operation."
            )

        num_content = len(self._content)

        if num_content == n:
            return self

        if num_content == 1:
            return Batch(content=self._content * n, indices=self._indices * n)

        raise ValueError(f"Could not broadcast batch of size {num_content} to size {n}")


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


class ParentOrigin(BaseModel):
    offset_x: int = Field(
        description="Offset from the left of the parent image in pixels"
    )
    offset_y: int = Field(
        description="Offset from the top of the parent image in pixels"
    )
    width: int = Field(
        description="Width of the parent image in pixels",
        gt=0,
    )
    height: int = Field(
        description="Height of the parent image in pixels",
        gt=0,
    )

    @classmethod
    def from_origin_coordinates_system(
        cls, origin_coordinates_system: "OriginCoordinatesSystem"
    ) -> "ParentOrigin":
        return cls(
            offset_x=origin_coordinates_system.left_top_x,
            offset_y=origin_coordinates_system.left_top_y,
            width=origin_coordinates_system.origin_width,
            height=origin_coordinates_system.origin_height,
        )

    def to_origin_coordinates_system(self) -> "OriginCoordinatesSystem":
        return OriginCoordinatesSystem(
            left_top_x=self.offset_x,
            left_top_y=self.offset_y,
            origin_width=self.width,
            origin_height=self.height,
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
    """Container hosting the image in one of several representations, materialised
    lazily on access.

    Layout contract:
      * ``numpy_image``: HWC uint8 BGR (cv2 native); 2-D ``(H, W)`` for
        single-channel images (grayscale / threshold outputs).
      * ``tensor_image``: CHW uint8 RGB on ``WORKFLOWS_IMAGE_TENSOR_DEVICE``;
        ``(1, H, W)`` for single-channel images. Single-channel data carries no
        BGR/RGB semantics, so it is never channel-reversed in either direction.

    Mutation contract: representations may be cached simultaneously (fan-out
    readers alternate between them for free) and are exposed as the raw mutable
    buffers - legacy blocks mutate ``numpy_image`` in place by design (e.g.
    visualizations with ``copy_image=False``), and the class does not police
    that: blind in-place mutation has never been safe here and clients know the
    limitation. A client that mutates a representation in place MUST declare it
    via ``declare_numpy_image_mutated()`` / ``declare_tensor_image_mutated()``,
    which marks the derived sibling caches (tensor/base64 or numpy/base64) as
    stale and removes them, so later readers re-derive from the mutated pixels.
    Undeclared in-place mutation leaves sibling caches stale - the same caveat
    the numpy + base64 pair has always had."""

    def __init__(
        self,
        parent_metadata: ImageParentMetadata,
        workflow_root_ancestor_metadata: Optional[ImageParentMetadata] = None,
        image_reference: Optional[str] = None,
        base64_image: Optional[str] = None,
        numpy_image: Optional[np.ndarray] = None,
        video_metadata: Optional[VideoMetadata] = None,
        tensor_image: Optional[torch.Tensor] = None,
    ):
        if (
            not base64_image
            and numpy_image is None
            and not image_reference
            and tensor_image is None
        ):
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
        self._tensor_image = (
            tensor_image.to(WORKFLOWS_IMAGE_TENSOR_DEVICE)
            if tensor_image is not None
            else None
        )
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
        * tensor_image
        * video_metadata

        When more than one from ["numpy_image", "base64_image", "image_reference", "tensor_image"]
        args are given, they MUST be compliant.
        """
        parent_metadata = origin_image_data._parent_metadata
        workflow_root_ancestor_metadata = (
            origin_image_data._workflow_root_ancestor_metadata
        )
        image_reference = origin_image_data._image_reference
        base64_image = origin_image_data._base64_image
        numpy_image = origin_image_data._numpy_image
        tensor_image = origin_image_data._tensor_image
        video_metadata = origin_image_data._video_metadata
        if any(
            k in kwargs
            for k in ["numpy_image", "base64_image", "image_reference", "tensor_image"]
        ):
            numpy_image = kwargs.get("numpy_image")
            base64_image = kwargs.get("base64_image")
            image_reference = kwargs.get("image_reference")
            tensor_image = kwargs.get("tensor_image")
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
            tensor_image=tensor_image,
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

    @classmethod
    def create_crop_from_tensor(
        cls,
        origin_image_data: "WorkflowImageData",
        crop_identifier: str,
        cropped_tensor_image: torch.Tensor,
        offset_x: int,
        offset_y: int,
        preserve_video_metadata: bool = False,
    ) -> "WorkflowImageData":
        """
        Tensor-native mirror of `create_crop`. Identical metadata math;
        the child carries `tensor_image` instead of `numpy_image`.
        """
        origin_h, origin_w = origin_image_data._read_shape_without_materialization()
        parent_metadata = ImageParentMetadata(
            parent_id=crop_identifier,
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=offset_x,
                left_top_y=offset_y,
                origin_width=origin_w,
                origin_height=origin_h,
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
            tensor_image=cropped_tensor_image,
            video_metadata=video_metadata,
        )

    @property
    def parent_metadata(self) -> ImageParentMetadata:
        if self._parent_metadata.origin_coordinates is None:
            h, w = self._read_shape_without_materialization()
            origin_coordinates = OriginCoordinatesSystem(
                left_top_y=0,
                left_top_x=0,
                origin_width=w,
                origin_height=h,
            )
            self._parent_metadata = replace(
                self._parent_metadata, origin_coordinates=origin_coordinates
            )
        return self._parent_metadata

    @property
    def workflow_root_ancestor_metadata(self) -> ImageParentMetadata:
        if self._workflow_root_ancestor_metadata.origin_coordinates is None:
            h, w = self._read_shape_without_materialization()
            origin_coordinates = OriginCoordinatesSystem(
                left_top_y=0,
                left_top_x=0,
                origin_width=w,
                origin_height=h,
            )
            self._workflow_root_ancestor_metadata = replace(
                self._workflow_root_ancestor_metadata,
                origin_coordinates=origin_coordinates,
            )
        return self._workflow_root_ancestor_metadata

    def _read_shape_without_materialization(self) -> Tuple[int, int]:
        """Returns (height, width). Prefers whichever representation is already
        set, so no numpy<->tensor conversion is ever triggered. When neither is
        materialised yet (a base64/reference-born image), the source is decoded
        into the representation the current mode works with - tensor under
        ENABLE_TENSOR_DATA_REPRESENTATION (every caller of this helper is a
        tensor-mode block that will need the tensor anyway), numpy otherwise -
        so the shape read does not leave behind a representation the run has no
        use for."""
        if self._numpy_image is not None:
            return self._numpy_image.shape[0], self._numpy_image.shape[1]
        if self._tensor_image is not None:
            # tensor_image is CHW -> H=shape[1], W=shape[2]
            return (
                int(self._tensor_image.shape[1]),
                int(self._tensor_image.shape[2]),
            )
        if ENABLE_TENSOR_DATA_REPRESENTATION:
            tensor = self.tensor_image
            return int(tensor.shape[1]), int(tensor.shape[2])
        np_img = self.numpy_image
        return np_img.shape[0], np_img.shape[1]

    @property
    def numpy_image(self) -> np.ndarray:
        # Layout + mutation contract: see the class docstring. In-place mutators
        # of the returned buffer must call declare_numpy_image_mutated().
        if self._numpy_image is not None:
            return self._numpy_image
        if self._tensor_image is not None:
            if int(self._tensor_image.shape[0]) == 1:
                # Single-channel: (1, H, W) -> (H, W), no channel reversal.
                self._numpy_image = (
                    self._tensor_image.detach().squeeze(0).to("cpu").numpy().copy()
                )
            else:
                # CHW RGB -> HWC (permute on-device before host transfer) -> BGR
                hwc_rgb = self._tensor_image.detach().permute(1, 2, 0).to("cpu").numpy()
                self._numpy_image = hwc_rgb[:, :, ::-1].copy()
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
    def tensor_image(self) -> torch.Tensor:
        # Layout + mutation contract: see the class docstring. In-place mutators
        # of the returned tensor must call declare_tensor_image_mutated().
        if self._tensor_image is not None:
            return self._tensor_image
        if self._numpy_image is not None:
            bgr_np = self._numpy_image
            if bgr_np.ndim == 2:
                # Single-channel (grayscale / threshold outputs): (H, W) ->
                # (1, H, W), no channel reversal - there is no BGR/RGB
                # semantics to convert.
                chw = torch.from_numpy(np.ascontiguousarray(bgr_np)).unsqueeze(0)
            else:
                # HWC BGR -> HWC RGB -> CHW RGB; contiguous so model ingestion
                # gets a dense buffer.
                chw = torch.from_numpy(bgr_np[:, :, ::-1].copy()).permute(2, 0, 1)
        else:
            # A base64/reference-born image asked for the tensor first: the
            # source decodes DIRECTLY into a CHW RGB tensor - no numpy hop and
            # nothing cached besides the tensor itself.
            chw = self._decode_source_to_tensor()
        # Allocated on the globally configured WORKFLOWS_IMAGE_TENSOR_DEVICE.
        self._tensor_image = chw.contiguous().to(WORKFLOWS_IMAGE_TENSOR_DEVICE)
        return self._tensor_image

    def _decode_source_to_tensor(self) -> torch.Tensor:
        """Decode the base64 / file / URL source straight into a CHW RGB uint8
        tensor via ``torchvision.io`` - no numpy intermediate, no cv2. Only
        reached when neither in-memory representation exists (the constructor
        guarantees a source). EXIF handling mirrors the numpy path: cv2.imdecode
        (the base64 route) does not apply EXIF orientation, cv2.imread (local
        files) does. URLs are the one exception: their bytes come through the
        SSRF-guarded numpy loader today, so they keep a single numpy hop until a
        bytes-level guarded fetcher exists."""

        if self._base64_image:
            payload = torch.frombuffer(
                bytearray(base64.b64decode(self._base64_image)), dtype=torch.uint8
            )
            return decode_image(payload, mode=ImageReadMode.RGB)
        if self._image_reference.startswith(
            "http://"
        ) or self._image_reference.startswith("https://"):
            hwc_bgr = load_image_from_url(value=self._image_reference)
            return torch.from_numpy(hwc_bgr[:, :, ::-1].copy()).permute(2, 0, 1)
        return decode_image(
            read_file(self._image_reference),
            mode=ImageReadMode.RGB,
            apply_exif_orientation=True,
        )

    def declare_numpy_image_mutated(self) -> None:
        """A client that mutated the ``numpy_image`` buffer in place declares it
        here: the derived sibling caches (tensor / base64) are stale and get
        removed, so later readers re-derive from the mutated pixels. See the
        class-level mutation contract."""
        self._tensor_image = None
        self._base64_image = None

    def declare_tensor_image_mutated(self) -> None:
        """A client that mutated the ``tensor_image`` in place declares it here:
        the derived sibling caches (numpy / base64) are stale and get removed,
        so later readers re-derive from the mutated pixels. See the class-level
        mutation contract."""
        self._numpy_image = None
        self._base64_image = None

    def is_tensor_materialised(self) -> bool:
        """Whether the CHW RGB tensor image already exists on device.

        ``True`` only when a tensor representation is already present (the caller fed
        one in, or something downstream already built it) — reading ``tensor_image`` is
        then free. ``False`` means only the numpy (HWC BGR) representation is available,
        so accessing ``tensor_image`` would trigger an eager numpy->device conversion.

        Blocks use this to pick the representation that is already materialised instead
        of forcing a conversion: ``tensor_image`` (RGB) when ``True``, ``numpy_image``
        (BGR) otherwise. The check itself does no I/O and never materialises anything.
        """
        return self._tensor_image is not None

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
