"""Tensor image payload of the native V2 image catalogue.

``ImageData`` holds pixels as a ``(channels, height, width)`` ``uint8`` torch
tensor (``channels`` is 1 or 3; three channels are RGB) and records where the
pixels come from. Two ``FrameMapping`` records place the image in a reference
frame::

    frame_xy = local_xy * scale_xy + offset_xy

``parent`` maps into the image this one was directly cropped or resized from;
``root`` maps into the workflow input it ultimately comes from. A workflow input
is its own parent and root. Sizes are ``(height, width)``; points, offsets and
scales are ``(x, y)``; pixel ``i`` covers the interval ``[i, i + 1)``.

======================  =================================  ======================
operation               new ``parent``                     new ``root``
======================  =================================  ======================
input                   identity onto itself               identity onto itself
crop at ``(x0, y0)``    scale 1, offset ``(x0, y0)``       ``parent.then(root)``
resize ``hw -> HW``     scale ``(w/W, h/H)``, offset 0     ``parent.then(root)``
``with_pixels``         unchanged                          unchanged
``copy``                unchanged                          unchanged
``composite`` (mosaic)  identity onto the new canvas       identity onto canvas
======================  =================================  ======================

A composite canvas is a new root. Its ``composite_sources`` say where each
source image was placed; no single mapping leads back to the sources, so
geometry helpers must refuse to restore composite predictions to a source.
Crops and resizes of a composite keep ``composite_sources``.

Ownership: ``ImageData`` never copies the tensor it is given and never moves
it to another device unasked. One payload is shared by every consumer of a step
output. Blocks that modify ``tensor_image`` in place must declare that mutation
in their contract so the compiler can detect conflicting branches. To preserve
an input, build a new tensor and call ``with_pixels``. ``crop`` returns an
independent copy and ``resize`` a new tensor. A block that keeps an image
beyond the current pulse calls ``copy``, so the producer's buffer (e.g. a
camera pool slot) is released::

    kept = image.copy()               # same device, owned pixels, ready
    kept = image.copy(device="cpu")   # owned host pixels

Only the explicit host boundaries (``from_numpy_rgb``, ``copy(device="cpu")``,
the V1 adapter and ``IMAGE_KIND`` serialization) touch host memory.

``prediction_metadata()`` describes the image for a native prediction's
``image_metadata``: the V1 keys plus the typed ``frames.FrameRef`` record,
which also carries source and time. ``legacy_prediction_metadata`` writes the
V1 keys alone.
"""

import math
import uuid
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn.functional as F
from roboflow_workflows.v2.data._resize import resize_area
from roboflow_workflows.v2.data.row_identity import RowRef

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.entities.base import (
        VideoMetadata,
        WorkflowImageData,
    )

SUPPORTED_CHANNELS = (1, 3)
INTERPOLATIONS = ("bilinear", "nearest", "area")

Pair = Tuple[float, float]
SizeHW = Tuple[int, int]
Rectangle = Tuple[int, int, int, int]


@dataclass(frozen=True)
class FrameMapping:
    """Placement of an image's pixel coordinates in one reference frame.

    ``frame_xy = local_xy * scale_xy + offset_xy``. ``scale_xy`` counts frame
    pixels per local pixel, so a two-fold downscale has scale ``(2.0, 2.0)``.

    Args:
        frame_id: Identity of the reference frame; ``None`` only when an
            imported V1 image cannot name it.
        frame_size_hw: Size of the reference frame, ``(height, width)``.
        scale_xy: Frame pixels per local pixel, finite and positive.
        offset_xy: Frame position of the local origin, finite.

    Raises:
        ValueError: On an empty id, a non-positive size or scale, or a
            non-finite number.
    """

    frame_id: Optional[str]
    frame_size_hw: SizeHW
    scale_xy: Pair = (1.0, 1.0)
    offset_xy: Pair = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.frame_id is not None and not (
            isinstance(self.frame_id, str) and self.frame_id
        ):
            raise ValueError(
                f"frame_id must be a non-empty string or None, got {self.frame_id!r}"
            )

        object.__setattr__(self, "frame_size_hw", _size_hw(self.frame_size_hw))
        object.__setattr__(
            self, "scale_xy", _pair(self.scale_xy, name="scale_xy", positive=True)
        )
        object.__setattr__(self, "offset_xy", _pair(self.offset_xy, name="offset_xy"))

    @classmethod
    def identity(cls, frame_id: Optional[str], size_hw: SizeHW) -> "FrameMapping":
        """Map an image onto its own frame.

        Args:
            frame_id: Identity of the image.
            size_hw: Size of the image, ``(height, width)``.

        Returns:
            Mapping with scale 1 and offset 0.
        """
        mapping = cls(frame_id=frame_id, frame_size_hw=size_hw)

        return mapping

    def map_xy(self, x: float, y: float) -> Pair:
        """Map one local point into the reference frame.

        Args:
            x: Local x coordinate.
            y: Local y coordinate.

        Returns:
            ``(x, y)`` in the reference frame.
        """
        scale_x, scale_y = self.scale_xy
        offset_x, offset_y = self.offset_xy
        mapped = (x * scale_x + offset_x, y * scale_y + offset_y)

        return mapped

    def then(self, outer: "FrameMapping") -> "FrameMapping":
        """Compose this mapping with a mapping of its reference frame.

        Args:
            outer: Mapping from this mapping's reference frame into another one.

        Returns:
            Mapping from local coordinates straight into ``outer``'s frame.
        """
        composed = FrameMapping(
            frame_id=outer.frame_id,
            frame_size_hw=outer.frame_size_hw,
            scale_xy=(
                outer.scale_xy[0] * self.scale_xy[0],
                outer.scale_xy[1] * self.scale_xy[1],
            ),
            offset_xy=outer.map_xy(*self.offset_xy),
        )

        return composed

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-friendly form used by serialization.

        Returns:
            ``frame_id``, ``size_hw``, ``scale_xy`` and ``offset_xy``.
        """
        serialized = {
            "frame_id": self.frame_id,
            "size_hw": list(self.frame_size_hw),
            "scale_xy": list(self.scale_xy),
            "offset_xy": list(self.offset_xy),
        }

        return serialized

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrameMapping":
        """Rebuild a mapping from ``to_dict`` output.

        Args:
            value: Mapping with ``frame_id``, ``size_hw``, ``scale_xy`` and
                ``offset_xy``.

        Returns:
            The mapping.

        Raises:
            ValueError: When a key is missing or a value is invalid.
        """
        try:
            mapping = cls(
                frame_id=value["frame_id"],
                frame_size_hw=value["size_hw"],
                scale_xy=value["scale_xy"],
                offset_xy=value["offset_xy"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid frame mapping {value!r}: {error}") from error

        return mapping


@dataclass(frozen=True)
class CompositeSource:
    """One source image placed on a composite canvas, such as a mosaic tile.

    Both mappings take canvas coordinates inside ``canvas_xyxy``.

    Args:
        index: Logical index of the source in the group it came from: one or
            more nonnegative integers.
        image_id: Identity of the source image, a non-empty string.
        canvas_xyxy: Tile rectangle on the canvas, four integers with
            ``x0 < x1`` and ``y0 < y1``; upper bounds are exclusive.
        parent: Canvas coordinates to the source image frame.
        root: Canvas coordinates to the source's root frame.

    Raises:
        TypeError: When ``parent`` or ``root`` is not a ``FrameMapping``.
        ValueError: When the index, image id or rectangle break the contract
            above. Values are checked, never truncated or coerced.
    """

    index: Tuple[int, ...]
    image_id: str
    canvas_xyxy: Rectangle
    parent: FrameMapping
    root: FrameMapping

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _logical_index(self.index))
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError(
                f"image_id must be a non-empty string, got {self.image_id!r}"
            )

        object.__setattr__(self, "canvas_xyxy", _tile_rectangle(self.canvas_xyxy))
        for name in ("parent", "root"):
            if not isinstance(getattr(self, name), FrameMapping):
                raise TypeError(
                    f"{name} must be a FrameMapping, got "
                    f"{type(getattr(self, name)).__name__}"
                )

    @classmethod
    def place(
        cls, source: "ImageData", *, index: Sequence[int], canvas_xyxy: Rectangle
    ) -> "CompositeSource":
        """Describe ``source`` resized into ``canvas_xyxy`` of a canvas.

        Args:
            source: Image drawn into the rectangle.
            index: Logical index of ``source`` in its group.
            canvas_xyxy: Destination rectangle, exclusive upper bounds.

        Returns:
            The placement, with mappings from canvas coordinates to the source
            and to the source's root.

        Raises:
            ValueError: When the rectangle is not four integers with positive
                extent, or the index is invalid.
        """
        x0, y0, x1, y1 = _tile_rectangle(canvas_xyxy)
        scale_xy = (source.width / (x1 - x0), source.height / (y1 - y0))
        to_source = FrameMapping(
            frame_id=source.image_id,
            frame_size_hw=source.size_hw,
            scale_xy=scale_xy,
            offset_xy=(-x0 * scale_xy[0], -y0 * scale_xy[1]),
        )
        placement = cls(
            index=index,
            image_id=source.image_id,
            canvas_xyxy=(x0, y0, x1, y1),
            parent=to_source,
            root=to_source.then(source.root),
        )

        return placement

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-friendly form used by serialization and metadata.

        Returns:
            ``index``, ``image_id``, ``canvas_xyxy``, ``parent`` and ``root``.
        """
        serialized = {
            "index": list(self.index),
            "image_id": self.image_id,
            "canvas_xyxy": list(self.canvas_xyxy),
            "parent": self.parent.to_dict(),
            "root": self.root.to_dict(),
        }

        return serialized

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompositeSource":
        """Rebuild a placement from ``to_dict`` output.

        Args:
            value: Mapping produced by ``to_dict``.

        Returns:
            The placement.

        Raises:
            ValueError: When a key is missing or a value is invalid.
        """
        try:
            placement = cls(
                index=value["index"],
                image_id=value["image_id"],
                canvas_xyxy=value["canvas_xyxy"],
                parent=FrameMapping.from_dict(value["parent"]),
                root=FrameMapping.from_dict(value["root"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid composite source {value!r}: {error}") from error

        return placement


@dataclass(frozen=True, eq=False, repr=False)
class ImageData:
    """An image tensor with its identity and spatial provenance.

    Args:
        tensor_image: ``(channels, height, width)`` ``uint8`` tensor on any
            device; 1 or 3 channels, three channels in RGB order. Not copied.
        image_id: Identity of this pixel frame.
        parent: Mapping into the image this one was cropped or resized from.
        root: Mapping into the workflow input this image comes from.
        video_metadata: V1 ``VideoMetadata`` of the source frame, if any.
        composite_sources: ``None`` for an ordinary image. A tuple, possibly
            empty, when the root frame is a composite canvas; it lists the
            images placed on that canvas.
        parent_row: Detection row this image was cropped for, set by dynamic
            crop with ``with_parent_row``. Crop, resize and ``with_pixels`` keep
            it; inputs and composites have none. Unrelated to the frame ids of
            ``parent`` and ``root``.

    Raises:
        TypeError: When ``tensor_image`` is not a tensor or a record has the
            wrong type.
        ValueError: On a wrong dtype, layout, channel count, empty image or
            empty ``image_id``.
    """

    tensor_image: torch.Tensor
    image_id: str
    parent: FrameMapping
    root: FrameMapping
    video_metadata: Optional["VideoMetadata"] = None
    composite_sources: Optional[Tuple[CompositeSource, ...]] = None
    parent_row: Optional[RowRef] = None

    def __post_init__(self) -> None:
        _check_pixels(self.tensor_image)
        if not isinstance(self.image_id, str) or not self.image_id:
            raise ValueError(
                f"image_id must be a non-empty string, got {self.image_id!r}"
            )

        for name in ("parent", "root"):
            if not isinstance(getattr(self, name), FrameMapping):
                raise TypeError(
                    f"{name} must be a FrameMapping, got "
                    f"{type(getattr(self, name)).__name__}"
                )

        if self.composite_sources is not None:
            sources = tuple(self.composite_sources)
            if not all(isinstance(source, CompositeSource) for source in sources):
                raise TypeError("composite_sources must hold CompositeSource records")
            object.__setattr__(self, "composite_sources", sources)

        if self.video_metadata is not None:
            _check_video_metadata(self.video_metadata)

        if self.parent_row is not None and not isinstance(self.parent_row, RowRef):
            raise TypeError(
                "parent_row must be a RowRef or None, got "
                f"{type(self.parent_row).__name__}"
            )

    @classmethod
    def from_tensor(
        cls,
        tensor_image: torch.Tensor,
        *,
        image_id: Optional[str] = None,
        video_metadata: Optional["VideoMetadata"] = None,
    ) -> "ImageData":
        """Wrap a tensor as a workflow input image, without copying or moving it.

        Args:
            tensor_image: ``(channels, height, width)`` ``uint8`` tensor, 1 or 3
                channels (RGB). Channels-last tensors are rejected, not guessed.
            image_id: Identity of the image; a unique ``input-...`` id when
                omitted.
            video_metadata: V1 ``VideoMetadata`` of the frame, if any.

        Returns:
            An image that is its own parent and root.

        Raises:
            TypeError: When ``tensor_image`` is not a tensor.
            ValueError: On a wrong dtype, layout or channel count.
        """
        _check_pixels(tensor_image)
        image_id = image_id if image_id is not None else _new_id("input")
        size_hw = _pixels_size_hw(tensor_image)

        image = cls(
            tensor_image=tensor_image,
            image_id=image_id,
            parent=FrameMapping.identity(image_id, size_hw),
            root=FrameMapping.identity(image_id, size_hw),
            video_metadata=video_metadata,
        )

        return image

    @classmethod
    def from_numpy_rgb(
        cls,
        array: np.ndarray,
        *,
        image_id: Optional[str] = None,
        video_metadata: Optional["VideoMetadata"] = None,
    ) -> "ImageData":
        """Copy a host NumPy image into a CPU tensor image.

        V1 ``WorkflowImageData.numpy_image`` arrays are BGR; use
        ``from_workflow_image_data`` for those instead.

        Args:
            array: ``uint8`` array, RGB ``(height, width, 3)`` or grayscale
                ``(height, width)``. Later changes to it do not affect the image.
            image_id: Identity of the image; a unique ``input-...`` id when
                omitted.
            video_metadata: V1 ``VideoMetadata`` of the frame, if any.

        Returns:
            An image that is its own parent and root.

        Raises:
            ValueError: On a wrong dtype or shape.
        """
        pixels = _tensor_from_hwc(array, channel_order="RGB")
        image = cls.from_tensor(
            pixels, image_id=image_id, video_metadata=video_metadata
        )

        return image

    @classmethod
    def from_workflow_image_data(cls, image: "WorkflowImageData") -> "ImageData":
        """Adapt a V1 ``WorkflowImageData``, keeping the metadata it has.

        ``image_id`` is V1 ``parent_metadata.parent_id`` (V1's id of the current
        image or crop). The root comes from ``workflow_root_ancestor_metadata``;
        offsets and sizes come from both V1 origins, with scale 1 because V1
        images carry none. V1 does not record the id of a crop's immediate
        parent, so ``parent.frame_id`` is ``None`` for V1 crops. A tensor the
        V1 image already holds is used as is; otherwise its BGR NumPy image is
        converted to an RGB CPU tensor. Absent video metadata stays absent.

        Args:
            image: V1 image container.

        Returns:
            The adapted image.

        Raises:
            ValueError: When the V1 pixels have an unsupported layout.
        """
        if image.is_tensor_materialised():
            pixels = image.tensor_image
        else:
            pixels = _tensor_from_hwc(image.numpy_image, channel_order="BGR")

        image_id = image.parent_metadata.parent_id
        root_id = image.workflow_root_ancestor_metadata.parent_id
        parent_frame_id = image_id if image_id == root_id else None
        # The public video_metadata property invents defaults when V1 has none.
        video_metadata = image._video_metadata

        adapted = cls(
            tensor_image=pixels,
            image_id=image_id,
            parent=_frame_from_v1_origin(
                parent_frame_id, image.parent_metadata.origin_coordinates
            ),
            root=_frame_from_v1_origin(
                root_id, image.workflow_root_ancestor_metadata.origin_coordinates
            ),
            video_metadata=video_metadata,
        )

        return adapted

    @classmethod
    def composite(
        cls,
        tensor_image: torch.Tensor,
        *,
        sources: Iterable[CompositeSource],
        image_id: Optional[str] = None,
    ) -> "ImageData":
        """Wrap a canvas assembled from other images as a new root.

        Args:
            tensor_image: The canvas, same contract as ``from_tensor``.
            sources: Placement of every source image; may be empty.
            image_id: Identity of the canvas; a unique ``mosaic-...`` id when
                omitted.

        Returns:
            A composite image that is its own parent and root.
        """
        canvas = cls.from_tensor(
            tensor_image,
            image_id=image_id if image_id is not None else _new_id("mosaic"),
        )
        composite = replace(canvas, composite_sources=tuple(sources))

        return composite

    @property
    def size_hw(self) -> SizeHW:
        """``(height, width)`` in pixels, read from the tensor shape."""
        return _pixels_size_hw(self.tensor_image)

    @property
    def height(self) -> int:
        """Height in pixels."""
        return int(self.tensor_image.shape[1])

    @property
    def width(self) -> int:
        """Width in pixels."""
        return int(self.tensor_image.shape[2])

    @property
    def channels(self) -> int:
        """Number of channels: 3 for RGB, 1 for grayscale."""
        return int(self.tensor_image.shape[0])

    @property
    def device(self) -> torch.device:
        """Device holding ``tensor_image``."""
        return self.tensor_image.device

    @property
    def is_composite(self) -> bool:
        """Whether the root frame is a composite canvas (see ``composite``)."""
        return self.composite_sources is not None

    def crop(
        self, xyxy: Sequence[int], *, image_id: Optional[str] = None
    ) -> Optional["ImageData"]:
        """Cut a rectangle out of this image.

        Args:
            xyxy: ``(x0, y0, x1, y1)`` integer pixels, upper bounds exclusive.
                The rectangle is clipped to the image.
            image_id: Identity of the crop; a unique ``crop-...`` id when
                omitted.

        Returns:
            The crop as an independent contiguous tensor on the same device,
            or ``None`` when nothing remains after clipping.

        Raises:
            ValueError: When ``xyxy`` is not four integers.
        """
        x0, y0, x1, y1 = _integer_rectangle(xyxy)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.width, x1), min(self.height, y1)
        if x1 <= x0 or y1 <= y0:
            return None

        pixels = self.tensor_image[:, y0:y1, x0:x1].clone(
            memory_format=torch.contiguous_format
        )
        to_parent = FrameMapping(
            frame_id=self.image_id, frame_size_hw=self.size_hw, offset_xy=(x0, y0)
        )
        cropped = self._derived(
            pixels,
            to_parent=to_parent,
            image_id=image_id if image_id is not None else _new_id("crop"),
        )

        return cropped

    def resize(
        self,
        size_hw: SizeHW,
        *,
        interpolation: str = "bilinear",
        image_id: Optional[str] = None,
    ) -> "ImageData":
        """Resample this image to a new size, on its own device.

        ``"nearest"`` takes source pixel ``floor(i * source / target)`` along
        each axis (PyTorch ``"nearest"``). ``"bilinear"`` uses PyTorch bilinear
        interpolation with antialiasing and rounds to ``uint8``. ``"area"``
        reproduces OpenCV ``cv2.INTER_AREA``, the V1 ``image_preprocessing``
        resize: box averaging when shrinking, OpenCV's area-weighted linear law
        when any axis grows. Resampling loses information; only coordinates map
        back exactly.

        Args:
            size_hw: Target ``(height, width)``, positive integers.
            interpolation: ``"bilinear"``, ``"nearest"`` or ``"area"``.
            image_id: Identity of the result; a unique ``resize-...`` id when
                omitted.

        Returns:
            The resized image; its parent scale is the actual size ratio.

        Raises:
            ValueError: On a non-positive size or unknown interpolation.
        """
        target_hw = _size_hw(size_hw)
        if interpolation == "nearest":
            pixels = _resize_nearest(self.tensor_image, target_hw)
        elif interpolation == "bilinear":
            pixels = _resize_bilinear(self.tensor_image, target_hw)
        elif interpolation == "area":
            pixels = resize_area(self.tensor_image, target_hw)
        else:
            raise ValueError(
                f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}"
            )

        height, width = self.size_hw
        to_parent = FrameMapping(
            frame_id=self.image_id,
            frame_size_hw=self.size_hw,
            scale_xy=(width / target_hw[1], height / target_hw[0]),
        )
        resized = self._derived(
            pixels,
            to_parent=to_parent,
            image_id=image_id if image_id is not None else _new_id("resize"),
        )

        return resized

    def with_pixels(self, tensor_image: torch.Tensor) -> "ImageData":
        """Replace the pixels, keeping identity and provenance.

        Use this for operations that change pixel values but not geometry,
        such as inversion or color changes. The tensor is not copied.

        Args:
            tensor_image: New ``(channels, height, width)`` ``uint8`` tensor
                with this image's height and width; 1 or 3 channels.

        Returns:
            A new image sharing ``image_id``, mappings and metadata.

        Raises:
            ValueError: When the height or width differ, or the tensor is not
                a valid image tensor.
        """
        _check_pixels(tensor_image)
        if _pixels_size_hw(tensor_image) != self.size_hw:
            raise ValueError(
                f"with_pixels keeps geometry: expected (height, width) "
                f"{self.size_hw}, got {_pixels_size_hw(tensor_image)}; use crop "
                "or resize to change the size"
            )

        updated = replace(self, tensor_image=tensor_image)

        return updated

    def with_parent_row(self, parent_row: Optional[RowRef]) -> "ImageData":
        """Refer this image to the detection row it was cropped for.

        Dynamic crop sets the immediate source row, replacing the reference
        of an enclosing crop. Pixels, identity and mappings are kept.

        Args:
            parent_row: The source detection row, or ``None`` to clear it.

        Returns:
            A new image with ``parent_row`` set.

        Raises:
            TypeError: When ``parent_row`` is not a ``RowRef`` or ``None``.
        """
        updated = replace(self, parent_row=parent_row)

        return updated

    def copy(self, *, device: Union[None, str, torch.device] = None) -> "ImageData":
        """Copy the pixels into memory this image owns, ready on return.

        Use it where an image outlives the current pulse, such as in block
        history, so the producer's buffer can be reused. The result keeps
        ``image_id``, ``parent``, ``root``, ``video_metadata``,
        ``composite_sources`` and ``parent_row`` (the same immutable records;
        ``video_metadata`` is the same object). Its pixels share no memory
        with this image, also from CPU to CPU.

        Readiness: the caller must be able to read this image now, i.e. its
        producer finished it or the caller's current stream is ordered after
        the producer (the usual contract of a block input). On CUDA the copy
        is queued on the current stream of the image's device, and ``copy``
        waits on an event recorded after it. That wait is the cost of the
        retention boundary: the host blocks until that stream reaches the
        copy, including work queued on it before. Other streams and devices
        are not synchronized. On return the source buffer may be released and
        reused. MPS work runs in one ordered queue, so no wait is added there.

        Args:
            device: ``None`` keeps the image's device; ``"cpu"`` copies to
                host memory. Other devices are rejected.

        Returns:
            An image with the same identity, geometry and metadata and new,
            contiguous pixels.

        Raises:
            ValueError: When ``device`` is neither ``None`` nor the CPU.
        """
        if device is not None and torch.device(device).type != "cpu":
            raise ValueError(
                f"copy keeps the image's device (device=None) or copies to the "
                f"CPU (device='cpu'), got {device!r}"
            )

        if device is None:
            pixels = self.tensor_image.clone(memory_format=torch.contiguous_format)
        else:
            pixels = self.tensor_image.to(
                device="cpu", memory_format=torch.contiguous_format, copy=True
            )
        if self.device.type == "cuda":
            copied = torch.cuda.Event()
            copied.record(torch.cuda.current_stream(self.device))
            copied.synchronize()

        owned = replace(self, tensor_image=pixels)

        return owned

    def prediction_metadata(self) -> Dict[str, Any]:
        """Describe this image for a native prediction's ``image_metadata``.

        Attach the result to every prediction made on this image, so that
        ``frames.frame_of`` returns its frame including source and time.
        Same as ``FrameRef.from_image(image).prediction_metadata()``. Reads no
        pixels.

        Returns:
            A new dict with the V1 keys of ``legacy_prediction_metadata`` and
            the typed record under ``frames.FRAME_RECORD_KEY``. A block that
            edits a V1 frame key afterwards must rewrite the record as well:
            readers reject a record that disagrees with the V1 keys.
        """
        # Imported here: frames imports this module.
        from roboflow_workflows.v2.data.frames import FrameRef

        metadata = FrameRef.from_image(self).prediction_metadata()

        return metadata

    def __repr__(self) -> str:
        return (
            f"ImageData(image_id={self.image_id!r}, size_hw={self.size_hw}, "
            f"channels={self.channels}, device={self.device}, "
            f"root={self.root.frame_id!r}, is_composite={self.is_composite})"
        )

    def _derived(
        self, pixels: torch.Tensor, *, to_parent: FrameMapping, image_id: str
    ) -> "ImageData":
        # A geometric child: new identity, parent is this image, root composed.
        derived = ImageData(
            tensor_image=pixels,
            image_id=image_id,
            parent=to_parent,
            root=to_parent.then(self.root),
            video_metadata=self.video_metadata,
            composite_sources=self.composite_sources,
            parent_row=self.parent_row,
        )

        return derived


def legacy_prediction_metadata(
    *,
    image_id: Optional[str],
    size_hw: SizeHW,
    parent: FrameMapping,
    root: FrameMapping,
    composite_sources: Optional[Tuple[CompositeSource, ...]],
) -> Dict[str, Any]:
    """Write a frame as the V1 ``image_metadata`` keys of native predictions.

    The explicitly legacy writer: no typed record, so no source and time.
    ``FrameRef.prediction_metadata`` and ``ImageData.prediction_metadata``
    write these keys plus the record.

    V1 scales are the reciprocal of ``FrameMapping.scale_xy``, so
    ``root_xy = local_xy / scaling_relative_to_root_parent +
    root_parent_coordinates``. ``parent_id`` keeps its V1 meaning, the id of
    the image; ``parent_frame_id`` names the actual parent frame. Composite
    images add ``is_composite`` and ``composite_sources``.

    Args:
        image_id: Identity of the predicted image.
        size_hw: Its size, ``(height, width)``.
        parent: Mapping into its direct parent.
        root: Mapping into its workflow root.
        composite_sources: Mosaic placements, or ``None`` for an ordinary image.

    Returns:
        A new dict with ``parent_id``, ``parent_frame_id``, ``root_parent_id``,
        ``image_dimensions``, ``parent_dimensions``, ``root_parent_dimensions``,
        ``parent_coordinates``, ``root_parent_coordinates``,
        ``scaling_relative_to_parent`` and ``scaling_relative_to_root_parent``.
        Dimensions are ``[height, width]``; coordinates are ``[x, y]``; a scale
        is a float when both axes agree and ``[x, y]`` otherwise.
    """
    metadata = {
        "parent_id": image_id,
        "parent_frame_id": parent.frame_id,
        "root_parent_id": root.frame_id,
        "image_dimensions": list(size_hw),
        "parent_dimensions": list(parent.frame_size_hw),
        "root_parent_dimensions": list(root.frame_size_hw),
        "parent_coordinates": list(parent.offset_xy),
        "root_parent_coordinates": list(root.offset_xy),
        "scaling_relative_to_parent": _v1_scaling(parent),
        "scaling_relative_to_root_parent": _v1_scaling(root),
    }
    if composite_sources is not None:
        metadata["is_composite"] = True
        metadata["composite_sources"] = [
            source.to_dict() for source in composite_sources
        ]

    return metadata


def _resize_nearest(pixels: torch.Tensor, size_hw: SizeHW) -> torch.Tensor:
    # Index gather: exact for uint8 on every device, unlike F.interpolate.
    height, width = size_hw
    source_height, source_width = pixels.shape[1], pixels.shape[2]
    rows = torch.arange(height, device=pixels.device) * source_height // height
    columns = torch.arange(width, device=pixels.device) * source_width // width
    resized = pixels[:, rows[:, None], columns[None, :]]

    return resized


def _resize_bilinear(pixels: torch.Tensor, size_hw: SizeHW) -> torch.Tensor:
    interpolated = F.interpolate(
        pixels.unsqueeze(0).float(),
        size=size_hw,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    resized = interpolated.squeeze(0).round().clamp(0, 255).to(torch.uint8)

    return resized


def _check_pixels(pixels: Any) -> None:
    if not isinstance(pixels, torch.Tensor):
        raise TypeError(
            f"Image pixels must be a torch.Tensor, got {type(pixels).__name__}"
        )
    if pixels.dtype != torch.uint8:
        raise ValueError(f"Image tensor must have dtype uint8, got {pixels.dtype}")
    if pixels.ndim != 3 or pixels.shape[0] not in SUPPORTED_CHANNELS:
        raise ValueError(
            "Image tensor must have shape (channels, height, width) with "
            f"channels in {SUPPORTED_CHANNELS}, got {tuple(pixels.shape)}; "
            "permute channels-last tensors with tensor.permute(2, 0, 1)"
        )
    if pixels.shape[1] == 0 or pixels.shape[2] == 0:
        raise ValueError(f"Image tensor must not be empty, got {tuple(pixels.shape)}")


def _pixels_size_hw(pixels: torch.Tensor) -> SizeHW:
    size_hw = (int(pixels.shape[1]), int(pixels.shape[2]))

    return size_hw


def _tensor_from_hwc(array: Any, *, channel_order: str) -> torch.Tensor:
    # Host boundary: always one fresh C-contiguous copy, never a view of `array`.
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8:
        raise ValueError(
            f"Expected a uint8 numpy array, got {type(array).__name__} "
            f"with dtype {getattr(array, 'dtype', None)}"
        )
    if array.ndim == 2:
        channels_first = array[None, :, :]
    elif array.ndim == 3 and array.shape[2] == 3:
        channels_first = array.transpose(2, 0, 1)
        if channel_order == "BGR":
            channels_first = channels_first[::-1]
    else:
        raise ValueError(
            f"Expected a (height, width, 3) {channel_order} or (height, width) "
            f"grayscale array, got shape {array.shape}"
        )

    pixels = torch.from_numpy(np.array(channels_first, order="C", copy=True))
    _check_pixels(pixels)

    return pixels


def _frame_from_v1_origin(frame_id: Optional[str], origin: Any) -> FrameMapping:
    # `origin` is a V1 OriginCoordinatesSystem: integer offsets and frame size.
    mapping = FrameMapping(
        frame_id=frame_id,
        frame_size_hw=(origin.origin_height, origin.origin_width),
        offset_xy=(origin.left_top_x, origin.left_top_y),
    )

    return mapping


def _v1_scaling(mapping: FrameMapping) -> Union[float, List[float]]:
    scaling_x, scaling_y = (1.0 / scale for scale in mapping.scale_xy)
    if scaling_x == scaling_y:
        return scaling_x

    return [scaling_x, scaling_y]


def _check_video_metadata(video_metadata: Any) -> None:
    # Imported lazily: the V1 entities module is slow to import.
    from roboflow_workflows.execution_engine.entities.base import VideoMetadata

    if not isinstance(video_metadata, VideoMetadata):
        raise TypeError(
            "video_metadata must be a V1 VideoMetadata, got "
            f"{type(video_metadata).__name__}"
        )


def _integer_rectangle(xyxy: Sequence[int]) -> Rectangle:
    corners = tuple(xyxy) if isinstance(xyxy, (list, tuple)) else ()
    if len(corners) != 4 or not all(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        for value in corners
    ):
        raise ValueError(f"Rectangle {xyxy!r} must be four integers (x0, y0, x1, y1)")

    rectangle = tuple(int(value) for value in corners)

    return rectangle


def _tile_rectangle(xyxy: Sequence[int]) -> Rectangle:
    rectangle = _integer_rectangle(xyxy)
    x0, y0, x1, y1 = rectangle
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"canvas_xyxy {list(rectangle)} must have x0 < x1 and y0 < y1")

    return rectangle


def _logical_index(index: Sequence[int]) -> Tuple[int, ...]:
    parts = tuple(index) if isinstance(index, (list, tuple)) else ()
    if not parts or not all(
        isinstance(part, (int, np.integer)) and not isinstance(part, bool) and part >= 0
        for part in parts
    ):
        raise ValueError(
            f"index must be one or more nonnegative integers, got {index!r}"
        )

    logical_index = tuple(int(part) for part in parts)

    return logical_index


def _size_hw(size_hw: Sequence[int]) -> SizeHW:
    values = tuple(size_hw) if isinstance(size_hw, (list, tuple)) else ()
    if len(values) != 2 or not all(
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and value > 0
        for value in values
    ):
        raise ValueError(
            f"Size must be two positive integers (height, width), got {size_hw!r}"
        )

    size = (int(values[0]), int(values[1]))

    return size


def _pair(value: Sequence[float], *, name: str, positive: bool = False) -> Pair:
    values = tuple(value) if isinstance(value, (list, tuple)) else ()
    valid = len(values) == 2 and all(
        isinstance(number, (int, float, np.integer, np.floating))
        and not isinstance(number, bool)
        and math.isfinite(number)
        and (number > 0 or not positive)
        for number in values
    )
    if not valid:
        requirement = "finite positive" if positive else "finite"
        raise ValueError(f"{name} must be two {requirement} numbers, got {value!r}")

    pair = (float(values[0]), float(values[1]))

    return pair


def _new_id(prefix: str) -> str:
    identifier = f"{prefix}-{uuid.uuid4().hex}"

    return identifier
