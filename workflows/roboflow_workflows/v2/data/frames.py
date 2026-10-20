"""Typed frame of a native prediction: which image it describes, and where.

Native predictions keep the V1 ``image_metadata`` keys written by
``image.legacy_prediction_metadata``. ``FrameRef`` reads them back as typed
mappings and adds one optional, versioned, JSON-only record, which
``ImageData.prediction_metadata()`` and ``FrameRef.prediction_metadata()``
write next to the V1 keys::

    image_metadata["workflows_v2_frame"] = {
        "version": 1,
        "image_id": "crop-1",
        "size_hw": [5, 10],
        "parent": FrameMapping.to_dict(),     # local -> direct parent
        "root": FrameMapping.to_dict(),       # local -> workflow input
        "composite_sources": [...] | None,    # mosaic placements
        "video_metadata": {...} | None,       # source and time, when known
    }

The record carries what the legacy keys cannot: source and time
(``video_metadata``) and exact, non-reciprocal scales. Legacy keys store the
reciprocal scale (local pixels per frame pixel). This module is the only
reader of the legacy keys: it accepts Python or numpy numbers and inverts
the scales once, so every ``FrameMapping.scale_xy`` counts frame pixels per
local pixel.

``frame_of`` prefers neither source silently. When a record and legacy keys
are both present, every legacy key must agree with the record, otherwise it
raises. Without a record it rebuilds the frame from the legacy keys; missing
geometry raises, unknown ids stay ``None`` and an absent ``video_metadata``
means unknown. Geometry is never guessed from mask or grid shapes.

Root conversion reads less than a full frame. ``root_placement_from_metadata``
reads the root mapping, plus the local size and id when present. From legacy
keys alone it needs only ``root_parent_coordinates``,
``root_parent_dimensions`` and ``root_parent_id``; ``frame_of`` also needs
``image_dimensions`` and the parent keys, but not the ids. Historical V1
metadata without ``image_dimensions`` or parent keys therefore converts to
the root but has no ``FrameRef``. A typed record always gives both.

``require_compatible_frame`` serves a block that already holds the image a
prediction or prompt must describe. Like root conversion, it needs no full
frame: it validates every frame field the metadata supplies, compares it with
the image and rejects a known difference. A missing field stays unknown, so passing
means "no known conflict", not "same image".

A composite (mosaic) root is a canvas, not one source: ``root`` maps to the
canvas and ``composite_sources`` lists the placements. In legacy keys,
``is_composite`` is a boolean flag: ``False`` marks an ordinary image and
``None`` leaves it unset. Model letterbox or
static-crop geometry is not part of the frame; it belongs to the model's mask
recipe. No function here reads pixels or tensor values, or copies tensors.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import (
    CompositeSource,
    FrameMapping,
    ImageData,
    SizeHW,
    _check_video_metadata,
    _size_hw,
    legacy_prediction_metadata,
)

from inference_models.models.base.classification import ClassificationPrediction

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.entities.base import VideoMetadata

FRAME_RECORD_KEY = "workflows_v2_frame"
FRAME_RECORD_VERSION = 1

_RECORD_KEYS = {
    "version",
    "image_id",
    "size_hw",
    "parent",
    "root",
    "composite_sources",
    "video_metadata",
}
_REQUIRED_LEGACY_KEYS = (
    "image_dimensions",
    "parent_dimensions",
    "root_parent_dimensions",
    "parent_coordinates",
    "root_parent_coordinates",
)
_REQUIRED_ROOT_KEYS = (
    "root_parent_coordinates",
    "root_parent_dimensions",
    "root_parent_id",
)
_ROOT_GEOMETRY_KEYS = (
    "root_parent_coordinates",
    "root_parent_dimensions",
    "scaling_relative_to_root_parent",
)
_SCALING_KEYS = ("scaling_relative_to_parent", "scaling_relative_to_root_parent")
_COMPOSITE_ROOT_MESSAGE = (
    "Prediction comes from a composite image, such as a stitched panorama or "
    "a mosaic, which has no single workflow input to map it into. To output "
    'it, set "coordinates_system": "own" on the JsonField output for '
    "coordinates in the composite image's pixels. To crop or draw, use the "
    "composite image itself."
)
# Legacy floats are written from the record, or are reciprocals of its scales.
_RELATIVE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FrameRef:
    """The image a prediction describes and its place in the workflow input.

    Args:
        image_id: Identity of the predicted image; ``None`` only when imported
            data cannot name it.
        size_hw: Size of the predicted image, ``(height, width)``.
        parent: Mapping of local coordinates into the direct parent image.
        root: Mapping of local coordinates into the workflow input, or into
            the canvas of a composite.
        composite_sources: ``None`` for an ordinary image; the placements on
            the canvas when the root is a composite.
        video_metadata: V1 ``VideoMetadata`` (source and time) when known;
            ``None`` means unknown, not frame zero.

    Raises:
        ContractError: When a field has the wrong type or value.
    """

    image_id: Optional[str]
    size_hw: SizeHW
    parent: FrameMapping
    root: FrameMapping
    composite_sources: Optional[Tuple[CompositeSource, ...]] = None
    video_metadata: Optional["VideoMetadata"] = None

    def __post_init__(self) -> None:
        if self.image_id is not None and not (
            isinstance(self.image_id, str) and self.image_id
        ):
            raise ContractError(
                f"image_id must be a non-empty string or None, got {self.image_id!r}"
            )

        try:
            object.__setattr__(self, "size_hw", _size_hw(self.size_hw))
            if self.video_metadata is not None:
                _check_video_metadata(self.video_metadata)
        except (TypeError, ValueError) as error:
            raise ContractError(f"Invalid frame: {error}") from error

        for name in ("parent", "root"):
            if not isinstance(getattr(self, name), FrameMapping):
                raise ContractError(
                    f"{name} must be a FrameMapping, got "
                    f"{type(getattr(self, name)).__name__}"
                )

        if self.composite_sources is not None:
            sources = tuple(self.composite_sources)
            if not all(isinstance(source, CompositeSource) for source in sources):
                raise ContractError(
                    "composite_sources must hold CompositeSource records"
                )
            object.__setattr__(self, "composite_sources", sources)

    @classmethod
    def from_image(cls, image: ImageData) -> "FrameRef":
        """Describe an image without reading its pixels.

        Args:
            image: The image a model or block predicts on.

        Returns:
            Its frame, including video metadata when the image has it.
        """
        frame = cls(
            image_id=image.image_id,
            size_hw=image.size_hw,
            parent=image.parent,
            root=image.root,
            composite_sources=image.composite_sources,
            video_metadata=image.video_metadata,
        )

        return frame

    @classmethod
    def from_record(cls, record: Any) -> "FrameRef":
        """Rebuild a frame from ``to_record`` output.

        Args:
            record: The JSON-compatible record.

        Returns:
            The frame.

        Raises:
            ContractError: On an unknown version, missing or extra keys, or an
                invalid value.
        """
        if not isinstance(record, Mapping):
            raise ContractError(
                f"Frame record must be a dict, got {type(record).__name__}"
            )

        version = record.get("version")
        if type(version) is not int or version != FRAME_RECORD_VERSION:
            raise ContractError(
                f"Frame record version must be {FRAME_RECORD_VERSION}, got "
                f"{version!r}"
            )
        if set(record) != _RECORD_KEYS:
            raise ContractError(
                f"Frame record keys must be {sorted(_RECORD_KEYS)}; missing "
                f"{sorted(_RECORD_KEYS - set(record))}, unexpected "
                f"{sorted(set(record) - _RECORD_KEYS)}"
            )

        try:
            sources = record["composite_sources"]
            frame = cls(
                image_id=record["image_id"],
                size_hw=record["size_hw"],
                parent=FrameMapping.from_dict(record["parent"]),
                root=FrameMapping.from_dict(record["root"]),
                composite_sources=(
                    None
                    if sources is None
                    else tuple(CompositeSource.from_dict(item) for item in sources)
                ),
                video_metadata=_video_metadata_from_json(record["video_metadata"]),
            )
        except (TypeError, ValueError) as error:
            raise ContractError(f"Invalid frame record: {error}") from error

        return frame

    @property
    def is_composite(self) -> bool:
        """Whether the root is a composite canvas with no single source."""
        return self.composite_sources is not None

    def to_record(self) -> Dict[str, Any]:
        """Return the versioned JSON-compatible record of this frame.

        Returns:
            A new dict; see the module docstring.
        """
        record = {
            "version": FRAME_RECORD_VERSION,
            "image_id": self.image_id,
            "size_hw": list(self.size_hw),
            "parent": self.parent.to_dict(),
            "root": self.root.to_dict(),
            "composite_sources": (
                None
                if self.composite_sources is None
                else [source.to_dict() for source in self.composite_sources]
            ),
            "video_metadata": (
                None
                if self.video_metadata is None
                else self.video_metadata.model_dump(mode="json")
            ),
        }

        return record

    def prediction_metadata(self) -> Dict[str, Any]:
        """Describe this frame for a native prediction's ``image_metadata``.

        Returns:
            A new dict with the V1 keys of ``legacy_prediction_metadata``
            plus the typed record under ``FRAME_RECORD_KEY``.
        """
        metadata = legacy_prediction_metadata(
            image_id=self.image_id,
            size_hw=self.size_hw,
            parent=self.parent,
            root=self.root,
            composite_sources=self.composite_sources,
        )
        metadata[FRAME_RECORD_KEY] = self.to_record()

        return metadata


@dataclass(frozen=True)
class RootPlacement:
    """Where a prediction's local pixels sit in its workflow root.

    The part of a frame that root conversion reads. See the module docstring
    for why legacy metadata can give a placement but no ``FrameRef``.

    Args:
        image_id: Identity of the predicted image; ``None`` when unknown.
        size_hw: Size of the predicted image; ``None`` only when legacy
            metadata omits ``image_dimensions``.
        root: Mapping of local coordinates into the workflow root.
        video_metadata: Source and time from the typed record; ``None`` when
            unknown.
    """

    image_id: Optional[str]
    size_hw: Optional[SizeHW]
    root: FrameMapping
    video_metadata: Optional["VideoMetadata"] = None

    @property
    def is_root(self) -> bool:
        """Whether the prediction already describes its workflow root."""
        root = self.root
        same_pixels = (
            root.offset_xy == (0.0, 0.0)
            and root.scale_xy == (1.0, 1.0)
            and self.size_hw in (None, root.frame_size_hw)
        )
        is_root = same_pixels and self.image_id in (None, root.frame_id)

        return is_root


def frame_of(prediction: Any) -> FrameRef:
    """Read the frame a native prediction describes.

    Args:
        prediction: ``Detections``, ``InstanceDetections``, ``KeyPoints``, a
            ``(KeyPoints, Optional[Detections])`` tuple,
            ``SemanticSegmentationResult``, a one-image
            ``ClassificationPrediction`` or ``MultiLabelClassificationPrediction``.

    Returns:
        The frame from the typed record, or rebuilt from the legacy keys.

    Raises:
        ContractError: When the prediction has no image metadata, required
            geometry is missing or invalid, or the record and the legacy keys
            disagree. For a keypoint tuple, also when the boxes describe
            another frame than the keypoints.
    """
    if isinstance(prediction, tuple) and len(prediction) == 2:
        key_points, detections = prediction
        frame = frame_of(key_points)
        if detections is not None and detections.image_metadata is not None:
            if frame_of(detections) != frame:
                raise ContractError(
                    "KeyPoints and their Detections describe different frames"
                )
        return frame

    metadata = _image_metadata_of(prediction)
    frame = frame_from_metadata(metadata)

    return frame


def frame_from_metadata(metadata: Mapping[str, Any]) -> FrameRef:
    """Read a frame from one native ``image_metadata`` dict.

    Args:
        metadata: ``image_metadata`` with the typed record, the legacy frame
            keys, or both.

    Returns:
        The frame from the record, or rebuilt from the legacy keys.

    Raises:
        ContractError: When required geometry is missing or invalid, or the
            record and the legacy keys disagree.
    """
    record = metadata.get(FRAME_RECORD_KEY)
    if record is None:
        frame = _frame_from_legacy_keys(metadata)
        return frame

    frame = FrameRef.from_record(record)
    _check_legacy_keys_agree(metadata, frame=frame)

    return frame


def root_placement_from_metadata(
    metadata: Mapping[str, Any],
) -> Optional[RootPlacement]:
    """Read where a native prediction sits in its workflow root.

    Args:
        metadata: ``image_metadata`` with the typed record, the legacy frame
            keys, or both.

    Returns:
        The placement of the record's frame when a record is present.
        Otherwise the placement read from the legacy root keys, or ``None``
        when none of them is present.

    Raises:
        ContractError: When the record or the legacy root keys are missing
            values, invalid or disagree, or when the frame is a composite
            (stitched panorama or mosaic), which has no single workflow root.
    """
    if metadata.get(FRAME_RECORD_KEY) is not None:
        frame = frame_from_metadata(metadata)
        if frame.is_composite:
            raise ContractError(_COMPOSITE_ROOT_MESSAGE)
        placement = RootPlacement(
            image_id=frame.image_id,
            size_hw=frame.size_hw,
            root=frame.root,
            video_metadata=frame.video_metadata,
        )
        return placement

    if _legacy_is_composite(metadata):
        raise ContractError(_COMPOSITE_ROOT_MESSAGE)
    present = [key for key in _ROOT_GEOMETRY_KEYS if key in metadata]
    if not present:
        return None

    for key in _REQUIRED_ROOT_KEYS:
        if key not in metadata:
            raise ContractError(
                f"Incomplete root frame metadata: '{key}' is missing while "
                f"{present} are set"
            )

    image_dimensions = metadata.get("image_dimensions")
    placement = RootPlacement(
        image_id=metadata.get("parent_id"),
        size_hw=(
            None
            if image_dimensions is None
            else _legacy_size(image_dimensions, key="image_dimensions")
        ),
        root=_legacy_root_mapping(metadata),
    )

    return placement


def require_compatible_frame(
    metadata: Optional[Mapping[str, Any]], *, image: ImageData
) -> None:
    """Check that a prediction's metadata may describe ``image``.

    For a block that receives an image and a prediction or prompt made on it.
    Every frame field the metadata supplies is validated, then compared with
    the image when the image knows it too. Compared fields: image id and
    size; frame id, size, scale and offset of the parent and root mappings;
    whether the root is a composite, and its placements; video source, frame
    number and timestamp. Ids, integers, flags and times must be equal;
    scales and offsets may differ by a relative ``1e-9``. Legacy keys next to
    a typed record are validated as if alone, then must match the record.

    A missing field is unknown and never proves or disproves a match:
    ``metadata=None``; a legacy key that is absent or ``None``, including a
    legacy scale, which ``frame_of`` would read as ``1.0``; a ``None`` id or
    ``video_metadata`` in the record; an image whose parent id or video
    metadata is ``None``. Legacy keys alone carry no source and time. Passing
    therefore means no known conflict; matching sizes do not prove identity.

    Args:
        metadata: ``image_metadata`` of the prediction or prompt, with the
            typed record, legacy frame keys, both or neither.
        image: The image the block works on. Its pixels are not read.

    Raises:
        ContractError: When ``metadata`` is not a dict or a supplied frame
            field is malformed, when the record and the legacy keys disagree,
            or when a field both sides know differs.
    """
    if metadata is None:
        return

    if not isinstance(metadata, Mapping):
        raise ContractError(
            f"image_metadata must be a dict, got {type(metadata).__name__}"
        )

    # Legacy keys are read the same way with or without a record.
    legacy = _legacy_frame_fields(metadata)
    record = metadata.get(FRAME_RECORD_KEY)
    if record is None:
        known = legacy
    else:
        known = _frame_fields(FrameRef.from_record(record))
        for name, value in legacy.items():
            # An absent or None mirror is unknown. A supplied mirror must
            # match the record, also where the record's id is None.
            if value is not None and not _same_value(
                value, known[name], exact_integers=True
            ):
                raise ContractError(
                    f"Conflicting frame: image_metadata gives {name} {value!r} but "
                    f"the {FRAME_RECORD_KEY} record gives {known[name]!r}"
                )
    image_fields = _frame_fields(FrameRef.from_image(image))

    for name, value in known.items():
        expected = image_fields[name]
        if value is None or expected is None:
            continue
        if not _same_value(value, expected, exact_integers=True):
            raise ContractError(
                f"Frame mismatch: image_metadata gives {name} {value!r}, but "
                f"image {image.image_id!r} has {expected!r}"
            )


def _frame_fields(frame: FrameRef) -> Dict[str, Any]:
    # Comparable fields of a whole frame; None marks an unknown value.
    video = frame.video_metadata
    fields = {
        "image_id": frame.image_id,
        "size_hw": list(frame.size_hw),
        **_mapping_fields(frame.parent, name="parent"),
        **_mapping_fields(frame.root, name="root"),
        "is_composite": frame.is_composite,
        "composite_sources": (
            None
            if frame.composite_sources is None
            else [source.to_dict() for source in frame.composite_sources]
        ),
        "video_identifier": None if video is None else video.video_identifier,
        "frame_number": None if video is None else video.frame_number,
        "frame_timestamp": None if video is None else video.frame_timestamp,
    }

    return fields


def _mapping_fields(mapping: FrameMapping, *, name: str) -> Dict[str, Any]:
    fields = {
        f"{name}.frame_id": mapping.frame_id,
        f"{name}.size_hw": list(mapping.frame_size_hw),
        f"{name}.scale_xy": list(mapping.scale_xy),
        f"{name}.offset_xy": list(mapping.offset_xy),
    }

    return fields


def _legacy_frame_fields(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the legacy frame keys that are present, in ``_frame_fields`` terms.

    No key is required. Each present key is validated by the same reader as
    in ``frame_of``; an absent or ``None`` key gives ``None``, unknown.
    """
    try:
        is_composite, composite_sources = _legacy_composite_fields(metadata)
        fields = {
            "image_id": _legacy_id(metadata, key="parent_id"),
            "size_hw": _legacy_known(
                metadata, key="image_dimensions", read=_legacy_size
            ),
            "parent.frame_id": _legacy_id(metadata, key="parent_frame_id"),
            "parent.size_hw": _legacy_known(
                metadata, key="parent_dimensions", read=_legacy_size
            ),
            "parent.scale_xy": _legacy_known(
                metadata, key="scaling_relative_to_parent", read=_frame_scale
            ),
            "parent.offset_xy": _legacy_known(
                metadata, key="parent_coordinates", read=_legacy_pair
            ),
            "root.frame_id": _legacy_id(metadata, key="root_parent_id"),
            "root.size_hw": _legacy_known(
                metadata, key="root_parent_dimensions", read=_legacy_size
            ),
            "root.scale_xy": _legacy_known(
                metadata, key="scaling_relative_to_root_parent", read=_frame_scale
            ),
            "root.offset_xy": _legacy_known(
                metadata, key="root_parent_coordinates", read=_legacy_pair
            ),
            "is_composite": is_composite,
            "composite_sources": composite_sources,
        }
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"Invalid frame metadata in image_metadata: {error}"
        ) from error

    return fields


def _legacy_id(metadata: Mapping[str, Any], *, key: str) -> Optional[str]:
    value = metadata.get(key)
    if value is not None and not (isinstance(value, str) and value):
        raise ContractError(
            f"'{key}' must be a non-empty string or None, got {value!r}"
        )

    return value


def _legacy_known(
    metadata: Mapping[str, Any], *, key: str, read: Callable[..., Tuple[Any, Any]]
) -> Optional[List[Any]]:
    value = metadata.get(key)
    if value is None:
        return None

    known = list(read(value, key=key))

    return known


def _legacy_composite_fields(
    metadata: Mapping[str, Any],
) -> Tuple[Optional[bool], Optional[List[Dict[str, Any]]]]:
    # (is_composite, placements); no flag and no placements leave both unknown.
    if not _legacy_is_composite(metadata):
        flag = _legacy_composite_flag(metadata)
        return flag, None

    if metadata.get("composite_sources") is None:
        return True, None

    sources = [source.to_dict() for source in _composite_sources_from_legacy(metadata)]

    return True, sources


def _image_metadata_of(prediction: Any) -> Mapping[str, Any]:
    if isinstance(prediction, ClassificationPrediction):
        images_metadata = prediction.images_metadata
        if not images_metadata or len(images_metadata) != 1:
            raise ContractError(
                "frame_of needs a ClassificationPrediction of exactly one image "
                "with images_metadata"
            )
        metadata = images_metadata[0]
    elif hasattr(prediction, "image_metadata"):
        metadata = prediction.image_metadata
    else:
        raise ContractError(
            f"frame_of does not support {type(prediction).__name__}; expected a "
            "native inference_models prediction"
        )

    if metadata is None:
        raise ContractError(
            f"{type(prediction).__name__} has no image_metadata; attach "
            "FrameRef.prediction_metadata() of the image it was predicted on"
        )
    if not isinstance(metadata, Mapping):
        raise ContractError(
            f"image_metadata must be a dict, got {type(metadata).__name__}"
        )

    return metadata


def _frame_from_legacy_keys(metadata: Mapping[str, Any]) -> FrameRef:
    missing = [key for key in _REQUIRED_LEGACY_KEYS if key not in metadata]
    if missing:
        raise ContractError(
            f"image_metadata lacks frame geometry {missing}; attach "
            "FrameRef.prediction_metadata() of the predicted image"
        )

    try:
        parent = _legacy_mapping(
            metadata,
            id_key="parent_frame_id",
            size_key="parent_dimensions",
            offset_key="parent_coordinates",
            scale_key="scaling_relative_to_parent",
        )
        frame = FrameRef(
            image_id=metadata.get("parent_id"),
            size_hw=_legacy_size(metadata["image_dimensions"], key="image_dimensions"),
            parent=parent,
            root=_legacy_root_mapping(metadata),
            composite_sources=_composite_sources_from_legacy(metadata),
        )
    except (TypeError, ValueError) as error:
        raise ContractError(
            f"Invalid frame geometry in image_metadata: {error}"
        ) from error

    return frame


def _legacy_root_mapping(metadata: Mapping[str, Any]) -> FrameMapping:
    mapping = _legacy_mapping(
        metadata,
        id_key="root_parent_id",
        size_key="root_parent_dimensions",
        offset_key="root_parent_coordinates",
        scale_key="scaling_relative_to_root_parent",
    )

    return mapping


def _legacy_mapping(
    metadata: Mapping[str, Any],
    *,
    id_key: str,
    size_key: str,
    offset_key: str,
    scale_key: str,
) -> FrameMapping:
    mapping = FrameMapping(
        frame_id=metadata.get(id_key),
        frame_size_hw=_legacy_size(metadata[size_key], key=size_key),
        scale_xy=_frame_scale(metadata.get(scale_key), key=scale_key),
        offset_xy=_legacy_pair(metadata[offset_key], key=offset_key),
    )

    return mapping


def _frame_scale(value: Any, *, key: str) -> Tuple[float, float]:
    # Legacy keys hold local pixels per frame pixel, the reciprocal of scale_xy.
    local_per_frame_x, local_per_frame_y = _legacy_scale(value, key=key)
    scale_xy = (1.0 / local_per_frame_x, 1.0 / local_per_frame_y)

    return scale_xy


def _legacy_scale(value: Any, *, key: str) -> Tuple[float, float]:
    """Read local pixels per frame pixel: a number or an ``[x, y]`` pair.

    A missing value means ``1.0``.
    """
    if value is None:
        return 1.0, 1.0

    if isinstance(value, (list, tuple, np.ndarray)):
        pair = _legacy_pair(value, key=key)
    else:
        number = _legacy_real(value, key=key)
        pair = (number, number)
    if min(pair) <= 0:
        raise ContractError(
            f"legacy scaling '{key}' must be positive (local pixels per frame "
            f"pixel), got {value!r}"
        )

    return pair


def _legacy_size(value: Any, *, key: str) -> SizeHW:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ContractError(f"'{key}' must be a [height, width] pair, got {value!r}")

    numbers = [_legacy_real(item, key=key) for item in value]
    if not all(number.is_integer() and number > 0 for number in numbers):
        raise ContractError(
            f"'{key}' must hold positive integers [height, width], got {value!r}"
        )

    size = (int(numbers[0]), int(numbers[1]))

    return size


def _legacy_pair(value: Any, *, key: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ContractError(f"'{key}' must be an [x, y] pair, got {value!r}")

    pair = (_legacy_real(value[0], key=key), _legacy_real(value[1], key=key))

    return pair


def _legacy_real(value: Any, *, key: str) -> float:
    """Read one Python or numpy real number; booleans are not numbers."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"'{key}' must hold real numbers, got {value!r}")
    if not math.isfinite(value):
        raise ContractError(f"'{key}' must hold finite numbers, got {value!r}")

    return float(value)


def _composite_sources_from_legacy(
    metadata: Mapping[str, Any],
) -> Optional[Tuple[CompositeSource, ...]]:
    if not _legacy_is_composite(metadata):
        return None

    sources = metadata.get("composite_sources")
    if not isinstance(sources, list):
        raise ValueError(
            "a composite prediction needs its 'composite_sources' list, got "
            f"{type(sources).__name__}"
        )

    placements = tuple(CompositeSource.from_dict(source) for source in sources)

    return placements


def _legacy_is_composite(metadata: Mapping[str, Any]) -> bool:
    """Read whether the legacy keys mark a composite.

    ``is_composite=True`` or any ``composite_sources`` other than ``None``,
    even an empty list, marks a composite. ``is_composite=False`` marks an
    ordinary image and contradicts ``composite_sources``.
    """
    flag = _legacy_composite_flag(metadata)
    has_sources = metadata.get("composite_sources") is not None
    if flag is False and has_sources:
        raise ContractError(
            "image_metadata sets 'is_composite' to False but also sets "
            "'composite_sources'"
        )

    is_composite = flag is True or has_sources

    return is_composite


def _legacy_composite_flag(metadata: Mapping[str, Any]) -> Optional[bool]:
    """Read ``is_composite``: a Python or numpy boolean, ``None`` when unset."""
    flag = metadata.get("is_composite")
    if isinstance(flag, np.bool_):
        flag = bool(flag)
    if flag is not None and not isinstance(flag, bool):
        raise ContractError(f"'is_composite' must be a boolean, got {flag!r}")

    return flag


def _check_legacy_keys_agree(metadata: Mapping[str, Any], *, frame: FrameRef) -> None:
    expected = legacy_prediction_metadata(
        image_id=frame.image_id,
        size_hw=frame.size_hw,
        parent=frame.parent,
        root=frame.root,
        composite_sources=frame.composite_sources,
    )
    flag = _legacy_composite_flag(metadata)
    if flag is not None and flag != frame.is_composite:
        raise ContractError(
            f"Conflicting frame: image_metadata 'is_composite' is {flag!r} but "
            f"the {FRAME_RECORD_KEY} record describes "
            f"{'a' if frame.is_composite else 'no'} composite"
        )
    if not frame.is_composite and metadata.get("composite_sources") is not None:
        raise ContractError(
            "Conflicting frame: image_metadata sets 'composite_sources' but the "
            f"{FRAME_RECORD_KEY} record describes no composite"
        )

    for key, expected_value in expected.items():
        # The flag is checked above; None or absent leaves it unset.
        if key not in metadata or key == "is_composite":
            continue

        actual = metadata[key]
        if key in _SCALING_KEYS:
            actual, expected_value = _scale_pair(actual), _scale_pair(expected_value)
        if not _same_value(actual, expected_value):
            raise ContractError(
                f"Conflicting frame: image_metadata '{key}' is {actual!r} but the "
                f"{FRAME_RECORD_KEY} record gives {expected_value!r}"
            )


def _scale_pair(value: Any) -> Any:
    if _is_sequence(value):
        return list(value)

    pair = [value, value]

    return pair


def _is_sequence(value: Any) -> bool:
    # A 0-d numpy array has no length: it is a malformed scalar, not a pair.
    is_sequence = isinstance(value, (list, tuple)) or (
        isinstance(value, np.ndarray) and value.ndim > 0
    )

    return is_sequence


def _same_value(actual: Any, expected: Any, *, exact_integers: bool = False) -> bool:
    """Compare a metadata value with the expected one.

    Floats match within ``_RELATIVE_TOLERANCE``. Integers do too, unless
    ``exact_integers`` is set: frame numbers and sizes are discrete, and a
    relative tolerance would merge adjacent large values.
    """
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, (int, float)):
        if isinstance(actual, np.generic):
            actual = actual.item()
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        if exact_integers and isinstance(expected, int):
            return actual == expected

        same = math.isclose(actual, expected, rel_tol=_RELATIVE_TOLERANCE)
        return same
    if isinstance(expected, list):
        same = (
            _is_sequence(actual)
            and len(actual) == len(expected)
            and all(
                _same_value(a, e, exact_integers=exact_integers)
                for a, e in zip(actual, expected)
            )
        )
        return same
    if isinstance(expected, dict):
        same = (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(
                _same_value(actual[key], expected[key], exact_integers=exact_integers)
                for key in expected
            )
        )
        return same

    return actual == expected


def _video_metadata_from_json(value: Any) -> Optional["VideoMetadata"]:
    if value is None:
        return None

    # Imported lazily: the V1 entities module is slow to import.
    from roboflow_workflows.execution_engine.entities.base import VideoMetadata

    try:
        metadata = VideoMetadata.model_validate(value)
    except Exception as error:
        raise ValueError(f"invalid video_metadata: {error}") from error

    return metadata


__all__ = [
    "FRAME_RECORD_KEY",
    "FRAME_RECORD_VERSION",
    "FrameRef",
    "RootPlacement",
    "frame_from_metadata",
    "frame_of",
    "require_compatible_frame",
    "root_placement_from_metadata",
]
