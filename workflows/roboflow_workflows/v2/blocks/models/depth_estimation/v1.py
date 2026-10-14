"""Local ordinal depth, normalized independently for each image.

Nearest prediction is 1 and farthest is 0, never meters or comparable across
images. One model serves every image of a call. DA V2 and YOLO26 batch images
of equal size together (one chunk per size). DA V3 runs its three model phases
per image: its batch-wide flip can erase a row's contrast before
normalization. This compatibility policy can reduce throughput.
"""

import sys
from functools import lru_cache
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple, Union

import torch
from pydantic import Field, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import (
    model_request,
    models_cache,
    require_rgb,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.predictions import TENSOR_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase


_BACKENDS = (
    ("depth_anything_v2.depth_anything_v2_hf", "DepthAnythingV2HF", False, False),
    ("depth_anything_v3.depth_anything_v3_torch", "DepthAnythingV3Torch", False, True),
    (
        "yolo26.yolo26_depth_estimation_onnx",
        "YOLO26ForDepthEstimationOnnx",
        True,
        False,
    ),
    ("yolo26.yolo26_depth_estimation_trt", "YOLO26ForDepthEstimationTRT", True, False),
    (
        "yolo26.yolo26_depth_estimation_torch_script",
        "YOLO26ForDepthEstimationTorchScript",
        True,
        False,
    ),
)

class _DepthChunk(NamedTuple):
    """Images of one call that the backend preprocesses together."""

    rows: Tuple[int, ...]
    batch: Any
    metadata: Any


class _DepthInput(NamedTuple):
    """What ``network_input`` hands the later phases of one call."""

    request: ModelRequest
    model: Any
    metric: bool
    chunks: Tuple[_DepthChunk, ...]

def _backend_policy(model: Any, *, request: ModelRequest) -> Tuple[bool, bool]:
    for suffix, name, metric, single_row in _BACKENDS:
        module = sys.modules.get(f"inference_models.models.{suffix}")
        backend = getattr(module, name, None)
        if isinstance(backend, type) and isinstance(model, backend):
            return metric, single_row

    cls = type(model)
    raise TypeError(
        f"depth: model {str(request)!r} resolved to unsupported depth backend "
        f"{cls.__module__}.{cls.__qualname__}; supported classes are "
        "DepthAnythingV2HF, DepthAnythingV3Torch and YOLO26ForDepthEstimation "
        "(Onnx, TRT, TorchScript). Depth direction cannot be inferred safely."
    )

def _normalized_maps(
    maps: Any, *, chunk: _DepthChunk, depth_input: _DepthInput, image: Batch
) -> torch.Tensor:
    request = depth_input.request
    if not isinstance(maps, (list, tuple)) or len(maps) != len(chunk.rows):
        count = len(maps) if isinstance(maps, (list, tuple)) else type(maps).__name__
        raise ValueError(
            f"depth: model {str(request)!r} returned {count} depth maps "
            f"for {len(chunk.rows)} images; expected a list of per-image tensors"
        )

    device = None
    for depth_map, row in zip(maps, chunk.rows):
        size_hw = image.content[row].size_hw
        if (
            not isinstance(depth_map, torch.Tensor)
            or depth_map.ndim != 2
            or tuple(depth_map.shape) != size_hw
            or not depth_map.is_floating_point()
        ):
            raise ValueError(
                f"depth: model {str(request)!r}, row {row}: expected a "
                f"floating depth map of shape {size_hw}, got "
                f"{getattr(depth_map, 'shape', type(depth_map).__name__)}"
            )
        if device is not None and depth_map.device != device:
            raise ValueError(
                f"depth: model {str(request)!r} mixed output devices"
            )

        device = depth_map.device

    depths = torch.stack(maps)
    minimum = depths.amin(dim=(1, 2), keepdim=True)
    maximum = depths.amax(dim=(1, 2), keepdim=True)
    # A single compact host transfer per chunk. Numeric reductions
    # stay on device; scan flags in original-row order for useful diagnostics.
    flags = (
        torch.stack(
            (torch.isfinite(depths).all(dim=(1, 2)), (maximum != minimum).flatten()),
            dim=1,
        )
        .detach()
        .cpu()
        .tolist()
    )
    for row, (finite, varying) in zip(chunk.rows, flags):
        if not finite:
            raise ValueError(
                f"depth: model {str(request)!r}, row {row}: "
                "depth map contains nonfinite values"
            )
        if not varying:
            raise ValueError(
                f"depth: model {str(request)!r}, row {row}: "
                "Depth map has no variation (min equals max)"
            )

    # Metric models produce larger=farther; DA outputs already mean nearer.
    # Negate-then-normalize is V1's metric path; the algebraic form avoids
    # allocating that extra full-frame negation.
    numerator = maximum - depths if depth_input.metric else depths - minimum
    normalized = numerator / (maximum - minimum)

    return normalized

@lru_cache(maxsize=None)
def _viridis_lut(device: torch.device) -> torch.Tensor:
    from matplotlib import colormaps

    colors = colormaps["viridis"](range(256))[:, :3]
    lut = torch.tensor(colors * 255, dtype=torch.uint8, device=device)

    return lut

def _visualization(image: ImageData, *, normalized: torch.Tensor) -> ImageData:
    indices = (normalized * 255.0).to(torch.uint8).long()
    rgb = _viridis_lut(normalized.device)[indices].permute(2, 0, 1).contiguous()
    visualized = image.with_pixels(rgb)

    return visualized

class DepthEstimationBlock(Block):
    """Estimate per-image ordinal proximity with a host-owned local model.

    Supply models through ``ModelCache`` or an exact request-to-instance mapping.
    Optional backend dependencies are needed only by the chosen loaded model.
    """

    type = "roboflow_core/depth_estimation@v1"
    outputs = {
        "normalized_depth": Output(
            TENSOR_KIND,
            source="image",
            description=(
                "Floating (H,W) proximity on the model's device: nearest=1, "
                "farthest=0, independently per image; never meters."
            ),
        ),
        "image": Output(
            IMAGE_KIND,
            source="image",
            description="RGB viridis depth visualization with the input's frame linkage.",
        ),
    }

    class Params(BlockParams):
        model_id: StrictStr | Ref(STRING_KIND) = Field(
            default="depth-anything-v3/small",
            description=(
                "Local model ID/path or selector. Supported resolved backends: "
                "DA V2 HF, DA V3 Torch, YOLO26 depth Onnx/TRT/TorchScript. "
                "Use canonical YOLO26 IDs such as yolo26-pretrains/yolo26n-depth."
            ),
            examples=["depth-anything-v3/small", "yolo26-pretrains/yolo26n-depth"],
        )
        model_package_id: Optional[StrictStr] | Ref(
            STRING_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                "Exact model package to load, one for every image of the step; "
                "None lets the loader choose. A selector may choose it per run, "
                "not per image. A model supplied without a package id rejects a "
                "package. List a model's package ids with "
                'inference_models.AutoModel.describe_model("<model_id>").'
            ),
            examples=[None, "$inputs.model_package_id"],
        )
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to estimate; each map retains its input size.",
            examples=["$inputs.image", "$steps.crop.crops"],
        )

        @field_validator("model_id", "model_package_id")
        @classmethod
        def _reject_empty(cls, value: Optional[str]) -> Optional[str]:
            if value == "":
                raise ValueError("must be a non-empty string")

            return value

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Describe the local model request.

        Args:
            params: Validated step configuration.

        Returns:
            Model resource dependencies, including package selection.
        """
        resources = [
            DependentResource(
                resource_type="model",
                identifier=params.model_id,
                details={
                    "resource": "models",
                    "task_type": "depth-estimation",
                    "model_package_id": params.model_package_id,
                },
            )
        ]

        return resources

    @phase
    def network_input(
        self, image: Batch, model_id: str, model_package_id: Optional[str]
    ) -> Optional[_DepthInput]:
        """Resolve the call's backend, then preprocess shape-safe chunks.

        Args:
            image: Batch of RGB images.
            model_id: Local model ID for every image.
            model_package_id: Package for every image; None lets the loader
                choose.

        Returns:
            The call's model and preprocessed chunks; DAV3 chunks always
            contain one image. ``None`` for an empty batch, which loads no
            model.
        """
        for member in image:
            require_rgb(member, block=self.type)
        request = model_request(model_id, model_package_id, block=self.type)
        if not len(image):
            return None

        model = self._models.get(request)
        metric, single_row = _backend_policy(model, request=request)

        rows_by_chunk: Dict[Any, List[int]] = {}
        for row, member in enumerate(image):
            key = row if single_row else member.size_hw
            rows_by_chunk.setdefault(key, []).append(row)
        chunks = []
        for rows in rows_by_chunk.values():
            batch, metadata = model.pre_process(
                [image.content[row].tensor_image for row in rows],
                input_color_format="rgb",
            )
            chunks.append(
                _DepthChunk(rows=tuple(rows), batch=batch, metadata=metadata)
            )
        depth_input = _DepthInput(
            request=request, model=model, metric=metric, chunks=tuple(chunks)
        )

        return depth_input

    @phase
    def raw_output(self, network_input: Optional[_DepthInput]) -> Tuple[Any, ...]:
        """Run the model forward on each chunk.

        Args:
            network_input: The call's model and preprocessed chunks, or
                ``None`` for an empty batch.

        Returns:
            Raw model output in chunk order.
        """
        if network_input is None:
            return ()

        model = network_input.model
        raw = tuple(model.forward(chunk.batch) for chunk in network_input.chunks)

        return raw

    @phase
    def result(
        self,
        raw_output: Tuple[Any, ...],
        network_input: Optional[_DepthInput],
        image: Batch,
    ) -> List[Dict[str, Any]]:
        """Normalize each chunk's maps, rendering only a demanded image.

        Args:
            raw_output: Raw outputs in chunk order.
            network_input: The call's model and preprocessed chunks, or
                ``None`` for an empty batch.
            image: Original images for geometry and output order.

        Returns:
            One proximity map and, when demanded, visualization per input row.

        Raises:
            ValueError: On malformed, constant or nonfinite depth maps.
        """
        if network_input is None:
            return []

        wants_image = self.wants("image")
        results: List[Any] = [None] * len(image)
        for chunk, raw in zip(network_input.chunks, raw_output):
            maps = network_input.model.post_process(raw, chunk.metadata)
            normalized = _normalized_maps(
                maps, chunk=chunk, depth_input=network_input, image=image
            )
            for row, depth in zip(chunk.rows, normalized):
                result = {"normalized_depth": depth}
                if wants_image:
                    result["image"] = _visualization(
                        image.content[row], normalized=depth
                    )
                results[row] = result

        return results

    def run(
        self,
        *,
        image: Batch,
        model_id: str = "depth-anything-v3/small",
        model_package_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute the same three phases without pipeline scheduling.

        Args:
            image: Batch of RGB images.
            model_id: Local model ID/path for every image.
            model_package_id: Exact package for every image; None lets the
                loader choose.

        Returns:
            Per-image proximity and demanded visualizations in input row order.
        """
        prepared = self.network_input(image, model_id, model_package_id)
        raw = self.raw_output(prepared)
        results = self.result(raw, prepared, image)

        return results
