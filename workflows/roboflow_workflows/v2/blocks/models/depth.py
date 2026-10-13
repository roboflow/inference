"""Local ordinal depth, normalized independently for each image.

Nearest prediction is 1 and farthest is 0, never meters or comparable across
images. DA V2 and YOLO26 batch rows of equal request and image size. DA V3 runs
its three model phases per image: its batch-wide flip can erase a row's contrast
before normalization. This compatibility policy can reduce throughput.
"""

import sys
from functools import lru_cache
from typing import Any, Dict, List, Mapping, NamedTuple, Tuple, Union

import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import (
    ModelIdField,
    ModelSelectionParams,
    RequestField,
    model_dependency,
    models_cache,
    require_rgb,
    rows_by_request,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import TENSOR_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase

# Check actual loaded class objects, never model-id spelling. A loaded backend
# already has its module in sys.modules; importing every optional backend here
# would require Transformers, ONNX and TensorRT even for a supplied model.
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


class _DepthGroup(NamedTuple):
    request: ModelRequest
    model: Any
    rows: Tuple[int, ...]
    batch: Any
    metadata: Any
    metric: bool


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


def _normalized_maps(maps: Any, *, group: _DepthGroup, image: Batch) -> torch.Tensor:
    if not isinstance(maps, (list, tuple)) or len(maps) != len(group.rows):
        count = len(maps) if isinstance(maps, (list, tuple)) else type(maps).__name__
        raise ValueError(
            f"depth: model {str(group.request)!r} returned {count} depth maps "
            f"for {len(group.rows)} images; expected a list of per-image tensors"
        )

    device = None
    for depth_map, row in zip(maps, group.rows):
        size_hw = image.content[row].size_hw
        if (
            not isinstance(depth_map, torch.Tensor)
            or depth_map.ndim != 2
            or tuple(depth_map.shape) != size_hw
            or not depth_map.is_floating_point()
        ):
            raise ValueError(
                f"depth: model {str(group.request)!r}, row {row}: expected a "
                f"floating depth map of shape {size_hw}, got "
                f"{getattr(depth_map, 'shape', type(depth_map).__name__)}"
            )
        if device is not None and depth_map.device != device:
            raise ValueError(
                f"depth: model {str(group.request)!r} mixed output devices"
            )

        device = depth_map.device

    depths = torch.stack(maps)
    minimum = depths.amin(dim=(1, 2), keepdim=True)
    maximum = depths.amax(dim=(1, 2), keepdim=True)
    # A single compact host transfer per effective group. Numeric reductions
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
    for row, (finite, varying) in zip(group.rows, flags):
        if not finite:
            raise ValueError(
                f"depth: model {str(group.request)!r}, row {row}: "
                "depth map contains nonfinite values"
            )
        if not varying:
            raise ValueError(
                f"depth: model {str(group.request)!r}, row {row}: "
                "Depth map has no variation (min equals max)"
            )

    # Metric models produce larger=farther; DA outputs already mean nearer.
    # Negate-then-normalize is V1's metric path; the algebraic form avoids
    # allocating that extra full-frame negation.
    numerator = maximum - depths if group.metric else depths - minimum
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

    type = "v2/depth_estimation@v1"
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

    class Params(ModelSelectionParams):
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to estimate; each map retains its input size.",
            examples=["$inputs.image", "$steps.crop.crops"],
        )
        model_id: ModelIdField = Field(
            default="depth-anything-v3/small",
            description=(
                "Local model ID/path or selector. Supported resolved backends: "
                "DA V2 HF, DA V3 Torch, YOLO26 depth Onnx/TRT/TorchScript. "
                "Use canonical YOLO26 IDs such as yolo26-pretrains/yolo26n-depth."
            ),
            examples=["depth-anything-v3/small", "yolo26-pretrains/yolo26n-depth"],
        )

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
        resources = model_dependency(params, task_type="depth-estimation")

        return resources

    @phase
    def network_input(
        self, image: Batch, model_id: RequestField, model_package_id: RequestField
    ) -> Tuple[_DepthGroup, ...]:
        """Resolve supported backends before preprocessing shape-safe groups.

        Args:
            image: Batch of RGB images.
            model_id: Shared or per-image local model ID.
            model_package_id: Shared or per-image package selection.

        Returns:
            Preprocessed groups; DAV3 groups always contain one row.
        """
        for member in image:
            require_rgb(member, block=self.type)
        requests = rows_by_request(
            len(image), model_id=model_id, model_package_id=model_package_id
        )
        models = {request: self._models.get(request) for request in requests}
        policies = {
            request: _backend_policy(model, request=request)
            for request, model in models.items()
        }
        groups = []
        for request, rows in requests.items():
            model = models[request]
            metric, single_row = policies[request]
            rows_by_shape = {}
            for row in rows:
                key = row if single_row else image.content[row].size_hw
                rows_by_shape.setdefault(key, []).append(row)
            for group_rows in rows_by_shape.values():
                batch, metadata = model.pre_process(
                    [image.content[row].tensor_image for row in group_rows],
                    input_color_format="rgb",
                )
                groups.append(
                    _DepthGroup(
                        request, model, tuple(group_rows), batch, metadata, metric
                    )
                )
        prepared = tuple(groups)

        return prepared

    @phase
    def raw_output(self, network_input: Tuple[_DepthGroup, ...]) -> Tuple[Any, ...]:
        """Run each group's model forward.

        Args:
            network_input: Preprocessed depth groups.

        Returns:
            Raw model output in group order.
        """
        raw = tuple(group.model.forward(group.batch) for group in network_input)

        return raw

    @phase
    def result(
        self,
        raw_output: Tuple[Any, ...],
        network_input: Tuple[_DepthGroup, ...],
        image: Batch,
    ) -> List[Dict[str, Any]]:
        """Normalize and scatter model maps, rendering only a demanded image.

        Args:
            raw_output: Raw outputs in group order.
            network_input: Prepared model groups.
            image: Original images for geometry and output order.

        Returns:
            One proximity map and, when demanded, visualization per input row.

        Raises:
            ValueError: On malformed, constant or nonfinite depth maps.
        """
        wants_image = self.wants("image")
        results: List[Any] = [None] * len(image)
        for group, raw in zip(network_input, raw_output):
            maps = group.model.post_process(raw, group.metadata)
            normalized = _normalized_maps(maps, group=group, image=image)
            for row, depth in zip(group.rows, normalized):
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
        model_id: RequestField = "depth-anything-v3/small",
        model_package_id: RequestField = None,
    ) -> List[Dict[str, Any]]:
        """Execute the same three phases without pipeline scheduling.

        Args:
            image: Batch of RGB images.
            model_id: Local model ID/path or per-image selections.
            model_package_id: Exact package or per-image selections; None lets the loader choose.

        Returns:
            Per-image proximity and demanded visualizations in input row order.
        """
        prepared = self.network_input(image, model_id, model_package_id)
        raw = self.raw_output(prepared)
        results = self.result(raw, prepared, image)

        return results
