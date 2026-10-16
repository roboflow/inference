"""Validate the final Windows torch build and execute its native operators."""

import importlib.metadata
import json
import os

import click

# Windows cp312 wheels verified on the official PyTorch CPU / cu126 indexes.
# cu128 stops at torch 2.11, whose setuptools cap conflicts with this repo.
# cu126 keeps both flavors on torch 2.14.1 without weakening shared floors.
TORCH_BUILDS = {
    "cpu": ("2.14.1+cpu", "0.29.1+cpu", None),
    "gpu": ("2.14.1+cu126", "0.29.1+cu126", "12.6"),
}


def check_build_identity(
    flavor: str,
    *,
    torch_version: str,
    vision_version: str,
    cuda_version: str | None,
    distributions: tuple[str, str],
) -> None:
    """Reject wrong build flavors and mismatched installed metadata.

    Args:
        flavor: Requested installer flavor, cpu or gpu.
        torch_version: Imported torch version.
        vision_version: Imported torchvision version.
        cuda_version: CUDA version compiled into torch, or None.
        distributions: Installed torch and torchvision distribution versions.

    Raises:
        RuntimeError: Any version or CUDA build differs from the pinned pair.
    """
    expected = TORCH_BUILDS[flavor]
    actual = (torch_version, vision_version, cuda_version)
    if actual != expected or distributions != expected[:2]:
        raise RuntimeError(
            f"Wrong {flavor} torch build: imported={actual}, metadata={distributions}; "
            f"expected={expected}. Reinstall the explicit Windows wheel pair."
        )


def check_native_ops(flavor: str, *, require_gpu: bool = False) -> dict:
    """Execute NMS and image decoding with real installed native libraries.

    Args:
        flavor: Requested installer flavor, cpu or gpu.
        require_gpu: Also require CUDA NMS execution on NVIDIA hardware.

    Returns:
        Validated build versions, CUDA capability and native operation results.

    Raises:
        RuntimeError: A wrong build, missing native operator or GPU is detected.
    """
    # New torchvision versions expose the otherwise suppressed loader exception.
    # Older versions ignore this flag; their import traceback is still retained.
    os.environ["TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS"] = "1"
    import torch
    import torchvision
    from torchvision.io import decode_image, encode_png
    from torchvision.ops import nms

    check_build_identity(
        flavor,
        torch_version=torch.__version__,
        vision_version=torchvision.__version__,
        cuda_version=torch.version.cuda,
        distributions=(
            importlib.metadata.version("torch"),
            importlib.metadata.version("torchvision"),
        ),
    )
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]]).float()
    scores = torch.tensor([0.9, 0.8, 0.7])
    kept = nms(boxes, scores, 0.5).tolist()
    if kept != [0, 2]:
        raise RuntimeError(f"Unexpected CPU NMS result: {kept}")

    image = torch.arange(18, dtype=torch.uint8).reshape(3, 2, 3)
    if not torch.equal(decode_image(encode_png(image)), image):
        raise RuntimeError("Native PNG decode did not preserve the fixture")

    cuda_available = torch.cuda.is_available()
    if require_gpu:
        if flavor != "gpu" or not cuda_available:
            raise RuntimeError("Windows NVIDIA qualification requires an available GPU")

        if nms(boxes.cuda(), scores.cuda(), 0.5).cpu().tolist() != [0, 2]:
            raise RuntimeError("Unexpected CUDA NMS result")

    result = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_nms_executed": require_gpu,
        "cpu_nms": kept,
        "png_roundtrip": True,
    }
    return result


@click.command()
@click.option(
    "--flavor",
    type=click.Choice(["cpu", "gpu"]),
    required=True,
)
@click.option(
    "--require-gpu",
    is_flag=True,
    help="Require actual CUDA NMS execution; use only on NVIDIA hardware.",
)
def main(flavor: str, require_gpu: bool) -> None:
    """Check native operations in a fresh interpreter or frozen executable.

    Args:
        flavor: Requested CPU or GPU installer flavor.
        require_gpu: Require real CUDA NMS execution.
    """
    result = check_native_ops(flavor, require_gpu=require_gpu)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
