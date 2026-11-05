"""Build inference_models packages for the pretrained image-embedding models.

Downloads each model's ONNX graph and Hugging Face `preprocessor_config.json` at a
pinned revision, lays them out the way `PretrainedImageEmbeddingOnnx` loads them,
verifies the package and prints the size and sha256 of every file:

    <output-dir>/<model-id>/
        model.onnx                  graph, renamed from the repository file
        model.onnx_data             external weights, only when the graph has them
        preprocessor_config.json    verbatim from the repository
        embedding_config.json       output tensor, token, dimension, feature definition
        model_config.json           architecture, task and backend for local loading

Verification fails loudly when the graph lacks the expected output, references
external data that was not downloaded, when the preprocessing differs from what the
package was designed for, or when loading the package and embedding one image does
not give a vector of the declared dimension.

    python build_pretrained_image_embedding_packages.py --output-dir ./packages
    python build_pretrained_image_embedding_packages.py --output-dir ./packages \\
        --models dinov2/small siglip2/base-patch16-224

DINOv3 weights are gated on Hugging Face: accept the DINOv3 License on the model
page and export `HF_TOKEN` before building `dinov3/vits16`.
"""

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import torch
from huggingface_hub import hf_hub_download

from inference_models.models.pretrained_image_embeddings.preprocessing import (
    image_processor_config_from_dict,
)
from inference_models.models.pretrained_image_embeddings.pretrained_image_embeddings_onnx import (
    EmbeddingConfig,
    PretrainedImageEmbeddingOnnx,
)

IMAGENET_NORMALIZATION = {
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
}


@dataclass(frozen=True)
class PackageSpec:
    architecture: str
    repository: str
    revision: str
    graph_file: str
    output_name: str
    token_index: Optional[int]
    dimension: int
    feature_definition: str
    expected_preprocessing: dict
    external_data_files: Tuple[str, ...] = ()


PACKAGES: Dict[str, PackageSpec] = {
    "dinov2/small": PackageSpec(
        architecture="dinov2",
        repository="onnx-community/dinov2-small",
        revision="8b1f705a3a7f6f062f6bdd21986c1583d3ef105d",
        graph_file="onnx/model_fp16.onnx",
        output_name="last_hidden_state",
        token_index=0,
        dimension=384,
        feature_definition="dinov2-cls-token@v1",
        expected_preprocessing={
            "size": {"shortest_edge": 256},
            "resample": 3,
            "crop_size": {"height": 224, "width": 224},
            **IMAGENET_NORMALIZATION,
        },
    ),
    "dinov3/vits16": PackageSpec(
        architecture="dinov3",
        repository="onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX",
        revision="48988dfe73065df8d6f5ccc0edc7c8bcf307de41",
        graph_file="onnx/model.onnx",
        external_data_files=("onnx/model.onnx_data",),
        output_name="last_hidden_state",
        token_index=0,
        dimension=384,
        feature_definition="dinov3-cls-token@v1",
        expected_preprocessing={
            "size": {"height": 224, "width": 224},
            "resample": 2,
            "do_center_crop": False,
            **IMAGENET_NORMALIZATION,
        },
    ),
    "siglip2/base-patch16-224": PackageSpec(
        architecture="siglip2",
        repository="onnx-community/siglip2-base-patch16-224-ONNX",
        revision="ba1f3b0843f24bc5417d38e19c37b287d719b2f4",
        graph_file="onnx/vision_model_fp16.onnx",
        output_name="pooler_output",
        token_index=None,
        dimension=768,
        feature_definition="siglip2-pooler-output@v1",
        expected_preprocessing={
            "size": {"height": 224, "width": 224},
            "resample": 2,
            "do_center_crop": False,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        },
    ),
}


def main() -> int:
    args = _parse_args()
    for model_id in args.models:
        package_dir = os.path.join(args.output_dir, model_id)
        build_package(
            model_id=model_id,
            spec=PACKAGES[model_id],
            package_dir=package_dir,
            overwrite=args.overwrite,
            onnx_execution_providers=args.onnx_execution_providers,
        )
    return 0


def build_package(
    model_id: str,
    spec: PackageSpec,
    package_dir: str,
    overwrite: bool,
    onnx_execution_providers: List[str],
) -> None:
    if os.path.exists(package_dir):
        if not overwrite:
            raise SystemExit(f"{package_dir} exists; pass --overwrite to rebuild it")
        shutil.rmtree(package_dir)
    os.makedirs(package_dir)
    print(f"[{model_id}] {spec.repository}@{spec.revision}")
    _download(spec, spec.graph_file, os.path.join(package_dir, "model.onnx"))
    for data_file in spec.external_data_files:
        target = os.path.join(package_dir, os.path.basename(data_file))
        _download(spec, data_file, target)
    preprocessor_path = os.path.join(package_dir, "preprocessor_config.json")
    _download(spec, "preprocessor_config.json", preprocessor_path)
    _verify_graph(spec=spec, package_dir=package_dir)
    _verify_preprocessing(spec=spec, preprocessor_path=preprocessor_path)
    embedding_config = EmbeddingConfig.model_validate(
        {
            "output_name": spec.output_name,
            "token_index": spec.token_index,
            "dimension": spec.dimension,
            "feature_definition": spec.feature_definition,
            "source": {
                "repository": spec.repository,
                "revision": spec.revision,
                "file": spec.graph_file,
            },
        }
    )
    _write_json(
        os.path.join(package_dir, "embedding_config.json"),
        embedding_config.model_dump(mode="json"),
    )
    _write_json(
        os.path.join(package_dir, "model_config.json"),
        {
            "model_architecture": spec.architecture,
            "task_type": "embedding",
            "backend_type": "onnx",
        },
    )
    _verify_package_embeds(
        spec=spec,
        package_dir=package_dir,
        onnx_execution_providers=onnx_execution_providers,
    )
    for file_name in sorted(os.listdir(package_dir)):
        path = os.path.join(package_dir, file_name)
        print(f"  {file_name}  {os.path.getsize(path)} bytes  sha256={_sha256(path)}")


def _download(spec: PackageSpec, repository_file: str, target: str) -> None:
    downloaded = hf_hub_download(
        repo_id=spec.repository, filename=repository_file, revision=spec.revision
    )
    shutil.copyfile(downloaded, target)


def _verify_graph(spec: PackageSpec, package_dir: str) -> None:
    graph_path = os.path.join(package_dir, "model.onnx")
    model = onnx.load(graph_path, load_external_data=False)
    outputs = [output.name for output in model.graph.output]
    print(f"  graph outputs: {outputs}")
    if spec.output_name not in outputs:
        raise SystemExit(f"graph has no output `{spec.output_name}`: {outputs}")
    referenced = {
        entry.value
        for tensor in model.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        for entry in tensor.external_data
        if entry.key == "location"
    }
    downloaded = {os.path.basename(path) for path in spec.external_data_files}
    if referenced != downloaded:
        raise SystemExit(
            f"graph references external data {sorted(referenced)}, "
            f"downloaded {sorted(downloaded)}"
        )


def _verify_preprocessing(spec: PackageSpec, preprocessor_path: str) -> None:
    with open(preprocessor_path) as file:
        effective = image_processor_config_from_dict(json.load(file)).describe()
    print(f"  preprocessing: {effective}")
    differences = {
        key: (effective[key], expected)
        for key, expected in spec.expected_preprocessing.items()
        if effective[key] != expected
    }
    if differences:
        raise SystemExit(f"unexpected preprocessing (found, expected): {differences}")


def _verify_package_embeds(
    spec: PackageSpec, package_dir: str, onnx_execution_providers: List[str]
) -> None:
    model = PretrainedImageEmbeddingOnnx.from_pretrained(
        package_dir,
        onnx_execution_providers=onnx_execution_providers,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    image = np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    embeddings = model.embed_images(image)
    if tuple(embeddings.shape) != (1, spec.dimension):
        raise SystemExit(
            f"package embeds to shape {tuple(embeddings.shape)}, "
            f"expected (1, {spec.dimension})"
        )


def _write_json(path: str, content: dict) -> None:
    with open(path, "w") as file:
        json.dump(content, file, indent=2)
        file.write("\n")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(PACKAGES),
        default=sorted(PACKAGES),
        help="Model ids to build (default: all).",
    )
    parser.add_argument(
        "--onnx-execution-providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="Providers for the load-and-embed check, e.g. CUDAExecutionProvider "
        "where the CPU provider lacks float16 kernels a graph needs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild packages whose directory already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
