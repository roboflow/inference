import json

import numpy as np
import onnx
import pytest
import torch
from onnx import helper, numpy_helper

from inference_models.errors import CorruptedModelPackageError, ModelInputError
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.models_registry import resolve_model_class
from inference_models.models.pretrained_image_embeddings.preprocessing import (
    image_processor_config_from_dict,
    pre_process_images,
)
from inference_models.models.pretrained_image_embeddings.pretrained_image_embeddings_onnx import (
    PretrainedImageEmbeddingOnnx,
)

DINOV2_SMALL_PREPROCESSOR = {
    "crop_size": {"height": 224, "width": 224},
    "do_center_crop": True,
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_processor_type": "BitImageProcessor",
    "image_std": [0.229, 0.224, 0.225],
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {"shortest_edge": 256},
}
DINOV3_VITS16_PREPROCESSOR = {
    "crop_size": None,
    "data_format": "channels_first",
    "default_to_square": True,
    "device": None,
    "disable_grouping": None,
    "do_center_crop": None,
    "do_convert_rgb": None,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_processor_type": "DINOv3ViTImageProcessorFast",
    "image_std": [0.229, 0.224, 0.225],
    "input_data_format": None,
    "resample": 2,
    "rescale_factor": 0.00392156862745098,
    "return_tensors": None,
    "size": {"height": 224, "width": 224},
}
SIGLIP2_BASE_PREPROCESSOR = {
    "do_convert_rgb": None,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "SiglipImageProcessor",
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "SiglipProcessor",
    "resample": 2,
    "rescale_factor": 0.00392156862745098,
    "size": {"height": 224, "width": 224},
}
SMALL_PREPROCESSOR = {
    **SIGLIP2_BASE_PREPROCESSOR,
    "size": {"height": 16, "width": 16},
}
TOKENS = 3
DIMENSION = 4


def build_graph(
    path,
    dtype=np.float32,
    image_size=(16, 16),
    external_data=False,
):
    """Token j of image i is the channel means of image i times weight block j;
    the pooled output is the mean over tokens."""
    weight = np.random.default_rng(0).normal(size=(3, TOKENS * DIMENSION))
    weight = weight.astype(dtype)
    element_type = helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
    height, width = image_size or ("height", "width")
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "ReduceMean", ["pixel_values"], ["means"], axes=[2, 3], keepdims=0
            ),
            helper.make_node("MatMul", ["means", "weight"], ["flat"]),
            helper.make_node("Reshape", ["flat", "shape"], ["last_hidden_state"]),
            helper.make_node(
                "ReduceMean",
                ["last_hidden_state"],
                ["pooler_output"],
                axes=[1],
                keepdims=0,
            ),
        ],
        name="embeddings",
        inputs=[
            helper.make_tensor_value_info(
                "pixel_values", element_type, ["batch", 3, height, width]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "last_hidden_state",
                element_type,
                ["batch", TOKENS, DIMENSION],
            ),
            helper.make_tensor_value_info(
                "pooler_output", element_type, ["batch", DIMENSION]
            ),
        ],
        initializer=[
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(
                np.array([-1, TOKENS, DIMENSION], dtype=np.int64), "shape"
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    if external_data:
        onnx.save(
            model,
            str(path),
            save_as_external_data=True,
            location="model.onnx_data",
            # Keeps the reshape target inline, as ONNX Runtime needs it for shape
            # inference; the weights go to the sidecar.
            size_threshold=64,
        )
    else:
        onnx.save(model, str(path))
    return weight.astype(np.float64)


def write_package(
    directory,
    output_name="last_hidden_state",
    token_index=0,
    dimension=DIMENSION,
    preprocessor=SMALL_PREPROCESSOR,
    **graph_kwargs,
):
    directory.mkdir(exist_ok=True)
    weight = build_graph(directory / "model.onnx", **graph_kwargs)
    (directory / "preprocessor_config.json").write_text(json.dumps(preprocessor))
    (directory / "embedding_config.json").write_text(
        json.dumps(
            {
                "output_name": output_name,
                "token_index": token_index,
                "dimension": dimension,
                "feature_definition": "test-cls-token@v1",
                "source": None,
            }
        )
    )
    return weight


def load(directory, **kwargs):
    return PretrainedImageEmbeddingOnnx.from_pretrained(
        str(directory),
        onnx_execution_providers=["CPUExecutionProvider"],
        device=torch.device("cpu"),
        **kwargs,
    )


def expected_embeddings(model, weight, images, token_index):
    means = model.pre_process(images).double().mean(dim=(2, 3)).numpy()
    tokens = (means @ weight).reshape(len(means), TOKENS, -1)
    selected = tokens.mean(axis=1) if token_index is None else tokens[:, token_index]
    return torch.from_numpy(selected).float()


def random_images(count, seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, (20 + 3 * i, 31 - 2 * i, 3), dtype=np.uint8)
        for i in range(count)
    ]


@pytest.mark.parametrize(
    "output_name,token_index",
    [("last_hidden_state", 0), ("last_hidden_state", 2), ("pooler_output", None)],
)
def test_configured_output_and_token_become_the_embedding(
    tmp_path, output_name, token_index
):
    weight = write_package(tmp_path, output_name=output_name, token_index=token_index)
    model = load(tmp_path)
    images = random_images(2)

    embeddings = model.embed_images(images)

    assert embeddings.shape == (2, DIMENSION)
    assert embeddings.dtype == torch.float32
    torch.testing.assert_close(
        embeddings, expected_embeddings(model, weight, images, token_index)
    )


def test_fp16_graph_produces_float32_embeddings(tmp_path):
    weight = write_package(tmp_path, dtype=np.float16)
    model = load(tmp_path)
    images = random_images(2)

    embeddings = model.embed_images(images)

    assert embeddings.dtype == torch.float32
    torch.testing.assert_close(
        embeddings,
        expected_embeddings(model, weight, images, token_index=0),
        atol=2e-2,
        rtol=1e-2,
    )


def test_batches_beyond_max_batch_size_run_in_chunks(tmp_path):
    weight = write_package(tmp_path)
    model = load(tmp_path, max_batch_size=2)
    batch_sizes = []
    run = model._session.run

    def counting_run(output_names, input_feed):
        batch_sizes.append(len(input_feed["pixel_values"]))
        return run(output_names, input_feed)

    model._session.run = counting_run
    images = random_images(5)

    embeddings = model.embed_images(images)

    assert batch_sizes == [2, 2, 1]
    torch.testing.assert_close(
        embeddings, expected_embeddings(model, weight, images, token_index=0)
    )


@pytest.mark.parametrize(
    "package_kwargs",
    [
        {"dimension": DIMENSION + 1},
        {"output_name": "missing"},
        {"output_name": "pooler_output", "token_index": 0},
        {"image_size": (32, 32)},
    ],
    ids=["dimension", "output-name", "token-on-pooled-output", "input-size"],
)
def test_package_disagreeing_with_its_graph_is_rejected(tmp_path, package_kwargs):
    write_package(tmp_path, **package_kwargs)

    with pytest.raises(CorruptedModelPackageError):
        load(tmp_path)


def test_only_feature_vectors_are_produced(tmp_path):
    write_package(tmp_path)
    model = load(tmp_path)

    model.prepare_image_embeddings(output_type="feature_vector")
    with pytest.raises(ModelInputError):
        model.prepare_image_embeddings(output_type="logits")
    with pytest.raises(ModelInputError):
        model.embed_images(random_images(1), output_type="logits")
    with pytest.raises(ModelInputError):
        model.get_embedding_info("logits")


def test_embedding_info_and_effective_preprocessing(tmp_path):
    write_package(tmp_path)
    model = load(tmp_path)

    assert model.get_embedding_info() == {
        "feature_definition": "test-cls-token@v1",
        "output_type": "feature_vector",
        "normalization": "none",
    }
    assert model.embedding_info == model.get_embedding_info("feature_vector")
    assert json.loads(json.dumps(model.embedding_preprocessing)) == {
        "image_processor_type": "SiglipImageProcessor",
        "do_convert_rgb": False,
        "do_resize": True,
        "size": {"height": 16, "width": 16},
        "resample": 2,
        "do_center_crop": False,
        "crop_size": None,
        "do_rescale": True,
        "rescale_factor": 0.00392156862745098,
        "do_normalize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }


def test_graph_with_external_weights_loads(tmp_path):
    weight = write_package(tmp_path, external_data=True)
    assert (tmp_path / "model.onnx_data").stat().st_size >= weight.size * 4
    model = load(tmp_path)
    images = random_images(1)

    torch.testing.assert_close(
        model.embed_images(images),
        expected_embeddings(model, weight, images, token_index=0),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected_key": True},
        {"image_processor_type": "CLIPImageProcessor"},
        {"resample": 1},
        {"size": {"shortest_edge": 224, "longest_edge": 448}},
        {"do_center_crop": False},
        {"do_pad": True},
        {"return_tensors": "pt"},
    ],
)
def test_unsupported_preprocessor_configs_are_rejected(change):
    with pytest.raises(CorruptedModelPackageError):
        image_processor_config_from_dict({**DINOV2_SMALL_PREPROCESSOR, **change})


def synthetic_rgb_image(height, width, seed):
    y, x = np.mgrid[0:height, 0:width]
    channels = [
        (np.sin(x / 17.0 + c) + np.cos(y / 23.0 - c)) * 60 + 127 for c in range(3)
    ]
    noise = np.random.default_rng(seed).normal(0, 25, (height, width, 3))
    return np.clip(np.stack(channels, axis=-1) + noise, 0, 255).astype(np.uint8)


# transformers' torchvision processors run the same torchvision operations, so only
# float32 rounding differs. Its PIL processors resample in two separable passes with
# fixed-point weights and round to uint8 after each pass, where torch rounds once:
# rare pixels land up to two 8-bit levels away, i.e. 2 / 255 / std once normalised.
@pytest.mark.parametrize(
    "config,processor_name,pil",
    [
        (DINOV2_SMALL_PREPROCESSOR, "BitImageProcessor", False),
        (DINOV2_SMALL_PREPROCESSOR, "BitImageProcessorPil", True),
        (DINOV3_VITS16_PREPROCESSOR, "DINOv3ViTImageProcessor", False),
        (SIGLIP2_BASE_PREPROCESSOR, "SiglipImageProcessor", False),
        (SIGLIP2_BASE_PREPROCESSOR, "SiglipImageProcessorPil", True),
    ],
)
@pytest.mark.parametrize("image_size", [(301, 457), (517, 263), (190, 333)])
def test_preprocessing_matches_transformers(config, processor_name, pil, image_size):
    transformers = pytest.importorskip("transformers")
    processor_class = getattr(transformers, processor_name, None)
    if processor_class is None:
        pytest.skip(f"transformers has no {processor_name}")
    image = synthetic_rgb_image(*image_size, seed=sum(image_size))
    reference = processor_class.from_dict(config)(images=image, return_tensors="pt")
    reference = reference["pixel_values"]
    parsed = image_processor_config_from_dict(config)
    device = torch.device("cpu")

    result = pre_process_images(image, parsed, device, input_color_format="rgb")

    assert result.shape == reference.shape
    assert result.dtype == torch.float32
    atol = 2 / 255 / min(config["image_std"]) + 1e-5 if pil else 1e-5
    torch.testing.assert_close(result, reference, atol=atol, rtol=0)
    assert (result - reference).abs().mean() < 1e-3
    assert torch.equal(
        pre_process_images(np.ascontiguousarray(image[:, :, ::-1]), parsed, device),
        result,
    )
    assert torch.equal(
        pre_process_images(torch.from_numpy(image).permute(2, 0, 1), parsed, device),
        result,
    )


@pytest.mark.parametrize("architecture", ["dinov2", "dinov3", "siglip2"])
def test_registry_resolves_pretrained_embedding_architectures(architecture):
    model_class = resolve_model_class(
        model_architecture=architecture,
        task_type="embedding",
        backend=BackendType.ONNX,
    )

    assert model_class is PretrainedImageEmbeddingOnnx
