import numpy as np
import pytest
import torch

from inference_models.models.common.trt import TRTCudaGraphLRUCache


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_cudagraph_cache_reuses_previously_seen_input_shapes(
    yolov8n_640_t4_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    from inference_models import AutoModel

    device = torch.device("cuda:0")
    model = AutoModel.from_pretrained(
        model_id_or_path=yolov8n_640_t4_trt_package,
        device=device,
    )

    pre_processed_single, _ = model.pre_process(dog_image_numpy)
    model._trt_cuda_graph_cache = TRTCudaGraphLRUCache()

    seen_shapes = set()
    capture_outputs = {}
    test_sequence = [1, 2, 1, 4, 2, 1, 4, 3, 3]

    for batch_size in test_sequence:
        batch = pre_processed_single.repeat(batch_size, 1, 1, 1)
        cache_key = (tuple(batch.shape), batch.dtype, device)

        cache_before = model._trt_cuda_graph_cache
        cache_size_before = len(cache_before.cache) if cache_before is not None else 0

        output = model.forward(batch, use_cuda_graph=True)

        cache_after = model._trt_cuda_graph_cache
        assert cache_after is not None
        cache_size_after = len(cache_after.cache)

        if cache_key not in seen_shapes:
            assert cache_size_after == cache_size_before + 1
            seen_shapes.add(cache_key)
            capture_outputs[cache_key] = output.clone()
            continue

        assert cache_size_after == cache_size_before
        assert torch.allclose(capture_outputs[cache_key], output, atol=1e-6)

    assert set(model._trt_cuda_graph_cache.cache.keys()) == seen_shapes


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_cudagraph_output_matches_non_cudagraph_output(
    yolov8n_640_t4_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    from inference_models import AutoModel

    device = torch.device("cuda:0")
    model = AutoModel.from_pretrained(
        model_id_or_path=yolov8n_640_t4_trt_package,
        device=device,
    )
    pre_processed_single, _ = model.pre_process(dog_image_numpy)

    for batch_size in [1, 4]:
        batch = pre_processed_single.repeat(batch_size, 1, 1, 1)

        no_graph = model.forward(batch, use_cuda_graph=False)

        model._trt_cuda_graph_cache = TRTCudaGraphLRUCache()
        capture_graph = model.forward(batch, use_cuda_graph=True)
        replay_graph = model.forward(batch, use_cuda_graph=True)

        assert torch.allclose(no_graph, capture_graph, atol=1e-6)
        assert torch.allclose(no_graph, replay_graph, atol=1e-6)
