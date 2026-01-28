import logging

import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_cudagraph_output_matches_non_cudagraph_output(
    rfdetr_nano_t4_trt_package: str,
    dog_image_numpy: np.ndarray,
    bike_image_numpy: np.ndarray,
) -> None:
    from inference_models import AutoModel

    model = AutoModel.from_pretrained(
        model_id_or_path=rfdetr_nano_t4_trt_package,
        device=torch.device("cuda:0"),
    )

    pre_processed_1, _ = model.pre_process(dog_image_numpy)
    pre_processed_2, _ = model.pre_process(bike_image_numpy)

    outputs = []
    for pre_processed in [pre_processed_1, pre_processed_2]:
        no_graph = model.forward(pre_processed, use_cuda_graph=False)
        model._trt_cuda_graph_state = None
        capture_graph = model.forward(pre_processed, use_cuda_graph=True)
        replay_graph = model.forward(pre_processed, use_cuda_graph=True)

        outputs.append((no_graph, capture_graph, replay_graph))

    for image_outputs in outputs:
        no_graph, capture_graph, replay_graph = image_outputs
        for result_idx in range(2):
            assert torch.allclose(
                no_graph[result_idx],
                capture_graph[result_idx],
                atol=1e-6,
            )
            assert torch.allclose(
                no_graph[result_idx],
                replay_graph[result_idx],
                atol=1e-6,
            )

    # make sure that the allcloses aren't true because of buffer aliasing or something weird
    # outputs should be different between images and the same between execution branches.
    for execution_branch_idx in range(3):
        for result_idx in range(2):
            assert not torch.allclose(
                outputs[0][execution_branch_idx][result_idx],
                outputs[1][execution_branch_idx][result_idx],
                atol=1e-6,
            )

@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_outputs_shapes(
    rfdetr_nano_t4_trt_package: str,
    dog_image_numpy: np.ndarray,
) -> None:
    from inference_models import AutoModel

    model = AutoModel.from_pretrained(
        model_id_or_path=rfdetr_nano_t4_trt_package,
        device=torch.device("cuda:0"),
    )

    pre_processed, _ = model.pre_process(dog_image_numpy)

    output = model.forward(pre_processed, use_cuda_graph=False)

    assert output[0].shape == (1, 300, 4)
    assert output[1].shape == (1, 300, 91)

    output = model.forward(pre_processed, use_cuda_graph=True) # capture

    assert output[0].shape == (1, 300, 4)
    assert output[1].shape == (1, 300, 91)

    output = model.forward(pre_processed, use_cuda_graph=True) # replay

    assert output[0].shape == (1, 300, 4)
    assert output[1].shape == (1, 300, 91)