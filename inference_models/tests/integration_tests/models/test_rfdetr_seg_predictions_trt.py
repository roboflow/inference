import numpy as np
import pytest
import torch


@pytest.mark.slow
@pytest.mark.trt_extras
def test_trt_cudagraph_output_matches_non_cudagraph_output(
    rfdetr_seg_nano_t4_trt_package: str,
    snake_image_numpy: np.ndarray,
    dog_image_numpy: np.ndarray,
) -> None:
    from inference_models import AutoModel

    model = AutoModel.from_pretrained(
        model_id_or_path=rfdetr_seg_nano_t4_trt_package,
        device=torch.device("cuda:0"),
    )

    pre_processed_1, _ = model.pre_process(snake_image_numpy)
    pre_processed_2, _ = model.pre_process(dog_image_numpy)

    outputs = []
    for pre_processed in [pre_processed_1, pre_processed_2]:
        no_graph = model.forward(pre_processed, use_cuda_graph=False)
        model._trt_cuda_graph_cache = None
        capture_graph = model.forward(pre_processed, use_cuda_graph=True)
        replay_graph = model.forward(pre_processed, use_cuda_graph=True)

        outputs.append((no_graph, capture_graph, replay_graph))

    for image_outputs in outputs:
        no_graph, capture_graph, replay_graph = image_outputs
        for result_idx in range(3):
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

    for execution_branch_idx in range(3):
        for result_idx in range(3):
            assert not torch.allclose(
                outputs[0][execution_branch_idx][result_idx],
                outputs[1][execution_branch_idx][result_idx],
                atol=1e-6,
            )
