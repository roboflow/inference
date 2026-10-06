import sys
from unittest import mock

from inference.core.entities.responses import server_state
from inference.core.managers.entities import ModelDescription


def _fake_torch(
    *,
    free: int = 20,
    total: int = 100,
    allocated: int = 45,
    reserved: int = 60,
) -> mock.MagicMock:
    torch = mock.MagicMock(name="torch")
    torch.cuda.is_available.return_value = True
    torch.cuda.mem_get_info.return_value = (free, total)
    torch.cuda.memory_allocated.return_value = allocated
    torch.cuda.memory_reserved.return_value = reserved
    return torch


def test_get_gpu_memory_stats_includes_pytorch_allocator_breakdown() -> None:
    with mock.patch.dict(sys.modules, {"torch": _fake_torch()}):
        assert server_state._get_gpu_memory_stats() == (80, 100, 45, 60)


def test_models_descriptions_derives_allocator_and_non_torch_memory() -> None:
    model = ModelDescription(
        model_id="example/1",
        task_type="object-detection",
        batch_size=1,
        input_height=640,
        input_width=640,
        vram_bytes=25,
    )

    with mock.patch.object(
        server_state,
        "_get_gpu_memory_stats",
        return_value=(80, 100, 45, 60),
    ):
        response = server_state.ModelsDescriptions.from_models_descriptions([model])

    assert response.total_vram_bytes == 25
    assert response.gpu_memory_used == 80
    assert response.torch_cuda_allocated == 45
    assert response.torch_cuda_reserved == 60
    assert response.torch_cuda_allocator_cache == 15
    assert response.non_torch_gpu_memory == 20


def test_memory_breakdown_is_optional_without_cuda() -> None:
    with mock.patch.object(
        server_state,
        "_get_gpu_memory_stats",
        return_value=(None, None, None, None),
    ):
        response = server_state.ModelsDescriptions.from_models_descriptions([])

    assert response.torch_cuda_allocated is None
    assert response.torch_cuda_reserved is None
    assert response.torch_cuda_allocator_cache is None
    assert response.non_torch_gpu_memory is None


def test_memory_differences_are_clamped_when_samples_are_inconsistent() -> None:
    assert server_state._non_negative_difference(10, 20) == 0

