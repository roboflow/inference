import shutil
from collections import Counter
from pathlib import Path

import onnx
import pytest

from inference_models.models.common.onnx_attention_fusion import (
    fuse_attention_for_cuda_execution,
)

CUDA_PROVIDERS = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]


@pytest.mark.slow
@pytest.mark.onnx_extras
@pytest.mark.parametrize(
    "package_fixture, expected_fused_blocks",
    [
        ("coin_counting_rfdetr_nano_onnx_cs_stretch_package", 14),
        ("snakes_rfdetr_seg_onnx_static_bs_stretch_package", 16),
        ("balloons_rfdetr_sem_nano_onnx_static_bs_stretch_package", 12),
        ("rfdetr_kp_preview_onnx_glue_sticks_package", 16),
    ],
)
def test_fuses_every_rfdetr_self_attention_block(
    package_fixture: str,
    expected_fused_blocks: int,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    # given
    package_dir = Path(request.getfixturevalue(package_fixture))
    model_path = tmp_path / "weights.onnx"
    shutil.copy(package_dir / "weights.onnx", model_path)

    # when
    with fuse_attention_for_cuda_execution(
        str(model_path), providers=CUDA_PROVIDERS
    ) as fused_model_path:
        operators = Counter(
            node.op_type for node in onnx.load(fused_model_path).graph.node
        )

    # then
    assert operators["MultiHeadAttention"] == expected_fused_blocks
