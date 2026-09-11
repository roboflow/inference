from unittest import mock

import cv2
import numpy as np
import pytest

from inference.core.exceptions import InputImageLoadError
from inference.core.interfaces.workflows_image_codec import install_guarded_image_codec
from inference.core.utils import image_utils
from inference.core.workflows.core_steps.visualizations.icon import v1 as icon_v1
from inference.core.workflows.core_steps.visualizations.icon import (
    v1_tensor as icon_v1_tensor,
)
from inference.core.workflows.errors import WorkflowImageLoadError
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.image_codec import reset_image_codec

BLOCK_MODULES = pytest.mark.parametrize(
    "block_module", [icon_v1, icon_v1_tensor], ids=["numpy", "tensor"]
)


@pytest.fixture(autouse=True)
def _reset_codec():
    reset_image_codec()
    yield
    reset_image_codec()


def _background() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="background"),
        numpy_image=np.ones((64, 64, 3), dtype=np.uint8) * 255,
    )


def _run(block_module, icon: WorkflowImageData):
    block = block_module.IconVisualizationBlockV1()
    return block.run(
        image=_background(),
        copy_image=True,
        mode="static",
        icon=icon,
        predictions=None,
        icon_width=16,
        icon_height=16,
        position=None,
        x_position=10,
        y_position=10,
    )


@BLOCK_MODULES
def test_icon_reload_is_refused_by_the_workflows_default_codec(tmp_path, block_module):
    # given: no codec installed -> get_image_codec() falls back to the refusing
    # WorkflowsLocalImageCodec default.
    icon_path = tmp_path / "icon.png"
    icon_bgr = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(icon_path), icon_bgr)
    icon = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="icon"),
        numpy_image=icon_bgr,
        image_reference=str(icon_path),
    )

    # when / then
    with mock.patch("cv2.imread") as imread_mock:
        with pytest.raises(WorkflowImageLoadError, match="local filesystem"):
            _run(block_module, icon)
        imread_mock.assert_not_called()


@BLOCK_MODULES
def test_icon_reload_is_refused_when_the_server_flag_is_off(tmp_path, block_module):
    # given: the guarded server codec is installed, but the deployment flag
    # disables local-filesystem reads.
    install_guarded_image_codec()
    icon_path = tmp_path / "icon.png"
    icon_bgr = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(icon_path), icon_bgr)
    icon = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="icon"),
        numpy_image=icon_bgr,
        image_reference=str(icon_path),
    )

    # when / then
    with mock.patch.object(
        image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", False
    ):
        with mock.patch("cv2.imread") as imread_mock:
            with pytest.raises(InputImageLoadError, match="local filesystem"):
                _run(block_module, icon)
            imread_mock.assert_not_called()


@BLOCK_MODULES
def test_icon_reload_still_recovers_alpha_when_permitted(tmp_path, block_module):
    # given: the guarded codec is installed and the flag is on (default) -> the
    # gate must let the reload through unchanged.
    install_guarded_image_codec()
    icon_path = tmp_path / "icon.png"
    icon_bgra = np.zeros((16, 16, 4), dtype=np.uint8)
    icon_bgra[..., 3] = 255
    cv2.imwrite(str(icon_path), icon_bgra)
    icon = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="icon"),
        numpy_image=icon_bgra[..., :3].copy(),
        image_reference=str(icon_path),
    )

    # when
    with mock.patch.object(
        image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", True
    ):
        with mock.patch("cv2.imread", wraps=cv2.imread) as imread_mock:
            output = _run(block_module, icon)

    # then: the reload happened (proving the guard sits before it) and the
    # happy path still produces a valid image. `assert_any_call`, not
    # `assert_called_once_with`: `sv.IconAnnotator` reloads the (temp) icon
    # file internally via its own `cv2.imread` call, which the global patch
    # also observes.
    imread_mock.assert_any_call(str(icon_path), cv2.IMREAD_UNCHANGED)
    result_image = output["image"].numpy_image
    assert result_image.shape == (64, 64, 3)
    assert result_image.dtype == np.uint8


@BLOCK_MODULES
def test_icon_without_a_reference_never_consults_the_gate(block_module):
    # given: no image_reference at all, and the refusing default codec
    # installed (via the autouse reset) -> the gate must not be reached.
    icon = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="icon"),
        numpy_image=np.zeros((16, 16, 3), dtype=np.uint8),
    )

    # when / then: run() must succeed rather than raise.
    output = _run(block_module, icon)
    assert output["image"].numpy_image.shape == (64, 64, 3)
