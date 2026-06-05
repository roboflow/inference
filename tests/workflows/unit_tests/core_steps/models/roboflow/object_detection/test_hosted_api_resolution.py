import pytest

from tests.workflows.unit_tests.core_steps.models.roboflow._hosted_api_resolution import (
    assert_hosted_selects_v0,
    assert_non_hosted_uses_local,
)

FAMILY = "object_detection"
HOSTED_URL_ATTR = "HOSTED_DETECT_URL"
BLOCK_CLASSES = {
    "v1": "RoboflowObjectDetectionModelBlockV1",
    "v2": "RoboflowObjectDetectionModelBlockV2",
    "v3": "RoboflowObjectDetectionModelBlockV3",
}


@pytest.mark.parametrize("version", list(BLOCK_CLASSES))
def test_hosted_remote_execution_selects_v0_and_hosted_url(version):
    assert_hosted_selects_v0(FAMILY, version, BLOCK_CLASSES[version], HOSTED_URL_ATTR)


@pytest.mark.parametrize("version", list(BLOCK_CLASSES))
def test_non_hosted_remote_execution_uses_local_url_and_no_v0(version):
    assert_non_hosted_uses_local(FAMILY, version, BLOCK_CLASSES[version])
