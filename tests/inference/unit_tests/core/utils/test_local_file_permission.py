"""One owner for the local-filesystem permission.

`inference/core/workflows` cannot read `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM`,
so the permission is injected through the `ImageCodec` port. To avoid a second
copy of the rule on the server side (which "injected, never reimplemented"
forbids), the check is a named helper in `image_utils` that both the existing
loader and the codec adapter call.
"""

from unittest import mock

import pytest

from inference.core.exceptions import InputImageLoadError
from inference.core.utils import image_utils
from inference.core.utils.image_utils import (
    ImageType,
    ensure_local_file_load_allowed,
    load_image,
    load_image_with_known_type,
)


@mock.patch.object(image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", True)
def test_helper_permits_when_the_flag_is_on() -> None:
    assert ensure_local_file_load_allowed() is None


@mock.patch.object(image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", False)
def test_helper_refuses_when_the_flag_is_off() -> None:
    with pytest.raises(InputImageLoadError) as error:
        ensure_local_file_load_allowed()
    assert "local filesystem" in str(error.value)


@pytest.mark.parametrize("flag_value", [True, False])
def test_declared_file_loader_delegates_to_the_helper(
    flag_value: bool, image_as_local_path: str
) -> None:
    # The extraction must not change `load_image_with_known_type`'s behaviour:
    # the loader raises exactly when the helper raises, with the same type.
    with mock.patch.object(
        image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", flag_value
    ):
        helper_raised = False
        try:
            ensure_local_file_load_allowed()
        except InputImageLoadError:
            helper_raised = True

        loader_raised = False
        try:
            load_image_with_known_type(
                value=image_as_local_path, image_type=ImageType.FILE
            )
        except InputImageLoadError:
            loader_raised = True

        dispatcher_raised = False
        try:
            load_image({"type": "file", "value": image_as_local_path})
        except InputImageLoadError:
            dispatcher_raised = True

    assert helper_raised == loader_raised == dispatcher_raised == (not flag_value)


@mock.patch.object(image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", False)
def test_the_loader_actually_calls_the_helper(image_as_local_path: str) -> None:
    # Guards against the extraction being reverted into an inline check.
    with mock.patch.object(
        image_utils, "ensure_local_file_load_allowed"
    ) as helper_mock:
        helper_mock.side_effect = InputImageLoadError(
            message="stub", public_message="stub"
        )
        with pytest.raises(InputImageLoadError, match="stub"):
            load_image_with_known_type(
                value=image_as_local_path, image_type=ImageType.FILE
            )
    helper_mock.assert_called_once_with()
