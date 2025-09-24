from unittest import mock
from unittest.mock import MagicMock, call

from inference.core.utils import s3
from inference.core.utils.s3 import download_s3_files_to_directory


@mock.patch.object(s3.os, "makedirs")
def test_download_s3_files_to_directory(makedirs_mock: MagicMock) -> None:
    # this is just regression test to mocked boto API checking if parameters were
    # passed correctly.
    # given
    s3_client = MagicMock()

    # when
    download_s3_files_to_directory(
        bucket="some-bucket",
        keys=["a.jpg", "sub_dir/b.txt"],
        target_dir="/some/local/dir",
        s3_client=s3_client,
    )

    # then
    # according to docs:
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/download_file.html
    s3_client.download_file.assert_has_calls(
        [
            call("some-bucket", "a.jpg", "/some/local/dir/a.jpg"),
            call("some-bucket", "sub_dir/b.txt", "/some/local/dir/sub_dir/b.txt"),
        ]
    )
    makedirs_mock.assert_called_once_with("/some/local/dir", exist_ok=True)
