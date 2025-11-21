from unittest.mock import MagicMock, patch

import pytest
from inference_cli.lib.roboflow_cloud.data_staging.api_operations import (
    _parse_bucket_path,
    _get_fs_kwargs,
    _match_glob_pattern,
    _list_and_filter_files_streaming,
    _generate_presigned_urls_parallel,
)


def test_parse_bucket_path_without_glob():
    base, pattern = _parse_bucket_path("s3://bucket/path/")
    assert base == "s3://bucket/path/"
    assert pattern is None


def test_parse_bucket_path_with_glob():
    base, pattern = _parse_bucket_path("s3://bucket/path/**/*.jpg")
    assert base == "s3://bucket/path/"
    assert pattern == "**/*.jpg"


def test_get_fs_kwargs_with_endpoint(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://r2.example.com")
    monkeypatch.delenv("AWS_REGION", raising=False)
    kwargs = _get_fs_kwargs()
    assert kwargs["endpoint_url"] == "https://r2.example.com"
    assert "client_kwargs" not in kwargs  # No region specified


def test_get_fs_kwargs_without_endpoint(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    kwargs = _get_fs_kwargs()
    assert kwargs == {}


def test_get_fs_kwargs_with_profile(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "my-profile")
    kwargs = _get_fs_kwargs()
    assert kwargs["profile"] == "my-profile"


def test_get_fs_kwargs_with_endpoint_and_profile(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://r2.example.com")
    monkeypatch.setenv("AWS_PROFILE", "my-profile")
    monkeypatch.delenv("AWS_REGION", raising=False)
    kwargs = _get_fs_kwargs()
    assert kwargs["endpoint_url"] == "https://r2.example.com"
    assert kwargs["profile"] == "my-profile"
    assert "client_kwargs" not in kwargs  # No region specified


def test_get_fs_kwargs_with_endpoint_and_region(monkeypatch):
    """Test S3-compatible service (R2) with endpoint and region"""
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://r2.example.com")
    monkeypatch.setenv("AWS_REGION", "auto")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    kwargs = _get_fs_kwargs()
    assert kwargs["endpoint_url"] == "https://r2.example.com"
    assert kwargs["client_kwargs"]["region_name"] == "auto"
    assert "profile" not in kwargs


def test_get_fs_kwargs_with_endpoint_region_and_profile(monkeypatch):
    """Test full R2 configuration with endpoint, region, and profile"""
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://r2.example.com")
    monkeypatch.setenv("AWS_REGION", "auto")
    monkeypatch.setenv("AWS_PROFILE", "r2-read")
    kwargs = _get_fs_kwargs()
    assert kwargs["endpoint_url"] == "https://r2.example.com"
    assert kwargs["client_kwargs"]["region_name"] == "auto"
    assert kwargs["profile"] == "r2-read"


def test_get_fs_kwargs_gcs(monkeypatch):
    """Test GCS authentication - gcsfs auto-detects GOOGLE_APPLICATION_CREDENTIALS"""
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/service-account.json")

    # gcsfs auto-detects GOOGLE_APPLICATION_CREDENTIALS, so no kwargs needed
    kwargs = _get_fs_kwargs("gs")
    assert kwargs == {}


def test_get_fs_kwargs_azure(monkeypatch):
    """Test Azure authentication with account name and key"""
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_SAS_TOKEN", raising=False)
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "myaccount")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "mykey")

    kwargs = _get_fs_kwargs("az")
    assert kwargs["account_name"] == "myaccount"
    assert kwargs["account_key"] == "mykey"
    assert "sas_token" not in kwargs


def test_get_fs_kwargs_azure_sas_token(monkeypatch):
    """Test Azure authentication with SAS token (preferred over account key)"""
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "myaccount")
    monkeypatch.setenv("AZURE_STORAGE_SAS_TOKEN", "sv=2021-06-08&ss=b&srt=sco&sp=rl&se=2024-12-31")

    kwargs = _get_fs_kwargs("az")
    assert kwargs["account_name"] == "myaccount"
    assert kwargs["sas_token"] == "sv=2021-06-08&ss=b&srt=sco&sp=rl&se=2024-12-31"
    assert "account_key" not in kwargs


def test_get_fs_kwargs_azure_sas_token_precedence(monkeypatch):
    """Test that SAS token takes precedence over account key"""
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "myaccount")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "mykey")
    monkeypatch.setenv("AZURE_STORAGE_SAS_TOKEN", "sv=2021-06-08&ss=b&srt=sco&sp=rl")

    kwargs = _get_fs_kwargs("az")
    assert kwargs["account_name"] == "myaccount"
    assert kwargs["sas_token"] == "sv=2021-06-08&ss=b&srt=sco&sp=rl"
    assert "account_key" not in kwargs  # SAS token preferred


def test_get_fs_kwargs_azure_cli_naming(monkeypatch):
    """Test Azure authentication with Azure CLI standard naming convention"""
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_NAME", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_ACCOUNT_KEY", raising=False)
    monkeypatch.delenv("AZURE_STORAGE_SAS_TOKEN", raising=False)

    # Use Azure CLI standard naming
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT", "mycliaccount")
    monkeypatch.setenv("AZURE_STORAGE_KEY", "myclikey")

    kwargs = _get_fs_kwargs("az")
    assert kwargs["account_name"] == "mycliaccount"
    assert kwargs["account_key"] == "myclikey"


def test_get_fs_kwargs_azure_naming_precedence(monkeypatch):
    """Test that adlfs convention takes precedence over Azure CLI standard"""
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    # Set both naming conventions
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "adlfs_account")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT", "cli_account")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "adlfs_key")
    monkeypatch.setenv("AZURE_STORAGE_KEY", "cli_key")

    kwargs = _get_fs_kwargs("az")
    # adlfs convention should take precedence
    assert kwargs["account_name"] == "adlfs_account"
    assert kwargs["account_key"] == "adlfs_key"


def test_get_fs_kwargs_protocol_filtering(monkeypatch):
    """Test that protocol filtering only returns relevant kwargs"""
    monkeypatch.setenv("AWS_PROFILE", "my-profile")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/gcs.json")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "myaccount")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "mykey")

    # S3 protocol should only get AWS kwargs
    s3_kwargs = _get_fs_kwargs("s3")
    assert "profile" in s3_kwargs
    assert "account_name" not in s3_kwargs

    # GCS protocol returns empty (auto-detection)
    gcs_kwargs = _get_fs_kwargs("gs")
    assert gcs_kwargs == {}

    # Azure protocol should only get Azure kwargs
    az_kwargs = _get_fs_kwargs("az")
    assert "account_name" in az_kwargs
    assert "account_key" in az_kwargs
    assert "profile" not in az_kwargs


def test_match_glob_pattern():
    """Test glob pattern matching"""
    # Test ** recursive matching
    assert _match_glob_pattern("2024/01/image.jpg", "**/*.jpg")
    assert _match_glob_pattern("a/b/c/file.png", "**/*.png")
    assert not _match_glob_pattern("file.txt", "**/*.jpg")

    # Test * wildcard matching
    assert _match_glob_pattern("2024-01/image.jpg", "2024-*/*.jpg")
    assert _match_glob_pattern("prefix-test/file.png", "prefix-*/*.png")
    assert not _match_glob_pattern("2025-01/image.jpg", "2024-*/*.jpg")

    # Test literal matching
    assert _match_glob_pattern("exact/path/file.jpg", "exact/path/file.jpg")
    assert not _match_glob_pattern("different/path/file.jpg", "exact/path/file.jpg")


def test_list_and_filter_files_streaming():
    """Test streaming file listing with glob patterns"""
    mock_fs = MagicMock()

    # Mock fs.exists to return True (path exists)
    mock_fs.exists.return_value = True

    # Mock fs.walk to return directory structure
    mock_fs.walk.return_value = [
        ("s3://bucket/path", [], ["image1.jpg", "image2.png", "document.pdf"]),
        ("s3://bucket/path/sub", [], ["image3.jpg", "video.mp4"]),
    ]

    # Mock rich.progress.Progress to avoid actual progress bar
    with patch('inference_cli.lib.roboflow_cloud.data_staging.api_operations.Progress'):
        result = list(_list_and_filter_files_streaming(
            mock_fs,
            "s3://bucket/path/",
            "**/*.jpg",
            ["jpg", "png"]
        ))

    # Should find .jpg files (even though png is in extensions, pattern limits to .jpg)
    assert len(result) == 2
    assert "s3://bucket/path/image1.jpg" in result
    assert "s3://bucket/path/sub/image3.jpg" in result


def test_list_and_filter_files_streaming_no_pattern():
    """Test streaming without pattern (all files with matching extensions)"""
    mock_fs = MagicMock()

    # Mock fs.exists to return True (path exists)
    mock_fs.exists.return_value = True

    mock_fs.walk.return_value = [
        ("s3://bucket/path", [], ["image1.jpg", "image2.JPG", "document.pdf", "image3.png"]),
    ]

    with patch('inference_cli.lib.roboflow_cloud.data_staging.api_operations.Progress'):
        result = list(_list_and_filter_files_streaming(
            mock_fs,
            "s3://bucket/path/",
            None,  # No pattern
            ["jpg", "png"]
        ))

    # Should find all jpg and png files (case insensitive)
    assert len(result) == 3
    assert "s3://bucket/path/image1.jpg" in result
    assert "s3://bucket/path/image2.JPG" in result
    assert "s3://bucket/path/image3.png" in result


def test_generate_presigned_urls_parallel():
    """Test parallel presigned URL generation from generator"""
    mock_fs = MagicMock()
    mock_fs.sign.side_effect = lambda path, expiration: f"https://signed-url.com/{path.split('/')[-1]}"

    def file_generator():
        """Generator to simulate streaming files"""
        yield "bucket/path/file1.jpg"
        yield "bucket/path/file2.jpg"
        yield "s3://bucket/path/file3.jpg"  # Already has protocol

    # Mock Progress to avoid actual progress bars during tests
    with patch('inference_cli.lib.roboflow_cloud.data_staging.api_operations.Progress'), \
         patch('builtins.print'):  # Mock print to avoid output during tests
        result = _generate_presigned_urls_parallel(
            mock_fs,
            file_generator(),
            "s3://bucket/path/",
            expiration_seconds=86400
        )

    assert len(result) == 3
    assert result[0]["name"] == "file1.jpg"
    assert result[0]["url"] == "https://signed-url.com/file1.jpg"
    assert result[1]["name"] == "file2.jpg"
    assert result[2]["name"] == "file3.jpg"

    # Verify fs.sign was called correct number of times
    assert mock_fs.sign.call_count == 3


def test_generate_presigned_urls_error_handling():
    """Test that errors during URL generation are properly escalated"""
    mock_fs = MagicMock()
    # Simulate connection error on second file
    mock_fs.sign.side_effect = [
        "https://signed-url.com/file1.jpg",
        ConnectionError("Failed to connect to S3"),
        "https://signed-url.com/file3.jpg"
    ]

    def file_generator():
        """Generator to simulate streaming files"""
        yield "bucket/path/file1.jpg"
        yield "bucket/path/file2.jpg"
        yield "bucket/path/file3.jpg"

    # Mock Progress to avoid actual progress bars during tests
    with patch('inference_cli.lib.roboflow_cloud.data_staging.api_operations.Progress'), \
         patch('builtins.print'):
        # Should raise the connection error
        with pytest.raises(ConnectionError) as exc_info:
            _generate_presigned_urls_parallel(
                mock_fs,
                file_generator(),
                "s3://bucket/path/",
                expiration_seconds=86400
            )

        # Verify the error message contains context
        error_msg = str(exc_info.value)
        assert "Cloud storage error" in error_msg
        assert "Presigned URL generation" in error_msg or "fs.sign" in error_msg


def test_generate_presigned_urls_generator_error():
    """Test that errors during file discovery are properly escalated"""
    mock_fs = MagicMock()
    mock_fs.sign.return_value = "https://signed-url.com/file.jpg"

    def failing_generator():
        """Generator that raises an error during iteration"""
        yield "bucket/path/file1.jpg"
        raise PermissionError("Access denied to bucket")

    # Mock Progress to avoid actual progress bars during tests
    with patch('inference_cli.lib.roboflow_cloud.data_staging.api_operations.Progress'), \
         patch('builtins.print'):
        # Should raise the permission error
        with pytest.raises(PermissionError) as exc_info:
            _generate_presigned_urls_parallel(
                mock_fs,
                failing_generator(),
                "s3://bucket/path/",
                expiration_seconds=86400
            )

        # Verify the error message contains context
        error_msg = str(exc_info.value)
        assert "Cloud storage error" in error_msg
        assert "File discovery" in error_msg or "fs.walk" in error_msg


def test_list_and_filter_files_path_not_exists():
    """Test that non-existent bucket/path is properly detected"""
    mock_fs = MagicMock()

    # Mock fs.exists to return False (bucket doesn't exist)
    mock_fs.exists.return_value = False

    with pytest.raises(FileNotFoundError) as exc_info:
        list(_list_and_filter_files_streaming(
            mock_fs,
            "s3://nonexistent-bucket/",
            None,
            ["jpg", "png"]
        ))

    # Verify error message is helpful
    error_msg = str(exc_info.value)
    assert "does not exist" in error_msg
    assert "s3://nonexistent-bucket/" in error_msg
    assert "Bucket doesn't exist" in error_msg or "Wrong region" in error_msg


def test_list_and_filter_files_access_error():
    """Test that access errors are properly escalated"""
    mock_fs = MagicMock()

    # Mock fs.exists to raise permission error
    mock_fs.exists.side_effect = PermissionError("Access Denied")

    with pytest.raises(Exception) as exc_info:
        list(_list_and_filter_files_streaming(
            mock_fs,
            "s3://restricted-bucket/",
            None,
            ["jpg", "png"]
        ))

    # Verify error message provides helpful context
    error_msg = str(exc_info.value)
    assert "Failed to access cloud storage" in error_msg
    assert "s3://restricted-bucket/" in error_msg
    assert "credentials" in error_msg or "Permission denied" in error_msg
