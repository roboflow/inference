from unittest.mock import call, patch

from inference_models.models.perception_encoder.vision_encoder import config


@patch.object(config, "hf_hub_download")
def test_fetch_pe_checkpoint_uses_hugging_face_cache_in_offline_mode(
    hf_hub_download,
):
    hf_hub_download.return_value = "/cache/checkpoint.pt"

    with patch.object(config, "OFFLINE_MODE", True):
        result = config.fetch_pe_checkpoint("PE-Core-B16-224")

    assert result == "/cache/checkpoint.pt"
    hf_hub_download.assert_called_once_with(
        repo_id="facebook/PE-Core-B16-224",
        filename="PE-Core-B16-224.pt",
        local_files_only=True,
    )


@patch.object(config, "hf_hub_download")
def test_fetch_pe_checkpoint_preserves_online_download_tracking(hf_hub_download):
    hf_hub_download.side_effect = ["/cache/config.yaml", "/cache/checkpoint.pt"]

    with patch.object(config, "OFFLINE_MODE", False):
        result = config.fetch_pe_checkpoint("PE-Core-B16-224")

    assert result == "/cache/checkpoint.pt"
    assert hf_hub_download.call_args_list == [
        call(repo_id="facebook/PE-Core-B16-224", filename="config.yaml"),
        call(
            repo_id="facebook/PE-Core-B16-224",
            filename="PE-Core-B16-224.pt",
            local_files_only=False,
        ),
    ]
