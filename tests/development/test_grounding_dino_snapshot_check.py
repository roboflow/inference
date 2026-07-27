from pathlib import Path

from development.grounding_dino_snapshot_check import (
    BERT_ALLOW_PATTERNS,
    BERT_REVISION,
    SNAPSHOT_CASES,
    snapshot_download_kwargs,
)


def test_current_snapshot_case_matches_production_call(tmp_path: Path) -> None:
    assert snapshot_download_kwargs(
        case_name="current-unrestricted",
        cache_dir=tmp_path,
    ) == {
        "repo_id": "bert-base-uncased",
        "revision": BERT_REVISION,
        "cache_dir": str(tmp_path),
        "local_files_only": False,
    }


def test_restricted_cases_use_only_runtime_bert_files(tmp_path: Path) -> None:
    for case_name in ("alias-restricted", "canonical-restricted"):
        kwargs = snapshot_download_kwargs(
            case_name=case_name,
            cache_dir=tmp_path / case_name,
            local_files_only=True,
        )

        assert kwargs["allow_patterns"] == BERT_ALLOW_PATTERNS
        assert kwargs["local_files_only"] is True
        assert "coreml/**" not in kwargs["allow_patterns"]


def test_canonical_case_avoids_repository_alias() -> None:
    assert SNAPSHOT_CASES["canonical-restricted"]["repo_id"] == (
        "google-bert/bert-base-uncased"
    )
