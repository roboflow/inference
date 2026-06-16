"""Tests for the deprecated native Qwen2.5-VL workflow block."""

import inference.core.workflows.core_steps.models.foundation.qwen.v1 as qwen25


def test_get_restrictions_marks_hosted_serverless_package_issue_when_enabled(
    monkeypatch,
):
    monkeypatch.setattr(qwen25, "QWEN_2_5_ENABLED", True)

    restrictions = [
        restriction.to_dict() for restriction in qwen25.BlockManifest.get_restrictions()
    ]

    assert any(
        restriction["severity"] == "hard"
        and restriction.get("applies_to_runtimes") == ["hosted_serverless"]
        and restriction.get("applies_to_step_execution_modes") == ["remote"]
        and "model package loading fails" in restriction["note"]
        for restriction in restrictions
    )


def test_get_restrictions_reports_hosted_serverless_disabled_flag(monkeypatch):
    monkeypatch.setattr(qwen25, "QWEN_2_5_ENABLED", False)

    restrictions = [
        restriction.to_dict() for restriction in qwen25.BlockManifest.get_restrictions()
    ]

    assert any(
        restriction["severity"] == "hard"
        and restriction.get("applies_to_runtimes") == ["hosted_serverless"]
        and restriction.get("applies_to_step_execution_modes") == ["remote"]
        and "QWEN_2_5_ENABLED=False" in restriction["note"]
        for restriction in restrictions
    )
