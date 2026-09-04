import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
FRAMEWORK_PATH = (
    REPO_ROOT
    / "development"
    / "video_poc"
    / "benchmarks"
    / "recommendations"
    / "decision-framework.json"
)
SCHEMA_PATH = FRAMEWORK_PATH.with_name("decision-framework.schema.json")
sys.path.insert(0, str(FRAMEWORK_PATH.parent))

from validate import FrameworkValidationError, validate_document  # noqa: E402
EXPECTED_DECISIONS = {
    "worker-topology",
    "gpu-stream-capacity-and-slo",
    "cpu-vs-gpu-routing",
    "trusted-shared-model-manager",
    "fairness-and-admission",
    "dedicated-and-multi-cell-placement",
    "mps",
    "failure-recovery-and-soak",
    "relay-and-network-capacity",
    "pricing-capacity-and-metering",
}
EVIDENCE_STATUSES = {
    "verified_observation",
    "preliminary_observation",
    "design_only",
    "pending",
}
DECISION_STATUSES = {
    "certified",
    "provisional",
    "design_candidate",
    "insufficient_evidence",
}
REQUIREMENT_STATUSES = {"satisfied", "pending"}


def load_framework():
    return json.loads(FRAMEWORK_PATH.read_text())


def load_schema():
    return json.loads(SCHEMA_PATH.read_text())


def test_framework_has_a_versioned_closed_structural_schema():
    schema = load_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schemaVersion"] == {"const": 1}
    assert schema["properties"]["environment"] == {"const": "staging"}
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["evidence"]["additionalProperties"] is False
    assert schema["$defs"]["decision"]["additionalProperties"] is False
    assert schema["$defs"]["requirement"]["additionalProperties"] is False


def test_current_framework_validates_against_schema_and_fail_closed_rules():
    document = load_framework()

    validated = validate_document(document, load_schema(), REPO_ROOT)

    assert validated == document
    assert validated is not document


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_schema_validation_rejects_instance_shape_drift(mutation):
    document = load_framework()
    if mutation == "extra":
        document["decisions"][0]["plausibleButUnknownField"] = True
    else:
        del document["decisions"][0]["currentRecommendation"]

    with pytest.raises(FrameworkValidationError):
        validate_document(document, load_schema(), REPO_ROOT)


def test_provisional_requires_verified_runtime_evidence():
    document = load_framework()
    decision = next(
        item
        for item in document["decisions"]
        if item["id"] == "trusted-shared-model-manager"
    )
    decision["certificationStatus"] = "provisional"

    with pytest.raises(FrameworkValidationError, match="verified observation"):
        validate_document(document, load_schema(), REPO_ROOT)


def test_certified_decision_cannot_retain_unknowns():
    document = load_framework()
    decision = document["decisions"][0]
    decision["certificationStatus"] = "certified"

    with pytest.raises(FrameworkValidationError, match="cannot retain unknowns"):
        validate_document(document, load_schema(), REPO_ROOT)


def test_requirement_evidence_must_be_visible_on_parent_decision():
    document = load_framework()
    decision = document["decisions"][0]
    decision["evidenceIds"].remove("selected-point-soaks")

    with pytest.raises(FrameworkValidationError, match="omitted from parent"):
        validate_document(document, load_schema(), REPO_ROOT)


def test_verified_observation_requires_nonempty_facts_and_exact_strength():
    document = load_framework()
    del document["evidence"][0]["facts"]
    with pytest.raises(FrameworkValidationError, match="nonempty facts"):
        validate_document(document, load_schema(), REPO_ROOT)

    document = load_framework()
    document["evidence"][0]["strength"] = "none"
    with pytest.raises(FrameworkValidationError, match="does not match"):
        validate_document(document, load_schema(), REPO_ROOT)


def test_framework_covers_every_required_decision_and_uses_unique_ids():
    framework = validate_document(load_framework(), load_schema(), REPO_ROOT)
    assert framework["schemaVersion"] == 1
    assert framework["environment"] == "staging"

    evidence_ids = [item["id"] for item in framework["evidence"]]
    decision_ids = [item["id"] for item in framework["decisions"]]
    assert len(evidence_ids) == len(set(evidence_ids))
    assert len(decision_ids) == len(set(decision_ids))
    assert set(decision_ids) == EXPECTED_DECISIONS


def test_evidence_is_bounded_to_existing_repository_sources():
    framework = validate_document(load_framework(), load_schema(), REPO_ROOT)
    for evidence in framework["evidence"]:
        assert evidence["status"] in EVIDENCE_STATUSES
        assert evidence["scope"].strip()
        source = evidence["source"]
        assert not source.startswith(("http://", "https://", "/"))
        assert (REPO_ROOT / source).is_file(), source


def test_decisions_fail_closed_when_evidence_is_incomplete():
    framework = validate_document(load_framework(), load_schema(), REPO_ROOT)
    evidence = {item["id"]: item for item in framework["evidence"]}

    for decision in framework["decisions"]:
        assert decision["certificationStatus"] in DECISION_STATUSES
        assert decision["currentRecommendation"]["action"].strip()
        assert decision["currentRecommendation"]["scope"].strip()
        assert decision["currentRecommendation"]["doNotClaim"].strip()
        assert decision["certificationRequirements"]
        if decision["certificationStatus"] == "certified":
            assert decision["unknowns"] == []
        else:
            assert decision["unknowns"]

        for evidence_id in decision["evidenceIds"]:
            assert evidence_id in evidence

        requirement_ids = []
        for requirement in decision["certificationRequirements"]:
            requirement_ids.append(requirement["id"])
            assert requirement["status"] in REQUIREMENT_STATUSES
            assert requirement["requirement"].strip()
            assert requirement["evidenceIds"]
            requirement_evidence = [
                evidence[evidence_id] for evidence_id in requirement["evidenceIds"]
            ]
            assert set(requirement["evidenceIds"]).issubset(decision["evidenceIds"])
            if requirement["status"] == "satisfied":
                assert all(
                    item["status"] == "verified_observation"
                    for item in requirement_evidence
                )
        assert len(requirement_ids) == len(set(requirement_ids))

        if decision["certificationStatus"] == "certified":
            assert all(
                requirement["status"] == "satisfied"
                for requirement in decision["certificationRequirements"]
            )


def test_current_native_capacity_fact_is_exactly_scoped_not_generalized():
    framework = validate_document(load_framework(), load_schema(), REPO_ROOT)
    evidence = {item["id"]: item for item in framework["evidence"]}
    capacity = evidence["abc-native-capacity"]

    assert capacity["facts"]["strictMaxConcurrencyByVariant"] == {
        "a-legacy-pyav-thread": 2,
        "b-v14-pyav-thread": 2,
        "c-v14-nvdec-thread": 2,
    }
    assert capacity["facts"]["firstRepeatedFailingConcurrency"] == 4
    assert "1280x720" in capacity["scope"]
    assert "native 60 FPS" in capacity["scope"]

    decision = next(
        item
        for item in framework["decisions"]
        if item["id"] == "gpu-stream-capacity-and-slo"
    )
    assert decision["certificationStatus"] == "provisional"
    assert "product packing limit unknown" in decision["currentRecommendation"][
        "action"
    ]


def test_pricing_and_mps_remain_uncertified_without_results_or_costs():
    framework = validate_document(load_framework(), load_schema(), REPO_ROOT)
    decisions = {item["id"]: item for item in framework["decisions"]}

    assert decisions["pricing-capacity-and-metering"]["certificationStatus"] == (
        "insufficient_evidence"
    )
    assert decisions["mps"]["certificationStatus"] == "insufficient_evidence"
    assert all(
        requirement["status"] == "pending"
        for decision_id in ("pricing-capacity-and-metering", "mps")
        for requirement in decisions[decision_id]["certificationRequirements"]
    )
