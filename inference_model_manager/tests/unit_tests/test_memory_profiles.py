"""Admission matching and publication lifecycle tests without GPU/cloud access."""

from copy import deepcopy
from dataclasses import replace

import pytest

from inference_model_manager.memory_profiles.contracts import (
    Catalog,
    Envelope,
    Evidence,
    Family,
    Policy,
    RangeEstimate,
    Reference,
    Request,
    RuntimeIdentity,
    Template,
    VersionRule,
    Workload,
    WorkloadRange,
)
from inference_model_manager.memory_profiles.firestore import FirestoreCatalog
from inference_model_manager.memory_profiles.matching import (
    MemoryProfileError,
    Snapshot,
    content_digest,
    resolve,
)


def workload(width=1920, batch=1, confidence=0.2):
    return Workload(values={"width": width, "batch": batch, "confidence": confidence})


def band(range_id, *, width=1920, confidence=0.1, batch=1):
    return WorkloadRange(
        id=range_id,
        lower=workload(width, 1, confidence),
        upper=workload(width, batch, 1.0),
    )


@pytest.fixture
def snapshot():
    runtime = RuntimeIdentity(
        components={"pytorch": "2.9.1", "cuda": "12.8", "nvidia.driver": "570.0"},
        inference_models_version="0.39.5rc2",
        inference_models_build="a" * 40,
    )
    environment = {
        "deviceFamily": "nvidia_discrete_gpu",
        "memoryModel": "discrete",
        "computeCapability": "8.9",
        "gpuSku": "NVIDIA L40S",
    }
    structural = {
        "memoryBehaviorKey": "rfdetr-seg-base-v1",
        "processLayout": "one-new-model-process",
        "backendType": "torch",
        "quantization": "fp32",
        "architecture": "rfdetr",
        "modelVariant": "medium",
        "taskType": "instance-segmentation",
        "batchPolicy": "batch1",
        "networkShapePolicyId": "432x432",
        "entrypointMethod": "infer",
        "executionConstraintsId": "base-v1",
        "familyConstraintsId": "head-v1",
    }
    evidence = Evidence(
        bucket="staging",
        object="capture/export.json",
        generation="1",
        export_sha256="a" * 64,
        record_sha256="b" * 64,
        measurement_identity_sha256="c" * 64,
        model_id="source/1",
        package_id="source-package",
        entrypoint_name="infer",
        scenario_name="base",
        profile_id="profile",
        runtime=runtime,
        source_provider="roboflow",
        source_registry_project="roboflow-staging",
        environment_snapshot_sha256="d" * 64,
        source_files=[
            dict(
                role=role,
                bucket="staging",
                object=f"capture/{role}.json",
                generation="1",
                sha256="e" * 64,
                size_bytes=10,
            )
            for role in ("batch", "run_metadata")
        ],
    )

    def measured(range_id, peak):
        return RangeEstimate(
            range_id=range_id,
            estimated_gpu_peak_bytes=peak,
            evidence=[
                evidence.model_copy(
                    update={
                        "profile_id": range_id,
                        "scenario_name": "capture-" + range_id,
                    }
                )
            ],
        )

    envelope = Envelope(
        id="hd",
        ranges=[
            band("r1", confidence=0.4),
            band("r2"),
            band("4k", width=3840),
            band("r3", confidence=0.01, batch=4),
        ],
    )
    first = Template(
        id="hd-template",
        version="1",
        family_contract_id="rfdetr-seg-v1",
        structural_key=structural,
        environment=environment,
        envelope_id=envelope.id,
        range_estimates=[
            measured("r1", 2000),
            measured("r2", 3000),
            measured("4k", 7000),
            measured("r3", 2800),
        ],
        allowed_deployment_policy_id="staging",
        validation_status="validated",
        limitations=["sampled_peak", "bounded_workload"],
    )
    wider = Envelope(id="large-batch", ranges=[band("r4", confidence=0.01, batch=32)])
    second = first.model_copy(
        update={
            "id": "large-template",
            "envelope_id": wider.id,
            "range_estimates": [measured("r4", 9000)],
            "limitations": ["finite_content_samples"],
        }
    )
    family = Family(
        id="rfdetr-seg-v1",
        required_fields=list(structural),
        runtime_components=list(runtime.components),
        behavior_keys=[structural["memoryBehaviorKey"]],
        workload_parameters={
            "width": "exact",
            "batch": "higher",
            "confidence": "lower",
        },
    )
    refs = [
        Reference(id=t.id, version=t.version, sha256=content_digest(t))
        for t in [first, second]
    ]
    return Snapshot(
        "roboflow-staging",
        1,
        Catalog(
            id="catalog-1",
            templates=refs,
            envelopes=[envelope, wider],
            families=[family],
        ),
        Policy(id="policy-1", deployment_policy_id="staging"),
        (first, second),
    )


def request_for(snapshot, **changes):
    template = snapshot.templates[0]
    values = dict(
        request_id="r1",
        package_context={"modelId": "different-target/2", "packageId": "other-weights"},
        family_contract_id=template.family_contract_id,
        structural_key=template.structural_key,
        environment=template.environment,
        runtime=template.range_estimates[0].evidence[0].runtime,
    )
    values.update(changes)
    return Request(**values)


@pytest.mark.parametrize(
    ("value", "range_id", "peak"),
    [(0.4, "r1", 2000), (0.2, "r2", 3000), (0.1, "r2", 3000), (0.01, "r3", 2800)],
)
def test_first_joint_range_in_declared_order(snapshot, value, range_id, peak):
    result = resolve(
        snapshot, request=request_for(snapshot, workload=workload(confidence=value))
    )
    assert result["rangeId"] == range_id
    assert result["estimatedGpuPeakBytes"] == peak
    assert [item["profile_id"] for item in result["evidenceRefs"]] == [range_id]


def test_next_range_must_contain_the_entire_current_range_and_never_reduce_estimate(
    snapshot,
):
    result = resolve(
        snapshot,
        request=request_for(snapshot, workload=workload(), selection="next_containing"),
    )
    assert result["rangeId"] == "r3"
    assert result["estimatedGpuPeakBytes"] == 3000
    assert {e["profile_id"] for e in result["evidenceRefs"]} == {"r2", "r3"}
    with pytest.raises(MemoryProfileError, match="No strictly broader"):
        resolve(
            snapshot,
            request=request_for(
                snapshot, workload=workload(batch=32), selection="next_containing"
            ),
        )


@pytest.mark.parametrize("supplied", [True, False])
def test_maximum_includes_all_compatible_templates(snapshot, supplied):
    fields = {"workload": workload(), "selection": "maximum"} if supplied else {}
    result = resolve(snapshot, request=request_for(snapshot, **fields))
    assert result["selection"] == "maximum"
    assert result["estimatedGpuPeakBytes"] == 9000
    assert "finite_content_samples" in result["limitations"]


def test_maximum_retains_coverage_and_measurement_from_distinct_ranges(snapshot):
    result = resolve(
        snapshot,
        request=request_for(
            snapshot, workload=workload(width=3840), selection="maximum"
        ),
    )
    assert result["estimatedGpuPeakBytes"] == 9000
    assert (result["servingEnvelopeId"], result["rangeId"]) == ("hd", "4k")
    assert [item["rangeId"] for item in result["contributingRanges"]] == ["4k", "r4"]
    assert {e["profile_id"] for e in result["evidenceRefs"]} == {"4k", "r4"}


@pytest.mark.parametrize(
    "selection", ["maximum", "next_containing", "first_containing"]
)
@pytest.mark.parametrize(
    "values",
    [
        {"width": 3840, "batch": 32, "confidence": 0.1},
        {"width": 1920, "batch": False, "confidence": 0.1},
        {"width": 1920, "batch": 1},
        {"width": 1920, "batch": 1, "confidence": 0.001},
        {"width": 1920, "batch": 1, "confidence": 1.1},
        {"width": 1920, "batch": 1, "confidence": 0.1, "extra": 5},
    ],
)
def test_uncovered_workload_never_returns_an_estimate(snapshot, selection, values):
    with pytest.raises(MemoryProfileError) as error:
        resolve(
            snapshot,
            request=request_for(
                snapshot, workload=Workload(values=values), selection=selection
            ),
        )
    assert error.value.code == "outside_profile_envelope"


def test_snapshot_requires_an_estimate_for_every_range(snapshot):
    changed = snapshot.templates[0].model_copy(
        update={"range_estimates": snapshot.templates[0].range_estimates[:-1]}
    )
    ref = snapshot.catalog.templates[0].model_copy(
        update={"sha256": content_digest(changed)}
    )
    broken = replace(
        snapshot,
        templates=(changed, snapshot.templates[1]),
        catalog=snapshot.catalog.model_copy(
            update={"templates": [ref, snapshot.catalog.templates[1]]}
        ),
    )
    with pytest.raises(MemoryProfileError, match="exactly one estimate"):
        resolve(broken, request=request_for(broken))


def test_range_boundaries_require_all_axes_and_finite_numbers(snapshot):
    directions = snapshot.catalog.families[0].workload_parameters
    with pytest.raises(ValueError, match="every family parameter"):
        band("invalid").profiled_values({**directions, "extra": "exact"})
    with pytest.raises(ValueError, match="identical bounds"):
        band("invalid").profiled_values({**directions, "confidence": "exact"})
    with pytest.raises(ValueError, match="ordered numeric bounds"):
        band("invalid", confidence=2.0).profiled_values(directions)
    with pytest.raises(ValueError):
        workload(confidence=float("nan"))
    with pytest.raises(ValueError):
        workload(confidence=float("inf"))


def test_unknown_behavior_and_missing_fields_reject(snapshot):
    for key in (
        {},
        {**snapshot.templates[0].structural_key, "memoryBehaviorKey": "new-plugin"},
        {**snapshot.templates[0].structural_key, "processLayout": "shared-base"},
    ):
        with pytest.raises(MemoryProfileError) as error:
            resolve(snapshot, request=request_for(snapshot, structural_key=key))
        assert error.value.code == "unsupported_family_descriptor"


def test_gpu_mismatch_cannot_relax_policy(snapshot):
    with pytest.raises(MemoryProfileError) as error:
        resolve(
            snapshot,
            request=request_for(
                snapshot,
                environment={
                    **snapshot.templates[0].environment,
                    "gpuSku": "NVIDIA L4",
                },
            ),
        )
    assert error.value.code == "no_compatible_template"


@pytest.mark.parametrize("value", ["unknown", "unavailable", "", " Unavailable "])
def test_unknown_gpu_fact_rejects_request_and_publication(snapshot, value):
    template = snapshot.templates[0]
    environment = {**template.environment, "gpuSku": value}
    request = request_for(snapshot, environment=environment)
    with pytest.raises(MemoryProfileError) as error:
        resolve(snapshot, request=request)
    assert error.value.code == "environment_metadata_unavailable"

    changed = template.model_copy(update={"environment": environment})
    snapshot.catalog.templates[0] = snapshot.catalog.templates[0].model_copy(
        update={"sha256": content_digest(changed)}
    )
    with pytest.raises(MemoryProfileError) as error:
        resolve(
            replace(snapshot, templates=(changed, *snapshot.templates[1:])),
            request=request,
        )
    assert error.value.code == "memory_publication_invalid"


@pytest.mark.parametrize("build", ["unknown", "unavailable"])
def test_identical_unknown_build_is_observed_or_rejected_when_enforced(snapshot, build):
    for index, template in enumerate(snapshot.templates):
        for estimate in template.range_estimates:
            evidence = estimate.evidence[0]
            runtime = evidence.runtime.model_copy(
                update={"inference_models_build": build}
            )
            estimate.evidence[0] = evidence.model_copy(update={"runtime": runtime})
        ref = snapshot.catalog.templates[index]
        snapshot.catalog.templates[index] = ref.model_copy(
            update={"sha256": content_digest(template)}
        )

    request = request_for(snapshot)
    result = resolve(snapshot, request=request)
    assert "runtime_version_unvalidated" in result["limitations"]
    assert result["runtimeCompatibility"][0]["coverage"] == ["covered", "unknown"]
    policy = snapshot.policy.model_copy(update={"runtime_coverage_mode": "enforce"})
    with pytest.raises(MemoryProfileError) as error:
        resolve(replace(snapshot, policy=policy), request=request)
    assert error.value.code == "no_compatible_template"


def test_uncovered_runtime_is_observe_only_but_known_incompatibility_rejects(snapshot):
    runtime = (
        snapshot.templates[0]
        .range_estimates[0]
        .evidence[0]
        .runtime.model_copy(
            update={
                "inference_models_version": "0.40.0",
                "inference_models_build": None,
            }
        )
    )
    request = request_for(snapshot, runtime=runtime)
    assert (
        "runtime_version_unvalidated"
        in resolve(snapshot, request=request)["limitations"]
    )
    deny = VersionRule(
        component="inference-models",
        source=">=0.39,<0.40",
        target=">=0.40",
        effect="deny",
    )
    snapshot = replace(
        snapshot, policy=snapshot.policy.model_copy(update={"version_rules": [deny]})
    )
    with pytest.raises(MemoryProfileError):
        resolve(snapshot, request=request)


def test_library_patch_changes_need_explicit_rule_and_deny_wins(snapshot):
    runtime = (
        snapshot.templates[0]
        .range_estimates[0]
        .evidence[0]
        .runtime.model_copy(
            update={
                "components": {
                    **snapshot.templates[0]
                    .range_estimates[0]
                    .evidence[0]
                    .runtime.components,
                    "pytorch": "2.9.2",
                }
            }
        )
    )
    request = request_for(snapshot, runtime=runtime)
    with pytest.raises(MemoryProfileError):
        resolve(snapshot, request=request)
    allow = VersionRule(
        component="pytorch",
        source="==2.9.1",
        target=">=2.9.1,<2.10",
        operator="same_minor_patch_gte",
        effect="allow",
    )
    policy = snapshot.policy.model_copy(update={"version_rules": [allow]})
    assert (
        resolve(replace(snapshot, policy=policy), request=request)["status"]
        == "success"
    )
    policy = policy.model_copy(
        update={"version_rules": [allow, allow.model_copy(update={"effect": "deny"})]}
    )
    with pytest.raises(MemoryProfileError):
        resolve(replace(snapshot, policy=policy), request=request)


def test_changed_template_bytes_reject(snapshot):
    corrupt = snapshot.templates[0].model_copy(update={"limitations": []})
    with pytest.raises(MemoryProfileError, match="hash mismatch"):
        resolve(
            replace(snapshot, templates=(corrupt, snapshot.templates[1])),
            request=request_for(snapshot),
        )


class _Document:
    def __init__(self, db, path):
        self.db, self.path = db, path

    def collection(self, name):
        return _Document(self.db, self.path + "/" + name)

    def document(self, name):
        return _Document(self.db, self.path + "/" + name)

    def get(self, transaction=None):
        data = deepcopy(self.db.data.get(self.path))

        class Read:
            exists = data is not None

            def to_dict(self):
                return data

        return Read()


class _Database:
    project = "roboflow-staging"

    def __init__(self):
        self.data = {}

    def collection(self, name):
        return _Document(self, name)

    def create(self, ref, data):
        assert ref.path not in self.data
        self.data[ref.path] = deepcopy(data)

    def update(self, ref, data):
        self.data[ref.path].update(deepcopy(data))

    def set(self, ref, data):
        self.data[ref.path] = deepcopy(data)


@pytest.fixture
def store(monkeypatch):
    db = _Database()
    monkeypatch.setattr(
        "inference_model_manager.memory_profiles.firestore._transaction",
        lambda client, operation, **kwargs: operation(client),
    )
    return FirestoreCatalog(db, project=db.project)


def test_draft_ready_activate_cas_withdraw_and_idempotent_stage(store, snapshot):
    store.stage(snapshot)
    store.stage(snapshot)
    with pytest.raises(MemoryProfileError):
        store.activate("catalog-1", deployment_group="pilot", expected_revision=0)
    store.mark_ready("catalog-1", approval="review/test-only")
    store.activate("catalog-1", deployment_group="pilot", expected_revision=0)
    assert store.read_active("pilot").revision == 1
    with pytest.raises(MemoryProfileError):
        store.activate("catalog-1", deployment_group="pilot", expected_revision=0)
    store.withdraw("catalog-1")
    store.stage(snapshot)  # Must not reset withdrawn state.
    with pytest.raises(MemoryProfileError):
        store.read_active("pilot")
    with pytest.raises(MemoryProfileError):
        store.mark_ready("catalog-1", approval="cannot-restore")


def test_conflicting_immutable_policy_is_not_overwritten(store, snapshot):
    store.stage(snapshot)
    with pytest.raises(MemoryProfileError, match="different contents"):
        store.stage(
            replace(
                snapshot,
                policy=snapshot.policy.model_copy(
                    update={"deployment_policy_id": "production"}
                ),
            )
        )


def test_project_is_explicit(store):
    with pytest.raises(ValueError, match="project differs"):
        FirestoreCatalog(store.client, project="roboflow-platform")
