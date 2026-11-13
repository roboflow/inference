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
    Reference,
    Request,
    RuntimeIdentity,
    Template,
    VersionRule,
    Workload,
)
from inference_model_manager.memory_profiles.firestore import FirestoreCatalog
from inference_model_manager.memory_profiles.matching import (
    MemoryProfileError,
    Snapshot,
    content_digest,
    resolve,
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
    envelope = Envelope(
        id="default",
        workloads=[Workload(values={"width": 3840, "batch": 1})],
        required_scenario_names=["base"],
    )
    wider = Envelope(
        id="wide",
        workloads=envelope.workloads + [Workload(values={"width": 3840, "batch": 32})],
        required_scenario_names=["base", "wide"],
    )
    first = Template(
        id="floor",
        version="1",
        family_contract_id="rfdetr-seg-v1",
        structural_key=structural,
        environment=environment,
        envelope_id="default",
        estimated_gpu_peak_bytes=9000,
        allowed_deployment_policy_id="staging",
        validation_status="validated",
        evidence=[evidence],
        limitations=["sampled_peak", "bounded_workload"],
    )
    second = first.model_copy(
        update={
            "id": "covering",
            "envelope_id": "wide",
            "estimated_gpu_peak_bytes": 8000,
            "evidence": [
                evidence,
                evidence.model_copy(
                    update={"scenario_name": "wide", "profile_id": "wide-profile"}
                ),
            ],
            "limitations": ["finite_content_samples"],
        }
    )
    family = Family(
        id="rfdetr-seg-v1",
        required_fields=list(structural),
        runtime_components=list(runtime.components),
        behavior_keys=[structural["memoryBehaviorKey"]],
        default_envelope_id="default",
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
        runtime=template.evidence[0].runtime,
    )
    values.update(changes)
    return Request(**values)


def test_floor_retains_both_refs_and_does_not_select_lowest_bytes(snapshot):
    default = resolve(snapshot, request=request_for(snapshot))
    assert default["defaultTemplate"]["id"] == "floor"
    assert default["coveringTemplate"]["id"] == "floor"
    expanded = resolve(
        snapshot,
        request=request_for(
            snapshot, input_hints=[Workload(values={"width": 3840, "batch": 32})]
        ),
    )
    assert expanded["estimatedGpuPeakBytes"] == 9000
    assert expanded["coveringTemplate"]["id"] == "covering"
    assert "finite_content_samples" in expanded["limitations"]
    assert len(expanded["evidenceRefs"]) == 2


@pytest.mark.parametrize(
    "values",
    [
        {"width": 1920, "batch": 32},
        {"width": 3840, "batch": 8},
        {"width": 7680, "batch": 1},
    ],
)
def test_unmeasured_interiors_and_cross_products_are_not_coverage(snapshot, values):
    with pytest.raises(MemoryProfileError, match="No template covers") as error:
        resolve(
            snapshot,
            request=request_for(snapshot, input_hints=[Workload(values=values)]),
        )
    assert error.value.code == "outside_profile_envelope"
    assert not hasattr(error.value, "estimated_gpu_peak_bytes")


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


def test_uncovered_runtime_is_observe_only_but_known_incompatibility_rejects(snapshot):
    runtime = (
        snapshot.templates[0]
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
        .evidence[0]
        .runtime.model_copy(
            update={
                "components": {
                    **snapshot.templates[0].evidence[0].runtime.components,
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
    corrupt = snapshot.templates[0].model_copy(update={"estimated_gpu_peak_bytes": 1})
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
