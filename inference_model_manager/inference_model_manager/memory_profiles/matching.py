"""Pure matching against one immutable publication snapshot; no model loading."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from inference_model_manager.memory_profiles.contracts import (
    Catalog,
    Contract,
    Policy,
    Request,
    Template,
    VersionRule,
)

_REQUIRED_STRUCTURE = {
    "architecture",
    "modelVariant",
    "taskType",
    "backendType",
    "quantization",
    "batchPolicy",
    "networkShapePolicyId",
    "entrypointMethod",
    "memoryBehaviorKey",
    "processLayout",
    "executionConstraintsId",
    "familyConstraintsId",
}


class MemoryProfileError(ValueError):
    """Typed failure with no numeric estimate.

    Args:
        code: Stable machine-readable reason.
        message: Actionable diagnostic.
        retryable: Whether a later metadata read can succeed without configuration changes.
    """

    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class Snapshot:
    """Verified project-local publication read for one lookup.

    Args:
        project: Firestore project.
        revision: Active control revision.
        catalog: Verified catalog.
        policy: Trusted active policy.
        templates: Verified template contents in catalog priority order.
    """

    project: str
    revision: int
    catalog: Catalog
    policy: Policy
    templates: tuple[Template, ...]


def content_digest(value: Contract) -> str:
    """Hash canonical-json/v1 storage content (distinct from producer identity).

    Args:
        value: Validated storage object containing JSON values.

    Returns:
        Hexadecimal SHA256 of sorted compact UTF-8 JSON.
    """
    data = json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(data.encode()).hexdigest()
    return digest


def _known(values):
    return all(
        value is not None and value not in ("", "unknown", "unavailable")
        for value in values.values()
    )


def _same(left, right):
    # bool and int are distinct contract values even though False == 0 in Python.
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def validate_snapshot(snapshot: Snapshot) -> None:
    """Check pinned references, family closure and required evidence coverage.

    Args:
        snapshot: Fully fetched catalog/policy/template set.

    Raises:
        MemoryProfileError: If definitions or evidence references are inconsistent.
    """
    catalog = snapshot.catalog
    envelopes = {e.id: e for e in catalog.envelopes}
    families = {f.id: f for f in catalog.families}
    if len(envelopes) != len(catalog.envelopes) or len(families) != len(
        catalog.families
    ):
        raise MemoryProfileError("memory_publication_invalid", "Duplicate definitions")
    if len(snapshot.templates) != len(catalog.templates):
        raise MemoryProfileError(
            "memory_publication_invalid", "Incomplete template set"
        )
    if len({(r.id, r.version) for r in catalog.templates}) != len(catalog.templates):
        raise MemoryProfileError(
            "memory_publication_invalid", "Duplicate template references"
        )
    for family in families.values():
        if (
            family.default_envelope_id not in envelopes
            or not _REQUIRED_STRUCTURE <= set(family.required_fields)
            or not {"cuda", "nvidia.driver"} <= set(family.runtime_components)
        ):
            raise MemoryProfileError(
                "memory_publication_invalid", "Incomplete family definition"
            )
    for ref, template in zip(catalog.templates, snapshot.templates):
        if (ref.id, ref.version, ref.sha256) != (
            template.id,
            template.version,
            content_digest(template),
        ):
            raise MemoryProfileError(
                "memory_publication_invalid", "Template reference/hash mismatch"
            )
        family = families.get(template.family_contract_id)
        envelope = envelopes.get(template.envelope_id)
        if (
            family is None
            or envelope is None
            or set(template.structural_key) != set(family.required_fields)
            or not _known(template.structural_key)
        ):
            raise MemoryProfileError(
                "memory_publication_invalid",
                "Dangling definition or incomplete structural key",
            )
        if (
            template.structural_key.get("memoryBehaviorKey") not in family.behavior_keys
            or template.structural_key.get("processLayout") != "one-new-model-process"
        ):
            raise MemoryProfileError(
                "memory_publication_invalid",
                "Unknown behavior or unsupported process layout",
            )
        if set(template.environment) != set(
            snapshot.policy.environment_fields
        ) or not all(template.environment.values()):
            raise MemoryProfileError(
                "memory_publication_invalid", "Incomplete GPU claim"
            )
        # Each source package/entrypoint in the reviewed derivation must have ALL cases.
        groups: dict[tuple[str, str, str], set[str]] = {}
        for evidence in template.evidence:
            if {file.role for file in evidence.source_files} != {
                "batch",
                "run_metadata",
            } or len(evidence.source_files) != 2:
                raise MemoryProfileError(
                    "memory_evidence_invalid",
                    "Incomplete or duplicate source file references",
                )
            group = (evidence.model_id, evidence.package_id, evidence.entrypoint_name)
            groups.setdefault(group, set()).add(evidence.scenario_name)
        if any(
            not set(envelope.required_scenario_names) <= names
            for names in groups.values()
        ):
            raise MemoryProfileError(
                "memory_evidence_invalid", "Required source scenarios are missing"
            )
    for rule in snapshot.policy.version_rules:
        try:
            SpecifierSet(rule.source)
            SpecifierSet(rule.target)
        except InvalidSpecifier as exc:
            raise MemoryProfileError(
                "memory_publication_invalid", "Invalid version range"
            ) from exc
        if not rule.source and not rule.source_versions:
            raise MemoryProfileError(
                "memory_publication_invalid", "Unbounded source version rule"
            )
        if not rule.target and not rule.target_versions:
            raise MemoryProfileError(
                "memory_publication_invalid", "Unbounded target version rule"
            )


def _rule_matches(rule: VersionRule, source: str, target: str) -> bool:
    if rule.operator == "exact_build":
        return source in rule.source_versions and target in rule.target_versions
    try:
        src, dst = Version(source), Version(target)
        source_match = (
            source in rule.source_versions
            or bool(rule.source)
            and SpecifierSet(rule.source).contains(src, prereleases=True)
        )
        target_match = (
            target in rule.target_versions
            or bool(rule.target)
            and SpecifierSet(rule.target).contains(dst, prereleases=True)
        )
        if rule.operator == "same_minor_patch_gte":
            return bool(
                source_match
                and target_match
                and not src.is_prerelease
                and not dst.is_prerelease
                and src.release[:2] == dst.release[:2]
                and dst >= src
            )
        return bool(source_match and target_match)
    except InvalidVersion:
        return False


def _coverage(
    component: str, source: str | None, target: str | None, policy: Policy
) -> str:
    if not source or not target:
        return "unknown"
    applicable = [
        r
        for r in policy.version_rules
        if r.component == component and _rule_matches(r, source, target)
    ]
    if any(r.effect == "deny" for r in applicable):
        return "incompatible"
    if any(r.effect == "allow" for r in applicable):
        return "covered"
    if component.endswith("build"):
        return "covered" if source == target else "uncovered"
    try:
        return "covered" if Version(source) == Version(target) else "uncovered"
    except InvalidVersion:
        return "unknown"


def _runtime_checks(
    template: Template, request: Request, policy: Policy
) -> tuple[bool, list[dict]]:
    checks = []
    for evidence in template.evidence:
        source = evidence.runtime
        if not source.components or not set(source.components) <= set(
            request.runtime.components
        ):
            return False, []
        for name, version in source.components.items():
            if (
                _coverage(name, version, request.runtime.components[name], policy)
                != "covered"
            ):
                return False, []
        statuses = [
            _coverage(
                "inference-models",
                source.inference_models_version,
                request.runtime.inference_models_version,
                policy,
            ),
            _coverage(
                "inference-models-build",
                source.inference_models_build,
                request.runtime.inference_models_build,
                policy,
            ),
        ]
        if (
            "incompatible" in statuses
            or policy.runtime_coverage_mode == "enforce"
            and any(s != "covered" for s in statuses)
        ):
            return False, []
        checks.append(
            {
                "templateId": template.id,
                "profileId": evidence.profile_id,
                "mode": policy.runtime_coverage_mode,
                "coverage": statuses,
            }
        )
    return True, checks


def resolve(snapshot: Snapshot, *, request: Request) -> dict[str, Any]:
    """Resolve the default floor and a covering template without shrinking coverage.

    Authorization and family normalization happen in the caller before this method.
    Only finite, explicitly validated correlated workload points are supported in M1.

    Args:
        snapshot: One project-local verified publication.
        request: Authorized effective pre-load facts.

    Returns:
        C15 success including both source references, limitations and runtime checks.

    Raises:
        MemoryProfileError: Typed C16 failure, never a zero-valued estimate.
    """
    validate_snapshot(snapshot)
    family = next(
        (f for f in snapshot.catalog.families if f.id == request.family_contract_id),
        None,
    )
    if (
        family is None
        or set(request.structural_key) != set(family.required_fields)
        or not _known(request.structural_key)
        or request.structural_key.get("memoryBehaviorKey") not in family.behavior_keys
        or request.structural_key.get("processLayout") != "one-new-model-process"
    ):
        raise MemoryProfileError(
            "unsupported_family_descriptor",
            "Required structural facts or execution behavior are unknown",
        )
    if any(
        not request.environment.get(field)
        for field in snapshot.policy.environment_fields
    ):
        raise MemoryProfileError(
            "environment_metadata_unavailable",
            "Required runtime GPU observations are missing",
        )
    envelopes = {e.id: e for e in snapshot.catalog.envelopes}
    default = envelopes[family.default_envelope_id]
    compatible = []
    checks = {}
    for template in snapshot.templates:
        if (
            template.family_contract_id != family.id
            or not _same(template.structural_key, request.structural_key)
            or template.allowed_deployment_policy_id
            != snapshot.policy.deployment_policy_id
        ):
            continue
        if any(
            template.environment[field] != request.environment[field]
            for field in snapshot.policy.environment_fields
        ):
            continue
        accepted, details = _runtime_checks(template, request, snapshot.policy)
        if accepted:
            compatible.append(template)
            checks[(template.id, template.version)] = details
    if not compatible:
        raise MemoryProfileError(
            "no_compatible_template",
            "No template matches the selected package behavior and environment",
        )

    def covering(points):
        for candidate in compatible:
            if all(
                any(
                    _same(p.values, w.values)
                    for w in envelopes[candidate.envelope_id].workloads
                )
                for p in points
            ):
                return candidate
        return None

    floor = covering(default.workloads)
    selected = covering(default.workloads + request.input_hints)
    if floor is None:
        raise MemoryProfileError(
            "serving_band_not_defined", "No compatible template covers the default band"
        )
    if selected is None:
        raise MemoryProfileError(
            "outside_profile_envelope",
            "No template covers the default band plus requested hints",
        )
    contributions = [floor] if selected == floor else [floor, selected]
    runtime_checks = [
        check for t in contributions for check in checks[(t.id, t.version)]
    ]
    limitations = {"family_template"}.union(
        *(set(t.limitations) for t in contributions)
    )
    if any(any(s != "covered" for s in check["coverage"]) for check in runtime_checks):
        limitations.add("runtime_version_unvalidated")
    refs = {
        t.id + ":" + t.version: r.model_dump()
        for r, t in zip(snapshot.catalog.templates, snapshot.templates)
    }
    evidence = {
        json.dumps(e.model_dump(), sort_keys=True): e.model_dump()
        for t in contributions
        for e in t.evidence
    }
    result = {
        "status": "success",
        "requestId": request.request_id,
        "estimatedGpuPeakBytes": max(
            floor.estimated_gpu_peak_bytes, selected.estimated_gpu_peak_bytes
        ),
        "packageContext": request.package_context,
        "publication": {
            "project": snapshot.project,
            "controlRevision": snapshot.revision,
            "catalogVersion": snapshot.catalog.id,
            "matchingPolicyVersion": snapshot.policy.id,
        },
        "defaultTemplate": refs[floor.id + ":" + floor.version],
        "coveringTemplate": refs[selected.id + ":" + selected.version],
        "servingEnvelopeId": selected.envelope_id,
        "evidenceRefs": list(evidence.values()),
        "runtimeCompatibility": runtime_checks,
        "limitations": sorted(limitations),
    }
    return result
