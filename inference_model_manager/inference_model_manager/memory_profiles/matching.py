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
    RangeEstimate,
    Reference,
    Request,
    Template,
    VersionRule,
    WorkloadRange,
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

    def to_result(self, request_id: str) -> dict:
        """Serialize C16 without a numeric estimate.

        Args:
            request_id: Request correlation identifier.

        Returns:
            Typed failure envelope with retry guidance.
        """
        categories = {
            "memory_publication_invalid": "integrity",
            "memory_evidence_invalid": "integrity",
            "memory_publication_conflict": "publication",
            "unsupported_family_descriptor": "unsupported",
            "environment_metadata_unavailable": "unsupported",
            "memory_access_denied": "authorization",
            "memory_lookup_unavailable": "dependency",
        }
        result = {
            "status": "failure",
            "requestId": request_id,
            "error": {
                "code": self.code,
                "category": categories.get(self.code, "coverage"),
                "message": str(self),
                "retryable": self.retryable,
            },
        }
        return result


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
        if not _REQUIRED_STRUCTURE <= set(family.required_fields) or not {
            "cuda",
            "nvidia.driver",
        } <= set(family.runtime_components):
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
        if sorted(r.id for r in envelope.ranges) != sorted(
            e.range_id for e in template.range_estimates
        ):
            raise MemoryProfileError(
                "memory_publication_invalid", "Each range requires exactly one estimate"
            )
        try:
            for workload_range in envelope.ranges:
                workload_range.profiled_values(family.workload_parameters)
        except ValueError as error:
            raise MemoryProfileError(
                "memory_publication_invalid", str(error)
            ) from error

        source_groups = None
        for estimate in template.range_estimates:
            groups = set()
            for evidence in estimate.evidence:
                if len(evidence.source_files) != 2 or {
                    file.role for file in evidence.source_files
                } != {"batch", "run_metadata"}:
                    raise MemoryProfileError(
                        "memory_evidence_invalid",
                        "Incomplete or duplicate source file references",
                    )
                if set(evidence.runtime.components) != set(family.runtime_components):
                    raise MemoryProfileError(
                        "memory_evidence_invalid",
                        "Evidence lacks the family's exact runtime components",
                    )
                groups.add(
                    (
                        evidence.source_provider,
                        evidence.source_registry_project,
                        evidence.model_id,
                        evidence.package_id,
                        evidence.entrypoint_name,
                    )
                )
            if source_groups is not None and groups != source_groups:
                raise MemoryProfileError(
                    "memory_evidence_invalid",
                    "Every range requires evidence for the same source packages and entrypoints",
                )
            source_groups = groups
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
    template_id: str, estimate: RangeEstimate, *, request: Request, policy: Policy
) -> tuple[bool, list[dict]]:
    checks = []
    for evidence in estimate.evidence:
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
                "templateId": template_id,
                "profileId": evidence.profile_id,
                "mode": policy.runtime_coverage_mode,
                "coverage": statuses,
            }
        )
    return True, checks


@dataclass
class _MeasuredRange:
    template: Template
    ref: Reference
    bounds: WorkloadRange
    estimate: RangeEstimate
    checks: list[dict]


def _select_ranges(candidates, *, request, directions):
    if request.workload is None or request.selection == "maximum":
        peak = max(candidates, key=lambda item: item.estimate.estimated_gpu_peak_bytes)
        if request.workload is None or peak.bounds.contains(
            request.workload, directions=directions
        ):
            return [peak]
        for item in candidates:
            if item.bounds.contains(request.workload, directions=directions):
                return [item, peak]
        raise MemoryProfileError(
            "outside_profile_envelope",
            "No published range covers the supplied workload",
        )

    for index, candidate in enumerate(candidates):
        if candidate.bounds.contains(request.workload, directions=directions):
            break
    else:
        raise MemoryProfileError(
            "outside_profile_envelope",
            "No published range covers the supplied workload",
        )

    if request.selection == "first_containing":
        return [candidate]

    for other in candidates[index + 1 :]:
        if (
            other.bounds.contains(candidate.bounds.lower, directions=directions)
            and other.bounds.contains(candidate.bounds.upper, directions=directions)
            and not (
                candidate.bounds.contains(other.bounds.lower, directions=directions)
                and candidate.bounds.contains(other.bounds.upper, directions=directions)
            )
        ):
            return [candidate, other]
    raise MemoryProfileError(
        "outside_profile_envelope",
        "No strictly broader published range covers the selected range",
    )


def resolve(snapshot: Snapshot, *, request: Request) -> dict[str, Any]:
    """Select a compatible measured range for a resolved model load.

    Authorization and family normalization happen in the caller before this method.
    The family defines exact and directional axes for joint workload containment.

    Args:
        snapshot: One project-local verified publication.
        request: Authorized effective pre-load facts.

    Returns:
        C15 success including the measured range, evidence, limitations and runtime checks.

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
    candidates = []
    for ref, template in zip(snapshot.catalog.templates, snapshot.templates):
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
        estimates = {e.range_id: e for e in template.range_estimates}
        for bounds in envelopes[template.envelope_id].ranges:
            estimate = estimates[bounds.id]
            accepted, checks = _runtime_checks(
                template.id, estimate, request=request, policy=snapshot.policy
            )
            if accepted:
                candidates.append(
                    _MeasuredRange(template, ref, bounds, estimate, checks)
                )

    if not candidates:
        raise MemoryProfileError(
            "no_compatible_template",
            "No template matches the selected package behavior and environment",
        )

    selected = _select_ranges(
        candidates, request=request, directions=family.workload_parameters
    )
    chosen = selected[-1]
    covering = selected[0] if request.selection == "maximum" else chosen
    runtime_checks = [check for item in selected for check in item.checks]
    limitations = {"family_template"}.union(
        *(set(item.template.limitations) for item in selected)
    )
    if any(any(s != "covered" for s in check["coverage"]) for check in runtime_checks):
        limitations.add("runtime_version_unvalidated")
    evidence = {
        json.dumps(e.model_dump(), sort_keys=True): e.model_dump()
        for item in selected
        for e in item.estimate.evidence
    }
    result = {
        "status": "success",
        "requestId": request.request_id,
        "estimatedGpuPeakBytes": max(
            item.estimate.estimated_gpu_peak_bytes for item in selected
        ),
        "packageContext": request.package_context,
        "publication": {
            "project": snapshot.project,
            "controlRevision": snapshot.revision,
            "catalogVersion": snapshot.catalog.id,
            "matchingPolicyVersion": snapshot.policy.id,
        },
        "coveringTemplate": covering.ref.model_dump(),
        "servingEnvelopeId": covering.template.envelope_id,
        "rangeId": covering.bounds.id,
        "contributingRanges": [
            {
                "template": item.ref.model_dump(),
                "envelopeId": item.template.envelope_id,
                "rangeId": item.bounds.id,
                "estimatedGpuPeakBytes": item.estimate.estimated_gpu_peak_bytes,
            }
            for item in selected
        ],
        "selection": request.selection if request.workload is not None else "maximum",
        "evidenceRefs": list(evidence.values()),
        "runtimeCompatibility": runtime_checks,
        "limitations": sorted(limitations),
    }
    return result
