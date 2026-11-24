"""Closed M1 contracts shared by publication tooling and the admission matcher."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Positive = Annotated[int, Field(strict=True, gt=0)]
LIMITATIONS = frozenset(
    {
        "sampled_peak",
        "bounded_workload",
        "finite_content_samples",
        "family_template",
        "cross_gpu_transfer",
        "relaxed_environment_match",
        "runtime_version_unvalidated",
        "limited_template_validation",
    }
)


class Contract(BaseModel):
    """Reject unknown fields and implicit scalar coercion at storage boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Reference(Contract):
    """An immutable object selected by ID, version and content hash."""

    id: Identifier = Field(description="Logical object identifier.")
    version: Identifier = Field(description="Immutable version identifier.")
    sha256: Digest = Field(description="SHA256 of canonical-json/v1 content.")


WorkloadValue = str | int | FiniteFloat | bool
Direction = Literal["exact", "higher", "lower"]
Selection = Literal["first_containing", "next_containing", "maximum"]


def _equal_axis(left, right):
    # Firestore preserves integers and doubles; both represent numeric settings.
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    return type(left) is type(right) and left == right


class Workload(Contract):
    """One joint observation of the effective request inputs."""

    values: dict[Identifier, WorkloadValue] = Field(
        min_length=1, description="All family-owned input and execution settings."
    )


class WorkloadRange(Contract):
    """Inclusive joint bounds, published in selection order."""

    id: Identifier = Field(description="Range identifier within its envelope.")
    lower: Workload = Field(description="Inclusive lower values for all parameters.")
    upper: Workload = Field(description="Inclusive upper values for all parameters.")

    def profiled_values(
        self, directions: dict[str, Direction]
    ) -> dict[str, WorkloadValue]:
        """Resolve the joint boundary the producer must measure.

        Args:
            directions: Family policy for every parameter in the range.

        Returns:
            Effective values at the profiled boundary.

        Raises:
            ValueError: If bounds are incomplete, reversed, or incorrectly typed.
        """
        if (
            self.lower.values.keys() != directions.keys()
            or self.upper.values.keys() != directions.keys()
        ):
            raise ValueError("Range must define both bounds for every family parameter")

        boundary = {}
        for key, direction in directions.items():
            low, high = self.lower.values[key], self.upper.values[key]
            if direction == "exact":
                if not _equal_axis(low, high):
                    raise ValueError("Exact parameters require identical bounds")
                boundary[key] = low
            elif (
                direction in ("higher", "lower")
                and type(low) in (float, int)
                and type(high) in (float, int)
                and low <= high
            ):
                boundary[key] = high if direction == "higher" else low
            else:
                raise ValueError(
                    "Directional parameters require ordered numeric bounds"
                )

        return boundary

    def contains(self, workload: Workload, *, directions: dict[str, Direction]) -> bool:
        """Check all inputs against this validated range.

        Args:
            workload: Effective inputs to compare.
            directions: Parameter comparison rules from the family.

        Returns:
            Whether every required input is present and within bounds.
        """
        if workload.values.keys() != directions.keys():
            return False

        for key, value in workload.values.items():
            low, high = self.lower.values[key], self.upper.values[key]
            if directions[key] == "exact":
                if not _equal_axis(value, low):
                    return False
            elif type(value) not in (float, int) or not low <= value <= high:
                return False
        return True


class Envelope(Contract):
    """Reviewed workload ranges in order of selection."""

    id: Identifier = Field(description="Envelope identifier.")
    ranges: list[WorkloadRange] = Field(
        min_length=1, description="Ordered joint workload ranges."
    )

    @model_validator(mode="after")
    def unique_range_ids(self) -> "Envelope":
        """Check each range has an unambiguous identifier.

        Returns:
            The checked envelope.

        Raises:
            ValueError: If a range identifier repeats.
        """
        if len({item.id for item in self.ranges}) != len(self.ranges):
            raise ValueError("Duplicate workload range ID")
        return self


class VersionRule(Contract):
    """A reviewed directional runtime version rule, scoped to one component."""

    component: Identifier = Field(description="Used component or inference-models.")
    source: str = Field(
        description="PEP440 source range; explicit lists are exceptions."
    )
    target: str = Field(description="PEP440 target range.")
    source_versions: list[str] = Field(
        default_factory=list, description="Exact source exceptions."
    )
    target_versions: list[str] = Field(
        default_factory=list, description="Exact target exceptions."
    )
    operator: Literal["range", "same_minor_patch_gte", "exact_build"] = Field(
        default="range", description="Explicit comparison algorithm."
    )
    effect: Literal["allow", "deny"] = Field(description="Deny rules take precedence.")


class RuntimeIdentity(Contract):
    """Observed versions of components used by this execution path."""

    components: dict[str, str] = Field(
        description="Selected available runtime probes only."
    )
    inference_models_version: str | None = Field(
        description="Installed distribution version."
    )
    inference_models_build: str | None = Field(
        description="Commit or immutable build identifier."
    )


class EvidenceFile(Contract):
    """A generation-pinned original source file retained with the export."""

    role: Literal["batch", "run_metadata"] = Field(
        description="Producer evidence role."
    )
    bucket: str = Field(min_length=1, description="Source object bucket.")
    object: str = Field(min_length=1, description="Source object path.")
    generation: Annotated[str, Field(pattern=r"^[1-9][0-9]*$")] = Field(
        description="Pinned source generation."
    )
    sha256: Digest = Field(description="Exact source bytes digest.")
    size_bytes: Positive = Field(description="Exact byte count.")


class Evidence(Contract):
    """Pinned unchanged producer export and the selected measurement record."""

    bucket: str = Field(min_length=1, description="Evidence bucket.")
    object: str = Field(min_length=1, description="Export object path.")
    generation: Annotated[str, Field(pattern=r"^[1-9][0-9]*$")] = Field(
        description="Pinned GCS generation."
    )
    export_sha256: Digest = Field(description="Exact export bytes digest.")
    record_sha256: Digest = Field(description="RFC8785 selected record digest.")
    measurement_identity_sha256: Digest = Field(
        description="Producer-validated measurement identity."
    )
    model_id: str = Field(min_length=1, description="Historical measured model ID.")
    package_id: str = Field(min_length=1, description="Historical measured package ID.")
    entrypoint_name: str = Field(min_length=1, description="Measured entrypoint.")
    scenario_name: str = Field(min_length=1, description="Measured scenario.")
    profile_id: str = Field(min_length=1, description="Producer profile identifier.")
    runtime: RuntimeIdentity = Field(
        description="Source runtime identity from the export."
    )
    source_provider: str = Field(
        min_length=1, description="Provider recorded during capture."
    )
    source_registry_project: str = Field(
        min_length=1, description="Registry project recorded during capture."
    )
    environment_snapshot_sha256: Digest = Field(
        description="RFC8785 complete source runtime environment hash."
    )
    source_files: list[EvidenceFile] = Field(
        min_length=2, description="Pinned full-bundle evidence objects."
    )


class RangeEstimate(Contract):
    """Measured peak and source records for one workload range."""

    range_id: Identifier = Field(description="Range in the template envelope.")
    estimated_gpu_peak_bytes: Positive = Field(
        description="Maximum measured process GPU peak for this range."
    )
    evidence: list[Evidence] = Field(
        min_length=1, description="Records contributing to this range."
    )


class Template(Contract):
    """A validated numeric estimate plus closed applicability and evidence."""

    schema_version: Literal["memory-template/v2"] = Field(
        default="memory-template/v2", description="Contract version."
    )
    id: Identifier = Field(description="Template identifier.")
    version: Identifier = Field(description="Immutable template version.")
    family_contract_id: Identifier = Field(
        description="Reviewed family adapter contract."
    )
    structural_key: dict[str, str | int | bool] = Field(
        min_length=1, description="Complete closed family/execution key."
    )
    environment: dict[str, str] = Field(
        min_length=1, description="Exact M1 GPU claim fields."
    )
    envelope_id: Identifier = Field(description="Catalog-owned serving envelope.")
    range_estimates: list[RangeEstimate] = Field(
        min_length=1, description="Peaks and supporting evidence for each range."
    )
    allowed_deployment_policy_id: Identifier = Field(
        description="Trusted deployment cohort authorized to use this template."
    )
    validation_status: Literal["validated"] = Field(
        description="Successful evidence and applicability validation."
    )
    limitations: list[str] = Field(description="Accepted limitation registry tags.")

    @model_validator(mode="after")
    def known_limitations(self) -> "Template":
        """Reject unknown limitation tags.

        Returns:
            The checked template.

        Raises:
            ValueError: If a tag is outside memory-limitations/v1.
        """
        if set(self.limitations) - LIMITATIONS:
            raise ValueError("Unknown limitation tag")
        return self


class Family(Contract):
    """Closed fields and known behavior keys maintained by a family owner."""

    id: Identifier = Field(description="Versioned family contract ID.")
    required_fields: list[str] = Field(
        min_length=1, description="Exact structural field set."
    )
    runtime_components: list[str] = Field(
        min_length=1,
        description="Used components required for this family/backend, including CUDA and driver.",
    )
    behavior_keys: list[str] = Field(
        min_length=1, description="Reviewed pre-load execution behaviors."
    )
    workload_parameters: dict[Identifier, Direction] = Field(
        min_length=1,
        description="Exact match or direction of increasing memory demand for each workload parameter.",
    )


class Policy(Contract):
    """Trusted server policy; callers cannot choose a weaker matching level."""

    id: Identifier = Field(description="Immutable policy version.")
    deployment_policy_id: Identifier = Field(
        description="Authorized deployment cohort."
    )
    environment_fields: list[str] = Field(
        default=["deviceFamily", "memoryModel", "computeCapability", "gpuSku"],
        description="Exact GPU-level fields required by M1.",
    )
    version_rules: list[VersionRule] = Field(
        default_factory=list,
        description="Explicit approved compatibility rules; none enabled by default.",
    )
    runtime_coverage_mode: Literal["observe_only", "enforce"] = Field(
        default="observe_only",
        description="Version coverage only; known incompatibility still rejects.",
    )

    @model_validator(mode="after")
    def gpu_level(self) -> "Policy":
        """Keep M1 at exact GPU matching.

        Returns:
            Checked policy.

        Raises:
            ValueError: If policy attempts an unsupported matching level.
        """
        if set(self.environment_fields) != {
            "deviceFamily",
            "memoryModel",
            "computeCapability",
            "gpuSku",
        }:
            raise ValueError("M1 supports GPU matching only")
        return self


class Catalog(Contract):
    """Immutable catalog; template order is published priority, never byte size."""

    id: Identifier = Field(description="Catalog version.")
    templates: list[Reference] = Field(
        min_length=1, description="Priority-ordered designated template versions."
    )
    envelopes: list[Envelope] = Field(
        min_length=1, description="Pinned finite workload definitions."
    )
    families: list[Family] = Field(
        min_length=1, description="Pinned closed family definitions."
    )


class Request(Contract):
    """Authorized pre-load facts supplied by a server-side family adapter."""

    request_id: str = Field(description="Correlation ID.")
    package_context: dict[str, str] = Field(
        description="Provider/project/canonical model/package identifiers."
    )
    family_contract_id: str = Field(description="Resolved family contract.")
    structural_key: dict[str, str | int | bool] = Field(
        description="Known effective settings and structural facts."
    )
    environment: dict[str, str] = Field(
        description="Current selected GPU observations."
    )
    runtime: RuntimeIdentity = Field(description="Serving runtime identity.")
    workload: Workload | None = Field(
        default=None,
        description="Optional normalized joint workload; absent selects the maximum.",
    )
    selection: Selection = Field(
        default="first_containing", description="Requested workload range selection."
    )
