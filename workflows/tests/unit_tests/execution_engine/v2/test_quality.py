"""Compile-time quality: ``block > workflow > deployment`` with provenance.

An implementation lists the labels it serves. The root ``execution`` section
sets a label per step (``step_quality``) or for the workflow (``quality``);
``CompileOptions.quality`` is the deployment default. The first level naming a
label a fitting implementation serves wins and the plan records which. A
step-level label nothing serves is an error; a lower level is recorded as
ignored. Legacy blocks without labels select exactly as before.
"""

import pytest
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
    spec_of,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DeclarationError,
    SelectorError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.kinds import STRING_KIND
from roboflow_workflows.execution_engine.v2.plan import CompileOptions
from roboflow_workflows.execution_engine.v2.targets import (
    ImplementationChoice,
    QualityChoice,
    QualityRequest,
    QualitySettings,
    Target,
    UnsupportedQualityError,
    check_choice,
    consider,
    select_implementation,
)

from tests.unit_tests.execution_engine.v2.m7_demand.blocks import (
    CATALOGUE,
    Accurate,
    Fast,
    GpuModel,
    Labeled,
    Legacy,
    Model,
    definition,
    nested,
    step,
)


def compiled(
    steps,
    outputs,
    *,
    execution=None,
    quality=None,
    target=Target.cpu(),
    **sections,
):
    definition_ = definition(steps, outputs, **sections)
    if execution is not None:
        definition_["execution"] = execution
    plan = compile_workflow(
        definition_,
        catalogue=CATALOGUE,
        options=CompileOptions(quality=quality, target=target),
    )

    return plan


def label_of(plan, value: float = 1.0) -> dict:
    rows = plan.create_session().run({"value": value}).rows()

    return rows[0]


MODEL = [step(Model, "model", value="$inputs.value")]
OUTPUT = {"label": "$steps.model.label"}


# Precedence -----------------------------------------------------------------


def test_without_any_request_the_first_fitting_implementation_wins_as_before():
    plan = compiled(MODEL, OUTPUT)
    choice = plan.step(("model",)).implementation

    assert choice.name == "accurate"
    assert choice.quality is None and choice.request.is_empty
    assert choice.describe() == {
        "name": "accurate",
        "target": ["cpu"],
        "considered": [
            {"name": "accurate", "missing": [], "selected": True},
            {"name": "fast", "missing": [], "selected": False},
        ],
    }
    assert "quality" not in plan.describe()
    assert label_of(plan) == {"label": "accurate@None"}


@pytest.mark.parametrize(
    ("execution", "deployment", "expected", "level"),
    [
        (
            {"step_quality": {"$steps.model": "fast"}, "quality": "accurate"},
            "accurate",
            "fast",
            "step",
        ),
        ({"quality": "fast"}, "accurate", "fast", "workflow"),
        (None, "fast", "fast", "deployment"),
        ({"step_quality": {"$steps.model": "balanced"}}, "fast", "accurate", "step"),
        ({"quality": "balanced"}, "fast", "accurate", "workflow"),
    ],
)
def test_block_beats_workflow_beats_deployment_and_the_plan_records_the_level(
    execution, deployment, expected, level
):
    plan = compiled(MODEL, OUTPUT, execution=execution, quality=deployment)
    choice = plan.step(("model",)).implementation

    assert choice.name == expected
    assert choice.quality.level == level
    assert choice.describe()["quality"]["selected"] == {
        "label": choice.quality.label,
        "level": level,
    }
    assert choice.ignored == ()
    assert label_of(plan) == {"label": f"{expected}@{choice.quality.label}"}


def test_step_quality_reaches_nested_block_steps_by_their_full_path():
    child = definition(MODEL, OUTPUT)
    plan = compiled(
        [nested("child", child, value="$inputs.value")],
        {"label": "$steps.child.label"},
        execution={"step_quality": {"$steps.child/model": "fast"}},
    )

    assert plan.step(("child", "model")).implementation.name == "fast"
    assert plan.quality.describe() == {
        "workflow": None,
        "steps": {"$steps.child/model": "fast"},
    }
    assert plan.describe()["quality"] == plan.quality.describe()


# Unsupported labels ---------------------------------------------------------


def test_step_level_label_nothing_serves_is_a_compile_error_listing_what_is_served():
    with pytest.raises(UnsupportedQualityError, match="sets quality 'draft'") as raised:
        compiled(MODEL, OUTPUT, execution={"step_quality": {"$steps.model": "draft"}})

    assert raised.value.step_path == ("model",)
    assert raised.value.served == {
        "accurate": ("accurate", "balanced"),
        "fast": ("fast",),
    }
    assert "'accurate' serves ['accurate', 'balanced']" in str(raised.value)


def test_step_level_label_on_a_legacy_block_is_an_error_not_a_silent_no_op():
    with pytest.raises(UnsupportedQualityError, match="declares quality labels"):
        compiled(
            [step(Legacy, "legacy", value="$inputs.value")],
            {"label": "$steps.legacy.label"},
            execution={"step_quality": {"$steps.legacy": "fast"}},
        )


def test_workflow_and_deployment_labels_nothing_serves_fall_through_and_are_recorded():
    plan = compiled(
        [
            step(Legacy, "legacy", value="$inputs.value"),
            step(Model, "model", value="$inputs.value"),
            step(Labeled, "labeled", value="$inputs.value"),
        ],
        {
            "legacy": "$steps.legacy.label",
            "model": "$steps.model.label",
            "labeled": "$steps.labeled.label",
        },
        execution={"quality": "draft"},
        quality="fast",
    )

    legacy = plan.step(("legacy",)).implementation
    model = plan.step(("model",)).implementation
    labeled = plan.step(("labeled",)).implementation
    assert legacy.name == "default" and legacy.quality is None
    assert [item.describe() for item in legacy.ignored] == [
        {
            "label": "draft",
            "level": "workflow",
            "reason": "no implementation of the block declares quality labels",
        },
        {
            "label": "fast",
            "level": "deployment",
            "reason": "no implementation of the block declares quality labels",
        },
    ]
    assert model.name == "fast" and model.quality == QualityChoice("fast", "deployment")
    assert [(item.label, item.level) for item in model.ignored] == [
        ("draft", "workflow")
    ]
    assert labeled.quality == QualityChoice("fast", "deployment")
    assert label_of(plan) == {
        "legacy": "legacy@None",
        "model": "fast@fast",
        "labeled": "labeled@fast",
    }


def test_target_and_quality_combine_a_label_served_only_off_target():
    steps = [step(GpuModel, "model", value="$inputs.value")]

    with pytest.raises(UnsupportedQualityError, match="do not fit the compile target"):
        compiled(steps, OUTPUT, execution={"step_quality": {"$steps.model": "fast"}})

    hinted = compiled(steps, OUTPUT, execution={"quality": "fast"})
    on_cuda = compiled(
        steps,
        OUTPUT,
        execution={"quality": "fast"},
        target=Target(frozenset({"cpu", "cuda"})),
    )

    choice = hinted.step(("model",)).implementation
    assert choice.name == "accurate" and choice.quality is None
    assert choice.ignored[0].reason == (
        "the implementation(s) ['cuda-fast'] serving 'fast' do not fit the compile "
        "target"
    )
    assert on_cuda.step(("model",)).implementation.name == "cuda-fast"


# Declarations ---------------------------------------------------------------


def test_quality_labels_are_literals_never_selectors_and_name_block_steps():
    with pytest.raises(WorkflowCompileError, match="must be a literal"):
        compiled(MODEL, OUTPUT, execution={"quality": "$inputs.quality"})
    with pytest.raises(WorkflowCompileError, match="must be a literal"):
        compiled(
            MODEL, OUTPUT, execution={"step_quality": {"$steps.model": "$inputs.q"}}
        )
    with pytest.raises(WorkflowCompileError, match="quality label of letters"):
        compiled(MODEL, OUTPUT, execution={"quality": 3})
    with pytest.raises(WorkflowCompileError, match="unsupported keys"):
        compiled(MODEL, OUTPUT, execution={"profile": "fast"})
    with pytest.raises(WorkflowCompileError, match="step selectors"):
        compiled(MODEL, OUTPUT, execution={"step_quality": {"model": "fast"}})
    with pytest.raises(SelectorError, match="not a block step") as raised:
        compiled(MODEL, OUTPUT, execution={"step_quality": {"$steps.other": "fast"}})
    assert raised.value.field_path == ("execution", "step_quality", "$steps.other")
    with pytest.raises(SelectorError, match="nested workflow steps set quality"):
        compiled(
            [nested("child", definition(MODEL, OUTPUT), value="$inputs.value")],
            {"label": "$steps.child.label"},
            execution={"step_quality": {"$steps.child": "fast"}},
        )


def test_execution_settings_are_root_only():
    child = definition(MODEL, OUTPUT)
    child["execution"] = {"quality": "fast"}

    with pytest.raises(WorkflowCompileError, match="root-only"):
        compiled(
            [nested("child", child, value="$inputs.value")],
            {"label": "$steps.child.label"},
        )


def test_compile_options_quality_must_be_a_label():
    with pytest.raises(ContractError, match="non-empty label"):
        CompileOptions(quality="")


def test_implementation_quality_declarations_are_validated():
    with pytest.raises(DeclarationError, match="quality must be a tuple"):

        class BadLabels(Implementation):
            name = "bad"
            quality = "fast"

            def run(self, *, value):
                return {"label": "x"}

        class Host(Block):
            type = "test/m7_bad_host@v1"
            outputs = {"label": Output(STRING_KIND)}
            implementations = (BadLabels,)

            class Params(BlockParams):
                value: Ref()

    with pytest.raises(
        DeclarationError, match="quality belongs to each Implementation"
    ):

        class Contract(Block):
            type = "test/m7_bad_contract@v1"
            quality = ("fast",)
            outputs = {"label": Output(STRING_KIND)}
            implementations = (Fast, Accurate)

            class Params(BlockParams):
                value: Ref()

    with pytest.raises(DeclarationError, match="restates contract attribute"):

        class Restating(Implementation):
            name = "restating"
            prunable = True

            def run(self, *, value):
                return {"label": "x"}

        class Host2(Block):
            type = "test/m7_bad_host2@v1"
            outputs = {"label": Output(STRING_KIND)}
            implementations = (Restating,)

            class Params(BlockParams):
                value: Ref()

    assert spec_of(Model).implementations[0].quality == frozenset(
        {"accurate", "balanced"}
    )
    assert spec_of(Model).implementations[0].describe()["quality"] == [
        "accurate",
        "balanced",
    ]
    assert "quality" not in spec_of(Legacy).implementations[0].describe()
    assert spec_of(Labeled).implementations[0].serves("fast")


# Plan integrity -------------------------------------------------------------


def test_a_forged_quality_choice_is_rejected_by_the_plan_checks():
    spec = spec_of(Model)
    request = QualityRequest(workflow="fast")
    forged = ImplementationChoice(
        spec=spec.implementations[0],
        target=Target.cpu(),
        considered=consider(spec, target=Target.cpu()),
        request=request,
        quality=QualityChoice("fast", "workflow"),
    )

    with pytest.raises(ContractError, match="is not the selection"):
        check_choice(spec, forged)

    genuine = select_implementation(
        spec, target=Target.cpu(), step_path=("model",), quality=request
    )
    check_choice(spec, genuine)
    assert genuine.spec is spec.implementations[1]


def test_quality_settings_build_the_request_of_each_step():
    settings = QualitySettings(workflow="balanced", steps={("child", "model"): "fast"})

    assert settings.request_for(
        ("child", "model"), deployment="draft"
    ) == QualityRequest(step="fast", workflow="balanced", deployment="draft")
    assert settings.request_for(("other",), deployment=None) == QualityRequest(
        workflow="balanced"
    )
    assert QualityRequest().levels() == ()
    assert QualityRequest(deployment="fast").levels() == (("deployment", "fast"),)
    with pytest.raises(ContractError, match="non-empty label"):
        QualityRequest(step="")
