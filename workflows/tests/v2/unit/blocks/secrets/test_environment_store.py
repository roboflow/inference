"""``v2/environment_secrets_store@v1``: sources, output names and the access flag.

Every value here is an invented sentinel, never a real credential.
"""

import pytest
from pydantic import ValidationError
from roboflow_workflows import environment
from roboflow_workflows.v2.blocks.secrets import (
    EnvironmentAccessError,
    EnvironmentSecrets,
    EnvironmentSecretsStoreBlock,
    StaticSecrets,
)
from roboflow_workflows.v2.data.secrets import SECRET_KIND, Secret
from roboflow_workflows.v2.engine.declaration import spec_of

SENTINEL = "sk-test-SENTINEL-7f3a"
FLAG = "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES"


def test_outputs_are_lowercased_secrets_named_by_the_original_name() -> None:
    # given
    block = EnvironmentSecretsStoreBlock(
        secrets=StaticSecrets({"SLACK_TOKEN": SENTINEL})
    )

    # when
    outputs = block.run(variables_storing_secrets=["SLACK_TOKEN", "UNSET_TOKEN"])

    # then
    assert list(outputs) == ["slack_token", "unset_token"]
    assert isinstance(outputs["slack_token"], Secret)
    assert outputs["slack_token"].name == "SLACK_TOKEN"
    assert outputs["slack_token"].reveal() == SENTINEL
    assert outputs["unset_token"] is None


def test_describe_outputs_declares_one_secret_output_per_name() -> None:
    # given
    params = EnvironmentSecretsStoreBlock.Params(
        variables_storing_secrets=["MY_API_KEY", "Other-Key"]
    )

    # when
    outputs = EnvironmentSecretsStoreBlock.describe_outputs(params)

    # then
    assert list(outputs) == ["my_api_key", "other-key"]
    assert all(output.kinds == (SECRET_KIND,) for output in outputs.values())


def test_the_block_is_prunable_and_reads_names_literally() -> None:
    # when
    spec = spec_of(EnvironmentSecretsStoreBlock)

    # then
    assert spec.prunable
    assert spec.output_fields == ("variables_storing_secrets",)


@pytest.mark.parametrize(
    "names, message",
    [
        ([], "at least 1"),
        (["API_KEY", "api_key"], "'API_KEY' and 'api_key' both become output"),
        (["API_KEY", "API_KEY"], "both become output 'api_key'"),
        (["MY.KEY"], "'MY.KEY' cannot name an output"),
        ([""], "'' cannot name an output"),
    ],
)
def test_names_that_cannot_be_distinct_outputs_are_rejected(names, message) -> None:
    # when
    with pytest.raises(ValidationError) as raised:
        EnvironmentSecretsStoreBlock.Params(variables_storing_secrets=names)

    # then
    assert message in str(raised.value)


def test_static_secrets_never_show_values() -> None:
    # given
    source = StaticSecrets({"SLACK_TOKEN": SENTINEL})

    # when
    text = repr(source)

    # then
    assert text == "StaticSecrets(['SLACK_TOKEN'])"
    assert source.lookup("SLACK_TOKEN") == SENTINEL
    assert source.lookup("MISSING") is None


def test_static_secrets_copy_their_mapping() -> None:
    # given
    values = {"SLACK_TOKEN": SENTINEL}
    source = StaticSecrets(values)

    # when
    values["SLACK_TOKEN"] = "changed"

    # then
    assert source.lookup("SLACK_TOKEN") == SENTINEL


def test_static_secrets_reject_non_string_values_without_showing_them() -> None:
    # when
    with pytest.raises(TypeError) as raised:
        StaticSecrets({"SLACK_TOKEN": 734921})

    # then
    assert "'SLACK_TOKEN'" in str(raised.value)
    assert "734921" not in str(raised.value)


def test_environment_source_reads_the_environment_when_allowed(monkeypatch) -> None:
    # given
    monkeypatch.setattr(environment, FLAG, True)
    monkeypatch.setenv("V2_TEST_SECRET_TOKEN", SENTINEL)
    monkeypatch.delenv("V2_TEST_SECRET_UNSET", raising=False)

    # when
    value = EnvironmentSecrets().lookup("V2_TEST_SECRET_TOKEN")

    # then
    assert value == SENTINEL
    assert EnvironmentSecrets().lookup("V2_TEST_SECRET_UNSET") is None


def test_environment_source_refuses_when_the_v1_flag_is_off(monkeypatch) -> None:
    # given
    monkeypatch.setattr(environment, FLAG, False)
    monkeypatch.setenv("V2_TEST_SECRET_TOKEN", SENTINEL)
    block = EnvironmentSecretsStoreBlock()

    # when
    with pytest.raises(EnvironmentAccessError) as raised:
        block.run(variables_storing_secrets=["V2_TEST_SECRET_TOKEN"])

    # then
    assert FLAG in str(raised.value)
    assert "'secrets' resource" in str(raised.value)
    assert SENTINEL not in str(raised.value)


def test_static_secrets_work_when_the_environment_is_closed(monkeypatch) -> None:
    # given
    monkeypatch.setattr(environment, FLAG, False)
    block = EnvironmentSecretsStoreBlock(
        secrets=StaticSecrets({"SLACK_TOKEN": SENTINEL})
    )

    # when
    outputs = block.run(variables_storing_secrets=["SLACK_TOKEN"])

    # then
    assert outputs["slack_token"].reveal() == SENTINEL
