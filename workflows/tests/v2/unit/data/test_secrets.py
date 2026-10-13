"""The ``Secret`` carrier, the ``secret`` kind, reveal/scrub and the record codec.

Every value here is an invented sentinel, never a real credential.
"""

import copy
import json
import logging
import pickle

import pytest
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Revealed,
    Secret,
    SecretUnavailableError,
    decode_secret_record,
    encode_secret_record,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.errors import WorkflowInputError
from roboflow_workflows.v2.engine.execution.inputs import decode_payload

SENTINEL = "sk-test-SENTINEL-7f3a"
OTHER_SENTINEL = "sk-test-OTHER-91c2"


def test_reveal_returns_the_value_and_every_text_form_shows_only_the_name() -> None:
    # given
    secret = Secret(SENTINEL, name="SLACK_TOKEN")

    # when
    texts = [repr(secret), str(secret), f"{secret}", f"{secret:>40}", "%s" % secret]

    # then
    assert secret.reveal() == SENTINEL
    assert secret.name == "SLACK_TOKEN"
    assert secret.available
    assert texts[0] == "Secret('SLACK_TOKEN')"
    assert texts[1:] == ["<secret SLACK_TOKEN>"] * 4
    assert all(SENTINEL not in text for text in texts)


def test_containers_and_json_default_str_show_only_the_name() -> None:
    # given
    secret = Secret(SENTINEL, name="SLACK_TOKEN")
    nested = {"headers": [("Authorization", secret)]}

    # when
    texts = [repr(nested), str(nested), json.dumps(nested, default=str)]

    # then
    assert all(SENTINEL not in text and "SLACK_TOKEN" in text for text in texts)


def test_json_dumps_without_default_fails_without_the_value() -> None:
    # given
    secret = Secret(SENTINEL, name="SLACK_TOKEN")

    # when
    with pytest.raises(TypeError) as raised:
        json.dumps({"token": secret})

    # then
    assert SENTINEL not in str(raised.value)


def test_logging_and_exception_text_show_only_the_name(caplog) -> None:
    # given
    secret = Secret(SENTINEL, name="SLACK_TOKEN")
    logger = logging.getLogger("tests.v2.secrets")

    # when
    with caplog.at_level(logging.INFO, logger="tests.v2.secrets"):
        logger.info("calling with %s and %r", secret, secret)
        try:
            raise RuntimeError(f"request with {secret} failed")
        except RuntimeError:
            logger.exception("failure for %s", {"token": secret})

    # then
    assert "SLACK_TOKEN" in caplog.text
    assert SENTINEL not in caplog.text


def test_pickling_is_refused_and_copies_are_the_same_object() -> None:
    # given
    secret = Secret(SENTINEL, name="SLACK_TOKEN")
    nested = {"headers": {"token": secret}}

    # when
    with pytest.raises(TypeError, match="cannot be pickled") as raised:
        pickle.dumps(nested)
    copied = copy.deepcopy(nested)

    # then
    assert SENTINEL not in str(raised.value)
    assert copy.copy(secret) is secret
    assert copied["headers"]["token"] is secret
    assert copied is not nested
    assert secret == secret
    assert Secret(SENTINEL, name="SLACK_TOKEN") != secret


def test_an_unavailable_secret_raises_on_reveal_naming_only_the_secret() -> None:
    # given
    secret = Secret.unavailable("SLACK_TOKEN")

    # when
    with pytest.raises(SecretUnavailableError) as raised:
        secret.reveal()

    # then
    assert not secret.available
    assert repr(secret) == "Secret('SLACK_TOKEN', unavailable)"
    assert str(secret) == "<secret SLACK_TOKEN>"
    assert "'SLACK_TOKEN'" in str(raised.value)
    assert "recordings and replays keep only its name" in str(raised.value)


@pytest.mark.parametrize(
    "value, name",
    [
        (734921, "TOKEN"),
        (b"sk-test-SENTINEL-7f3a", "TOKEN"),
        (SENTINEL, ""),
        (SENTINEL, 3),
    ],
)
def test_construction_errors_never_contain_the_value(value, name) -> None:
    # when
    with pytest.raises(TypeError) as raised:
        Secret(value, name=name)

    # then
    assert SENTINEL not in str(raised.value)
    assert "734921" not in str(raised.value)


def test_to_secret_wraps_strings_and_keeps_secrets_and_none() -> None:
    # given
    secret = Secret(SENTINEL, name="SLACK_TOKEN")

    # when
    wrapped = to_secret(OTHER_SENTINEL, name="api_key")

    # then
    assert isinstance(wrapped, Secret)
    assert wrapped.name == "api_key"
    assert wrapped.reveal() == OTHER_SENTINEL
    assert to_secret(secret, name="api_key") is secret
    assert to_secret(None, name="api_key") is None


def test_to_secret_rejects_other_types_without_the_value() -> None:
    # when
    with pytest.raises(TypeError, match="api_key must be a secret or a str, got int"):
        to_secret(734921, name="api_key")


def test_a_secret_input_wraps_a_string_and_keeps_a_secret() -> None:
    # given
    secret = Secret(SENTINEL, name="SLACK_TOKEN")

    # when
    from_string = decode_payload(SENTINEL, kinds=[SECRET_KIND], location="token")
    from_secret = decode_payload(secret, kinds=[SECRET_KIND], location="token")

    # then
    assert isinstance(from_string, Secret)
    assert from_string.name == "input"
    assert from_string.reveal() == SENTINEL
    assert from_secret is secret


def test_secret_wins_over_string_in_a_union_input() -> None:
    # when
    payload = decode_payload(
        SENTINEL, kinds=[SECRET_KIND, STRING_KIND], location="token"
    )

    # then
    assert isinstance(payload, Secret)


def test_a_non_string_secret_input_fails_without_the_value() -> None:
    # when
    with pytest.raises(WorkflowInputError) as raised:
        decode_payload(734921, kinds=[SECRET_KIND], location="token")

    # then
    assert "a secret must be a str, got int" in str(raised.value)
    assert "734921" not in str(raised.value)


def test_kind_validates_and_serializes_only_secrets() -> None:
    # given
    secret = Secret(SENTINEL, name="SLACK_TOKEN")

    # when
    serialized = SECRET_KIND.to_serialized(secret)

    # then
    assert serialized == "<secret SLACK_TOKEN>"
    SECRET_KIND.check(secret)
    with pytest.raises(Exception):
        SECRET_KIND.check(SENTINEL)
    with pytest.raises(TypeError):
        SECRET_KIND.to_serialized(SENTINEL)


def test_reveal_secrets_replaces_nested_secrets_without_mutating_input() -> None:
    # given
    token = Secret(SENTINEL, name="SLACK_TOKEN")
    other = Secret(OTHER_SENTINEL, name="QUERY_KEY")
    params = {
        "headers": {"Authorization": token, "Accept": "json"},
        "query": [("key", other), ("page", 2)],
        "plain": "unchanged",
    }

    # when
    revealed = reveal_secrets(params)

    # then
    assert revealed.value == {
        "headers": {"Authorization": SENTINEL, "Accept": "json"},
        "query": [("key", OTHER_SENTINEL), ("page", 2)],
        "plain": "unchanged",
    }
    assert params["headers"]["Authorization"] is token
    assert params["query"][0][1] is other
    assert revealed.secrets == (token, other)
    assert repr(revealed) == "Revealed(secrets=['SLACK_TOKEN', 'QUERY_KEY'])"


def test_reveal_secrets_of_an_unavailable_secret_raises() -> None:
    # given
    params = {"headers": {"Authorization": Secret.unavailable("SLACK_TOKEN")}}

    # when / then
    with pytest.raises(SecretUnavailableError, match="SLACK_TOKEN"):
        reveal_secrets(params)


def test_scrub_replaces_every_exact_occurrence_longest_first() -> None:
    # given
    short = Secret("abc", name="SHORT")
    long = Secret("abcdef", name="LONG")
    empty = Secret("", name="EMPTY")
    revealed = Revealed(value=None, secrets=(short, long, empty))

    # when
    scrubbed = revealed.scrub("GET /?k=abcdef&j=abc failed: abc")

    # then
    assert scrubbed == ("GET /?k=<secret LONG>&j=<secret SHORT> failed: <secret SHORT>")


def test_scrub_does_not_find_encoded_forms() -> None:
    # given
    revealed = reveal_secrets(Secret("a b/c", name="TOKEN"))

    # when
    scrubbed = revealed.scrub("url=a%20b%2Fc raw=a b/c")

    # then
    assert scrubbed == "url=a%20b%2Fc raw=<secret TOKEN>"


def test_record_codec_writes_the_name_only_and_restores_an_unavailable_secret() -> None:
    # given
    secret = Secret(SENTINEL, name="SLACK_TOKEN")

    # when
    node = encode_secret_record(secret, None, None)
    restored = decode_secret_record(json.loads(json.dumps(node)), None, None)

    # then
    assert node == {"name": "SLACK_TOKEN"}
    assert SENTINEL not in json.dumps(node)
    assert restored.name == "SLACK_TOKEN"
    assert not restored.available
    with pytest.raises(SecretUnavailableError):
        restored.reveal()


@pytest.mark.parametrize(
    "node",
    [None, "SLACK_TOKEN", {}, {"name": ""}, {"name": 3}, {"name": "A", "value": "x"}],
)
def test_record_codec_rejects_malformed_nodes(node) -> None:
    # when / then
    with pytest.raises(ValueError, match="recorded secret"):
        decode_secret_record(node, None, None)
