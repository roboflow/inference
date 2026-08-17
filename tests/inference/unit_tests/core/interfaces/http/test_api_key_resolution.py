from starlette.datastructures import Headers

from inference.core.interfaces.http import api_key_resolution
from inference.core.interfaces.http.api_key_resolution import (
    api_key_fallback,
    api_key_override,
    extract_api_key_from_headers,
    header_api_key,
)


class TestExtractApiKeyFromHeaders:
    def test_returns_none_when_no_authorization_header(self) -> None:
        # when
        result = extract_api_key_from_headers(Headers({}))

        # then
        assert result is None

    def test_extracts_bearer_token(self) -> None:
        # when
        result = extract_api_key_from_headers(
            Headers({"Authorization": "Bearer my-api-key"})
        )

        # then
        assert result == "my-api-key"

    def test_scheme_is_case_insensitive(self) -> None:
        # when
        result = extract_api_key_from_headers(
            Headers({"Authorization": "bearer my-api-key"})
        )

        # then
        assert result == "my-api-key"

    def test_header_name_lookup_is_case_insensitive(self) -> None:
        # given - starlette Headers normalise casing; a plain dict with
        # lowercase key must also work (middleware tests use raw dicts)
        starlette_result = extract_api_key_from_headers(
            Headers({"authorization": "Bearer my-api-key"})
        )
        plain_dict_result = extract_api_key_from_headers(
            {"authorization": "Bearer my-api-key"}
        )

        # then
        assert starlette_result == "my-api-key"
        assert plain_dict_result == "my-api-key"

    def test_ignores_non_bearer_scheme(self) -> None:
        # given - only the auth scheme matters to the extractor, so the value
        # is a plainly fake placeholder, not real base64 credentials
        result = extract_api_key_from_headers(
            Headers({"Authorization": "Basic fake-basic-credentials"})
        )

        # then
        assert result is None

    def test_ignores_value_without_scheme(self) -> None:
        # given - a bare token has no space, so the scheme split yields no token
        result = extract_api_key_from_headers(Headers({"Authorization": "my-api-key"}))

        # then
        assert result is None

    def test_ignores_bearer_with_empty_token(self) -> None:
        # when
        result = extract_api_key_from_headers(Headers({"Authorization": "Bearer   "}))

        # then
        assert result is None

    def test_strips_token_whitespace(self) -> None:
        # when
        result = extract_api_key_from_headers(
            Headers({"Authorization": "Bearer  my-api-key "})
        )

        # then
        assert result == "my-api-key"

    def test_disabled_by_flag(self, monkeypatch) -> None:
        # given
        monkeypatch.setattr(api_key_resolution, "ALLOW_API_KEY_FROM_HEADERS", False)

        # when
        result = extract_api_key_from_headers(
            Headers({"Authorization": "Bearer my-api-key"})
        )

        # then
        assert result is None


class TestApiKeyFallback:
    def test_explicit_value_wins_over_header_value(self) -> None:
        # given
        token = header_api_key.set("header-key")

        try:
            # when
            result = api_key_fallback("explicit-key")
        finally:
            header_api_key.reset(token)

        # then
        assert result == "explicit-key"

    def test_header_value_used_when_no_explicit_value(self) -> None:
        # given
        token = header_api_key.set("header-key")

        try:
            # when
            result = api_key_fallback(None)
        finally:
            header_api_key.reset(token)

        # then
        assert result == "header-key"

    def test_none_when_no_channel_carries_a_key(self) -> None:
        # when
        result = api_key_fallback(None)

        # then
        assert result is None

    def test_context_var_reset_restores_default(self) -> None:
        # given
        token = header_api_key.set("header-key")
        header_api_key.reset(token)

        # when
        result = api_key_fallback(None)

        # then
        assert result is None


class TestApiKeyOverride:
    def test_header_value_wins_over_body_value(self) -> None:
        # given
        token = header_api_key.set("header-key")

        try:
            # when
            result = api_key_override("body-key")
        finally:
            header_api_key.reset(token)

        # then
        assert result == "header-key"

    def test_body_value_used_when_no_header(self) -> None:
        # when
        result = api_key_override("body-key")

        # then
        assert result == "body-key"

    def test_none_when_no_channel_carries_a_key(self) -> None:
        # when
        result = api_key_override(None)

        # then
        assert result is None
