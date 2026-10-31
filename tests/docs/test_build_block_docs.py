"""Tests for the documentation build pipeline's Jinja2 escaping logic.

These tests ensure that generated Markdown docs do not contain raw
``{{ $parameters.xxx }}`` expressions that would cause mkdocs-macros
(Jinja2) parse errors like "unexpected char '$'".
"""

from jinja2 import Environment

from development.docs.build_block_docs import (
    _escape_jinja2_expressions,
    slugify_block_name,
)

jinja_env = Environment()


def _parses_as_jinja2(text: str) -> bool:
    """Return True if *text* can be parsed by Jinja2 without errors."""
    try:
        jinja_env.parse(text)
        return True
    except Exception:
        return False


class TestEscapeJinja2Expressions:
    """Tests for ``_escape_jinja2_expressions``."""

    def test_bare_dollar_parameters_in_braces(self):
        """The main bug: ``{{ $parameters.xxx }}`` must be escaped."""
        raw = 'replaces placeholders like `{{ $parameters.parameter_name }}`'
        result = _escape_jinja2_expressions(raw)
        assert _parses_as_jinja2(result)
        # The rendered output should still contain the human-readable text
        assert "$parameters.parameter_name" in result

    def test_multiple_dollar_expressions(self):
        raw = (
            "Detected {{ $parameters.num_objects }} objects. "
            "Classes: {{ $parameters.classes }}."
        )
        result = _escape_jinja2_expressions(raw)
        assert _parses_as_jinja2(result)
        assert "$parameters.num_objects" in result
        assert "$parameters.classes" in result

    def test_already_escaped_expression_not_double_escaped(self):
        """Expressions that are already escaped must not be broken."""
        already_escaped = "{{ '{{' }} $parameters.predicted_classes {{ '}}' }}"
        result = _escape_jinja2_expressions(already_escaped)
        assert _parses_as_jinja2(result)

    def test_normal_jinja2_variables_untouched(self):
        """Legitimate Jinja2 variables (no $) must not be modified."""
        normal = "Version: {{ VERSION }}"
        result = _escape_jinja2_expressions(normal)
        assert result == normal

    def test_no_braces_dollar_untouched(self):
        """Bare $inputs.xxx outside {{ }} must not be modified."""
        bare = '"smtp_server": "$inputs.smtp_server"'
        result = _escape_jinja2_expressions(bare)
        assert result == bare
        assert _parses_as_jinja2(result)

    def test_dollar_parameters_with_extra_spaces(self):
        raw = "{{  $parameters.foo  }}"
        result = _escape_jinja2_expressions(raw)
        assert _parses_as_jinja2(result)

    def test_css_double_braces_untouched(self):
        """CSS rules using ``{{ }}`` for escaping Python .format() must survive."""
        css = "article > a.md-content__button.md-icon:first-child {{ display: none; }}"
        result = _escape_jinja2_expressions(css)
        # CSS doesn't contain $ so should be untouched
        assert result == css

    def test_real_world_email_notification_description(self):
        """Reproduce the exact text that caused the original bug report."""
        text = (
            "3. Formats the email message by processing dynamic parameters "
            "(replaces placeholders like `{{ $parameters.parameter_name }}` "
            "with actual workflow data from `message_parameters`)"
        )
        assert not _parses_as_jinja2(text)  # Confirm it's broken before fix
        result = _escape_jinja2_expressions(text)
        assert _parses_as_jinja2(result)

    def test_real_world_field_description(self):
        """Field description with multiple {{ $parameters }} references."""
        text = (
            "SMS message content (plain text). Supports dynamic parameters "
            "using placeholder syntax: {{ $parameters.parameter_name }}. "
            "Example: 'Detected {{ $parameters.num_objects }} objects. "
            "Alert: {{ $parameters.classes }}.'"
        )
        assert not _parses_as_jinja2(text)
        result = _escape_jinja2_expressions(text)
        assert _parses_as_jinja2(result)
class TestSlugifyBlockName:
    """Tests for ``slugify_block_name`` (issue #1909).

    A block page's filename comes from its human-readable name, and every block
    page cross-references every other block, so a name that slugifies into an
    unusable filename breaks links across the whole site rather than on one page.
    """

    def test_parentheses_do_not_leak_into_the_slug(self):
        """The reported bug.

        ``PTZ Tracking (ONVIF)`` produced ``ptz_tracking(onvif)``, so the
        generated link read ``[PTZ Tracking (ONVIF)](ptz_tracking(onvif).md)``.
        A markdown link destination ends at the first unbalanced ``)``, so
        mkdocs cut it short and resolved a page that does not exist.
        """
        assert slugify_block_name("PTZ Tracking (ONVIF)") == "ptz_tracking_onvif"

    def test_square_brackets_do_not_leak_into_the_slug(self):
        """Renaming the block to use square brackets would not have fixed it.

        ``[`` opens a markdown link label, so ``ptz_tracking[onvif]`` is no more
        usable as a destination than the parenthesised form.
        """
        assert slugify_block_name("PTZ Tracking [ONVIF]") == "ptz_tracking_onvif"

    def test_dots_are_preserved(self):
        """Guards against over-sanitising.

        Dots are legal in a URL and several published pages depend on them, so
        widening the substitution to every non-alphanumeric character would move
        working page URLs while fixing the broken one.
        """
        assert slugify_block_name("Qwen2.5-VL") == "qwen2.5_vl"
        assert slugify_block_name("Llama 3.2 Vision") == "llama3.2_vision"

    def test_ordinary_names_are_unchanged(self):
        assert slugify_block_name("Object Detection Model") == "object_detection_model"
        assert slugify_block_name("Barcode Detection") == "barcode_detection"

    def test_no_leading_or_trailing_underscores(self):
        """A name that is entirely bracketed must not slugify to ``_foo_``."""
        assert slugify_block_name("(Beta)") == "beta"

    def test_repeated_brackets_collapse(self):
        assert slugify_block_name("Thing ((Nested))") == "thing_nested"
