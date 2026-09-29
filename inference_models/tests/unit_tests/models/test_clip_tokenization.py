import clip
import pytest

from inference_models.errors import ModelInputError
from inference_models.models.clip.preprocessing import tokenize_texts


def test_tokenize_texts_when_texts_fit_context_length() -> None:
    # when
    result = tokenize_texts(["a photo of a cat", "a dog"], clip.tokenize)

    # then
    assert tuple(result.shape) == (2, 77)


def test_tokenize_texts_when_text_exceeds_context_length() -> None:
    # when
    with pytest.raises(ModelInputError) as error:
        tokenize_texts(["a photo of a cat", "cat " * 100], clip.tokenize)

    # then
    assert "too long for the model context length" in str(error.value)
    assert "cat cat" not in str(error.value)


def test_tokenize_texts_when_tokenizer_fails_for_other_reason() -> None:
    # given
    def failing_tokenizer(texts):
        raise RuntimeError("CUDA error: out of memory")

    # when
    with pytest.raises(RuntimeError) as error:
        tokenize_texts(["a photo of a cat"], failing_tokenizer)

    # then
    assert not isinstance(error.value, ModelInputError)
    assert "out of memory" in str(error.value)
