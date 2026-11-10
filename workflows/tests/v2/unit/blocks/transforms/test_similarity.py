"""``roboflow_core/cosine_similarity@v1`` on analytic vectors."""

import pytest
import torch
from roboflow_workflows.v2.blocks.transforms.cosine_similarity.v1 import CosineSimilarityBlock


def _similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    result = CosineSimilarityBlock().run(embedding_1=first, embedding_2=second)

    return result["similarity"]


@pytest.mark.parametrize(
    "second, expected",
    [([2.0, 0.0], 1.0), ([0.0, 3.0], 0.0), ([-1.0, 0.0], -1.0), ([1.0, 1.0], 0.5**0.5)],
    ids=["same-direction", "orthogonal", "opposite", "45-degrees"],
)
def test_similarity_depends_only_on_direction(second: list, expected: float) -> None:
    similarity = _similarity(torch.tensor([1.0, 0.0]), torch.tensor(second))

    assert isinstance(similarity, float)
    assert similarity == pytest.approx(expected)


def test_mixed_dtypes_and_non_contiguous_views_are_compared_exactly() -> None:
    first = torch.tensor([[3.0, 9.0], [4.0, 9.0]], dtype=torch.float16)[:, 0]
    second = torch.tensor([3.0, 4.0], dtype=torch.float64)

    assert not first.is_contiguous()
    assert _similarity(first, second) == pytest.approx(1.0)


def test_result_stays_within_minus_one_and_one() -> None:
    vector = torch.full((513,), 0.1, dtype=torch.float32)

    similarity = _similarity(vector, vector.clone())

    assert -1.0 <= similarity <= 1.0
    assert similarity == pytest.approx(1.0)


def test_large_random_vectors_match_the_reference_formula() -> None:
    generator = torch.Generator().manual_seed(7)
    first, second = torch.randn((2, 768), generator=generator)
    expected = float(first @ second) / (float(first.norm()) * float(second.norm()))

    assert _similarity(first, second) == pytest.approx(expected, rel=1e-5)


def test_different_lengths_fail_with_both_lengths() -> None:
    with pytest.raises(ValueError, match="different lengths, 3 and 2"):
        _similarity(torch.ones(3), torch.ones(2))


@pytest.mark.parametrize("zero_first", [True, False])
def test_a_zero_vector_fails_instead_of_returning_nan(zero_first: bool) -> None:
    pair = [torch.zeros(3), torch.ones(3)]
    if not zero_first:
        pair.reverse()
    which = "embedding_1" if zero_first else "embedding_2"

    with pytest.raises(ValueError, match=f"{which} is a zero vector"):
        _similarity(*pair)


def test_different_devices_fail_without_a_hidden_copy() -> None:
    with pytest.raises(ValueError, match="different devices, cpu and meta"):
        _similarity(torch.ones(3), torch.ones(3, device="meta"))
