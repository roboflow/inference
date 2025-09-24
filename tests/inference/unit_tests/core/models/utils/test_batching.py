import numpy as np

from inference.core.models.utils.batching import create_batches


def test_create_batches_when_empty_sequence_given() -> None:
    # when
    result = list(create_batches(sequence=[], batch_size=4))

    # then
    assert result == []


def test_create_batches_when_not_allowed_batch_size_given() -> None:
    # when
    result = list(create_batches(sequence=[1, 2, 3], batch_size=0))

    # then
    assert result == [[1], [2], [3]]


def test_create_batches_when_batch_size_larger_than_sequence() -> None:
    # when
    result = list(create_batches(sequence=[1, 2], batch_size=4))

    # then
    assert result == [[1, 2]]


def test_create_batches_when_batch_size_multiplier_fits_sequence_length() -> None:
    # when
    result = list(create_batches(sequence=[1, 2, 3, 4], batch_size=2))

    # then
    assert result == [[1, 2], [3, 4]]


def test_create_batches_when_batch_size_multiplier_does_not_fir_sequence_length() -> (
    None
):
    # when
    result = list(create_batches(sequence=[1, 2, 3, 4], batch_size=3))

    # then
    assert result == [[1, 2, 3], [4]]
