import torch

from inference_models.models.common.torch import generate_batch_chunks


def test_generate_chunks_when_input_smaller_than_chunk_size() -> None:
    # given
    input_batch = torch.ones(size=(2, 3, 64, 96))
    input_batch[0] *= 3
    input_batch[1] *= 5

    # when
    result = list(
        generate_batch_chunks(
            input_batch=input_batch,
            chunk_size=3,
        )
    )

    # then
    assert len(result) == 1
    assert result[0][0].shape == (3, 3, 64, 96)
    assert torch.all(result[0][0][0] == 3)
    assert torch.all(result[0][0][1] == 5)
    assert torch.all(result[0][0][2] == 0)
    assert result[0][1] == 1


def test_generate_chunks_when_input_equal_to_chunk_size() -> None:
    # given
    input_batch = torch.ones(size=(2, 3, 64, 96))
    input_batch[0] *= 3
    input_batch[1] *= 5

    # when
    result = list(
        generate_batch_chunks(
            input_batch=input_batch,
            chunk_size=2,
        )
    )

    # then
    assert len(result) == 1
    assert result[0][0].shape == (2, 3, 64, 96)
    assert torch.all(result[0][0][0] == 3)
    assert torch.all(result[0][0][1] == 5)
    assert result[0][1] == 0


def test_generate_chunks_when_input_larger_than_chunk_size() -> None:
    # given
    input_batch = torch.ones(size=(2, 3, 64, 96))
    input_batch[0] *= 3
    input_batch[1] *= 5

    # when
    result = list(
        generate_batch_chunks(
            input_batch=input_batch,
            chunk_size=1,
        )
    )

    # then
    assert len(result) == 2
    assert result[0][0].shape == (1, 3, 64, 96)
    assert result[1][0].shape == (1, 3, 64, 96)
    assert torch.all(result[0][0][0] == 3)
    assert torch.all(result[1][0][0] == 5)
    assert result[0][1] == 0
    assert result[1][1] == 0


def test_generate_chunks_when_input_larger_than_chunk_size_and_chunk_size_more_than_one() -> (
    None
):
    # given
    input_batch = torch.ones(size=(5, 3, 64, 96))
    input_batch[0] *= 3
    input_batch[1] *= 5
    input_batch[2] *= 7
    input_batch[3] *= 11
    input_batch[4] *= 13

    # when
    result = list(
        generate_batch_chunks(
            input_batch=input_batch,
            chunk_size=2,
        )
    )

    # then
    assert len(result) == 3
    assert result[0][0].shape == (2, 3, 64, 96)
    assert result[1][0].shape == (2, 3, 64, 96)
    assert result[2][0].shape == (2, 3, 64, 96)
    assert torch.all(result[0][0][0] == 3)
    assert torch.all(result[0][0][1] == 5)
    assert torch.all(result[1][0][0] == 7)
    assert torch.all(result[1][0][1] == 11)
    assert torch.all(result[2][0][0] == 13)
    assert torch.all(result[2][0][1] == 0)
    assert result[0][1] == 0
    assert result[1][1] == 0
    assert result[2][1] == 1
