import pytest
from roboflow_workflows.core_steps.fusion.buffer.v1 import BufferBlockV1


def test_buffer() -> None:
    buffer_block = BufferBlockV1()

    # first result
    first = buffer_block.run(data=1, length=2, pad=False)
    assert first == {
        "output": [1],
    }

    # add more data
    second = buffer_block.run(data=2, length=2, pad=False)
    assert second == {
        "output": [2, 1],
    }

    # rollover
    third = buffer_block.run(data=3, length=2, pad=False)
    assert third == {
        "output": [3, 2],
    }


def test_with_padding() -> None:
    buffer_block = BufferBlockV1()

    # first result
    first = buffer_block.run(data=1, length=2, pad=True)
    assert first == {
        "output": [1, None],
    }


@pytest.mark.parametrize("pad", [False, True])
def test_buffer_results_preserve_each_frame_history(pad: bool) -> None:
    buffer_block = BufferBlockV1()
    results = [
        buffer_block.run(data=frame_number, length=8, pad=pad)
        for frame_number in range(1, 13)
    ]

    for frame_number, result in enumerate(results, start=1):
        expected = list(range(frame_number, max(0, frame_number - 8), -1))
        if pad:
            expected += [None] * (8 - len(expected))
        assert result["output"] == expected

    results[-1]["output"].clear()
    assert buffer_block.run(data=13, length=8, pad=pad)["output"] == list(
        range(13, 5, -1)
    )
