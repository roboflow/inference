from inference.core.workflows.core_steps.fusion.buffer.v1 import BufferBlockV1


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
