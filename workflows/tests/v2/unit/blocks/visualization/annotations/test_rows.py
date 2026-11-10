"""Host row packing preserves mixed dtypes and singleton-strided values."""

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.visualization.annotations.rows import _host_columns


@pytest.mark.parametrize("count", [0, 1, 3])
@pytest.mark.parametrize("strided", [False, True])
def test_host_columns_preserves_exact_mixed_values(count: int, strided: bool) -> None:
    expected = [
        np.array([[1.25, -2.5, 3.75, 4.125]] * count, dtype=np.float64).reshape(
            count, 4
        ),
        np.array([[2**60 + 3]] * count, dtype=np.int64).reshape(count, 1),
        np.array([[0.125]] * count, dtype=np.float32).reshape(count, 1),
        np.array([[-32767]] * count, dtype=np.int16).reshape(count, 1),
    ]
    columns = []
    for values in expected:
        column = torch.from_numpy(values.copy())
        if strided:
            storage = torch.zeros(38 * (column.numel() + 1), dtype=column.dtype)
            column = storage.as_strided(
                size=column.shape,
                stride=(38 * column.shape[1], 38),
                storage_offset=7,
            )
            column.copy_(torch.from_numpy(values))
        columns.append(column)

    if count == 1 and strided:
        # PyTorch regards a (1, 1) tensor as contiguous even with inner stride 38;
        # contiguous() alone leaves a dtype-changing view unable to read it.
        assert columns[1].is_contiguous()
        assert columns[1].contiguous().stride(-1) == 38

    actual = _host_columns(columns, count=count)

    assert len(actual) == len(expected)
    for output, values, column in zip(actual, expected, columns):
        assert output.shape == values.shape
        assert output.dtype == values.dtype
        np.testing.assert_array_equal(output, values)
        np.testing.assert_array_equal(column.numpy(), values)


def test_host_columns_preserves_contiguous_column_with_singleton_inner_stride() -> None:
    values = np.array([[2**60 + 3], [-(2**60) + 5], [17]], dtype=np.int64)
    column = torch.from_numpy(values.copy()).as_strided(size=(3, 1), stride=(1, 38))
    assert column.is_contiguous()
    assert column.stride(-1) == 38

    actual = _host_columns([column], count=3)

    np.testing.assert_array_equal(actual[0], values)
