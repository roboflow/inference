from development.profiling.data import build_data_source


def test_dummy_data_source_is_deterministic():
    data_source = build_data_source(
        name="dummy",
        config={"record_count": 2, "tensor_shape": [2, 2]},
    )

    records = list(data_source.iter_records())

    assert [record.id for record in records] == ["dummy-0", "dummy-1"]
    assert records[0].metadata["tensor"].tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert records[1].metadata["tensor"].tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_image_data_source_yields_sorted_local_paths_without_decode(tmp_path):
    first = tmp_path / "b.jpg"
    second = tmp_path / "a.png"
    first.write_bytes(b"not decoded")
    second.write_bytes(b"not decoded")

    data_source = build_data_source(
        name="images",
        config={"directory": str(tmp_path), "decode": False},
    )

    records = list(data_source.iter_records())

    assert [record.id for record in records] == ["a", "b"]
    assert [record.path for record in records] == [second, first]
    assert all(record.image is None for record in records)
