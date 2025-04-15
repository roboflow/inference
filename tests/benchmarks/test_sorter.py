from inference.core.bubble_sort import sorter


def test_sort(benchmark):
    result = benchmark(sorter, list(reversed(range(500))))
    assert result == list(range(500))

# This should not be picked up as a benchmark test
def test_sort2():
    result = sorter(list(reversed(range(500))))
    assert result == list(range(500))
