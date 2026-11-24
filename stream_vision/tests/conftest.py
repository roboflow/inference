"""Collection rules for the Streamvision test tree.

The Workflows V2 adapter suites run alone (``python -m pytest -c pytest-v2.ini``),
so the default suites never collect them.
"""

collect_ignore = ["v2"]
