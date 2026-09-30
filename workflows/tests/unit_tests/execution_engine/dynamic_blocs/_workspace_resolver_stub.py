"""StubResolver helper for dynamic-block scaffolding tests.

Extracted so `test_block_scaffolding.py` can run standalone without pulling in
the server-parity `test_workspace_resolver.py` (kept in the root suite because
those tests assert against `inference.core.*` module presence).
"""


class StubResolver:
    def __init__(self, workspace=None):
        self._workspace = workspace
        self.calls = []

    def resolve_workspace(self, api_key):
        self.calls.append(api_key)
        return self._workspace
