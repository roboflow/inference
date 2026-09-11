import ast
import sys
import tokenize

path = sys.argv[1]
# Preserve the file's own newline convention - llama_vision/v1.py is CRLF.
with tokenize.open(path) as fh:
    src = fh.read()
    newline = fh.newlines if isinstance(fh.newlines, str) else "\n"

tree = ast.parse(src)
lines = src.splitlines()


def _is_docstring(node) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(getattr(node, "value", None), ast.Constant)
        and isinstance(node.value.value, str)
    )


drop = {
    n.lineno - 1
    for n in tree.body
    if isinstance(n, ast.ImportFrom)
    and n.module in ("inference.core", "inference.core.logger")
    and any(a.name == "logger" for a in n.names)
}
assert drop, f"no logger import found in {path}"

# End of the LEADING import block, i.e. the last top-level import that still
# precedes the first top-level statement which is not an import or the module
# docstring. Using the last import in the whole module (as originally scoped)
# breaks `dynamic_blocks/modal_executor.py`, which calls `logger.info(...)` at
# module level between two import groups: the assignment would land after that
# call and raise NameError at import time.
first_code = min(
    (
        n.lineno
        for n in tree.body
        if not isinstance(n, (ast.Import, ast.ImportFrom)) and not _is_docstring(n)
    ),
    default=len(lines) + 1,
)
imports = [
    n
    for n in tree.body
    if isinstance(n, (ast.Import, ast.ImportFrom)) and n.lineno < first_code
]
assert imports, f"no leading import block in {path}"
insert_after = max(n.end_lineno for n in imports)

# Some modules already import `logging` at top level (vlm_as_detector/v2_tensor.py).
# A second `import logging` is an F811 redefinition, so skip the insertion there.
already_imports_logging = any(
    isinstance(n, ast.Import)
    and any(a.name == "logging" and a.asname is None for a in n.names)
    for n in tree.body
)

# `import logging` goes after the module docstring and any __future__ import.
head = 0
first = tree.body[0] if tree.body else None
if _is_docstring(first):
    head = first.end_lineno
for n in tree.body:
    if isinstance(n, ast.ImportFrom) and n.module == "__future__":
        head = max(head, n.end_lineno)

out = []
for i, line in enumerate(lines):
    if i not in drop:
        out.append(line)
    # The insertion checks run even for dropped lines. When the logger import
    # IS the last import - as in anthropic_claude/model_capabilities.py - the
    # insertion point is a dropped line, and a `continue` here skips the
    # insertion entirely, leaving the file compiling but with no `logger`.
    if i + 1 == head and not already_imports_logging:
        out.append("import logging")
    if i + 1 == insert_after:
        out.append("")
        out.append("logger = logging.getLogger(__name__)")
if head == 0 and not already_imports_logging:
    out.insert(0, "import logging")

with open(path, "w", encoding="utf-8", newline="") as fh:
    fh.write(newline.join(out) + newline)
