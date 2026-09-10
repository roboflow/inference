"""Cross-check the isolation probe's blocked list against the static baseline.

The probe blocks at the SHALLOWEST non-allowed ancestor, so
`inference.core.entities` legitimately has no exact baseline row - its leaves
do. The comparison is therefore "exact name OR dotted descendant", with the
`(exec'd string)` suffix normalised away.

Fails loudly if it cannot find a `blocked_import_attempts` block, or finds one
that parses to zero names: a silently empty extraction would make the whole
check vacuous.
"""

import argparse
import pathlib
import re
import sys

BLOCK = re.compile(r'"blocked_import_attempts":\s*\[(.*?)\]', flags=re.S)
NAME = re.compile(r'"([^"]+)"')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_output")
    parser.add_argument("baseline")
    parser.add_argument("--expected-blocks", type=int, default=2,
                        help="one blocked_import_attempts block per tensor mode")
    args = parser.parse_args()

    text = pathlib.Path(args.probe_output).read_text(encoding="utf-8")
    blocks = BLOCK.findall(text)
    if len(blocks) != args.expected_blocks:
        print(
            f"FAIL: found {len(blocks)} blocked_import_attempts blocks, "
            f"expected {args.expected_blocks} - the probe output shape changed",
            file=sys.stderr,
        )
        return 2
    blocked = set()
    for raw in blocks:
        names = NAME.findall(raw)
        if not names:
            print("FAIL: a blocked_import_attempts block parsed to zero names",
                  file=sys.stderr)
            return 2
        blocked.update(names)
    blocked = sorted(blocked)

    modules = set()
    for row in pathlib.Path(args.baseline).read_text(encoding="utf-8").splitlines():
        if not row or row.startswith("#") or "\t" not in row:
            continue
        modules.add(row.split("\t", 1)[1].strip().split(" ")[0])

    def covered(name: str) -> bool:
        return any(m == name or m.startswith(name + ".") for m in modules)

    uncovered = [name for name in blocked if not covered(name)]
    print("blocked names:", blocked)
    print("uncovered by the static baseline:", uncovered)
    if uncovered:
        print("FAIL: the probe blocked modules the lint cannot see", file=sys.stderr)
        return 1
    print("OK: every blocked module is covered by a baseline row or its descendants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
