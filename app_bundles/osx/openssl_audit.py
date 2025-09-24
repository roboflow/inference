#!/usr/bin/env python3
"""
openssl_audit.py  –  Inspect OpenSSL dylibs inside a PyInstaller bundle.

Usage:
    python openssl_audit.py dist/inference-app
"""

import hashlib, os, re, subprocess, sys
from pathlib import Path
from textwrap import indent

# ---------------------------------------------------------------------------

def sha256(path: Path, block=1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

def dylib_version(path: Path) -> str:
    """Return first 'OpenSSL …' line (or '??')."""
    try:
        out = subprocess.check_output(["strings", "-a", str(path)],
                                      text=True, errors="ignore")
    except subprocess.CalledProcessError:
        return "??"
    for line in out.splitlines():
        if line.startswith("OpenSSL"):
            return line.replace("OpenSSL", "").strip() or "??"
    return "??"

def otool_l(path: Path) -> list[str]:
    return subprocess.check_output(["otool", "-L", str(path)],
                                   text=True).splitlines()[1:]

# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python openssl_audit.py <bundle-path>")
        sys.exit(1)

    bundle = Path(sys.argv[1]).resolve()
    if not bundle.is_dir():
        sys.exit(f"Error: {bundle} is not a directory")

    openssl_re = re.compile(r"lib(?:crypto|ssl)\.3\.dylib")

    # 1 — list every libcrypto/libssl
    dylibs = sorted(p for p in bundle.rglob("lib*.dylib")
                    if openssl_re.search(p.name))
    print("\n== 🗄  All OpenSSL dylibs ==")
    if not dylibs:
        print("  (none found)")
    for p in dylibs:
        print(f"  {p.relative_to(bundle)}  "
              f"[{sha256(p)}]  OpenSSL {dylib_version(p)}")

    # 2 — extension modules that link to them
    print("\n== 🔗  Extension modules that link to OpenSSL ==")
    ext_dir = bundle / "_internal" / "lib-dynload"
    if ext_dir.is_dir():
        linked = False
        for so in sorted(ext_dir.glob("*.so")):
            deps = otool_l(so)
            if any("libcrypto" in d or "libssl" in d for d in deps):
                linked = True
                print(f"  {so.relative_to(bundle)}")
                print(indent("\n".join(deps), "    "))
        if not linked:
            print("  (no extension modules link to OpenSSL)")
    else:
        print("  (no _internal/lib-dynload directory found)")

    # 3 — dyld trace of what actually loads
    print("\n== 🚀  What actually loads at runtime ==")
    exe = bundle / bundle.name
    if not exe.exists():
        try:
            exe = next(p for p in bundle.iterdir()
                       if p.is_file() and os.access(p, os.X_OK))
        except StopIteration:
            print("  (cannot find runnable binary in bundle)")
            return

    env = os.environ.copy()
    env["DYLD_PRINT_LIBRARIES"] = "1"

    try:
        run = subprocess.run([str(exe), "--help"],
                             env=env, capture_output=True,
                             text=True, timeout=15)
        stderr = run.stderr
    except subprocess.TimeoutExpired as e:
        # e.stderr is bytes → decode; e.stderr may be None
        stderr = (e.stderr or b"").decode(errors="ignore") \
                 + "\n(executable did not exit within 15 s)"
    except Exception as exc:
        print(f"  (failed to run executable: {exc})")
        return

    trace = [l for l in stderr.splitlines() if openssl_re.search(l)]
    if trace:
        print(indent("\n".join(trace[:60]), "  "))
        if len(trace) > 60:
            print("  … (truncated)")
    else:
        print("  (no OpenSSL libraries appeared in DYLD trace)")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
