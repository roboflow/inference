"""Run the existing manual hosted checks with encrypted cross-job evidence."""

from __future__ import annotations

import base64
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import click
import hosted_e2e as runner
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_FILES = {
    "collection.json",
    "plan.json",
    "readiness.json",
    "warmup.json",
    "dependencies.txt",
    "acceptance.json",
    "stage.log",
    *{f"shard-{shard}.json" for shard in "ABCD"},
}
_CREDENTIALS = (
    "HOSTED_PLATFORM_TESTS_API_KEY",
    "OPENAI_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_VISION_API_KEY",
)


def _binding() -> bytes:
    runner._repository_identity()
    runner._require(
        len(os.environ.get("HOSTED_PLATFORM_TESTS_API_KEY", "")) >= 20,
        "missing evidence credential",
    )
    value = {
        key: os.environ[key]
        for key in (
            "GITHUB_REPOSITORY",
            "GITHUB_SHA",
            "HOSTED_E2E_RUN_KEY",
            "HOSTED_E2E_TARGET",
            "HOSTED_E2E_TEST_SHA",
            "HOSTED_E2E_RUNNER_SHA",
        )
    }
    result = json.dumps(value, sort_keys=True).encode()
    return result


def _cipher(binding: bytes) -> AESGCM:
    # Domain-separated derivation: never use the API token directly as an AES key.
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=binding,
        info=b"inference-hosted-e2e-evidence-v1",
    ).derive(os.environ["HOSTED_PLATFORM_TESTS_API_KEY"].encode())
    cipher = AESGCM(key)
    return cipher


def _seal(output: Path, *, destination: Path) -> None:
    binding = _binding()
    files = {
        path.name: base64.b64encode(path.read_bytes()).decode()
        for path in output.iterdir()
        if path.is_file() and path.name in _FILES
    }
    nonce = os.urandom(12)
    ciphertext = _cipher(binding).encrypt(
        nonce, json.dumps(files, sort_keys=True).encode(), binding
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"HE2E1" + nonce + ciphertext)


def _restore(source: Path, *, output: Path) -> None:
    binding = _binding()
    bundles = sorted(source.glob("*.bin"))
    runner._require(bool(bundles), "missing encrypted evidence")
    # Validate every bundle before writing anything. No archive paths are trusted.
    restored = {}
    for bundle in bundles:
        data = bundle.read_bytes()
        runner._require(data.startswith(b"HE2E1"), "invalid encrypted evidence")
        files = json.loads(_cipher(binding).decrypt(data[5:17], data[17:], binding))
        runner._require(isinstance(files, dict), "invalid evidence")
        for name, value in files.items():
            runner._require(name in _FILES, "invalid evidence path")
            if name == "stage.log":
                continue
            content = base64.b64decode(value, validate=True)
            runner._require(
                name not in restored or restored[name] == content,
                "conflicting evidence",
            )
            restored[name] = content
    runner._require(not any(output.glob("*.json")), "evidence directory not empty")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, content in restored.items():
        path = output / name
        path.touch(mode=0o600)
        path.write_bytes(content)


def _execute(stage: str, *, suite: Path, policy: Path, output: Path, shard: str) -> int:
    runner._require(
        all(stat.S_ISREG(os.fstat(fd).st_mode) for fd in (1, 2)),
        "pytest output must be captured",
    )
    _binding()
    os.environ["HOSTED_E2E_MODE"] = "repository"
    if stage in {"prepare", "warmup", "shard"}:
        runner._require(
            all(os.environ.get(key, "").strip() for key in _CREDENTIALS),
            "missing provider credentials",
        )
    if stage == "prepare":
        profile = os.environ.get("HOSTED_E2E_TIMING_PROFILE", "").strip()
        if profile:
            runner._write(policy / "timing_profile.json", json.loads(profile))
        runner._prepare(suite=suite, policy=policy, output=output)
    elif stage == "warmup":
        runner._warmup(
            suite=suite,
            policy=policy,
            output=output,
            skip_warmup=os.environ.get("HOSTED_E2E_SKIP_WARMUP") == "true",
        )
    elif stage == "shard":
        code = runner._run_shard(suite=suite, policy=policy, output=output, shard=shard)
        return code
    else:
        result = runner._aggregate(
            runner._read(output / "plan.json"),
            [runner._read(path) for path in sorted(output.glob("shard-*.json"))],
            readiness=runner._read(output / "readiness.json"),
        )
        runner._write(output / "acceptance.json", result)
        return 0 if result["accepted"] else 1
    return 0


def _run(stage: str, *, suite: Path, policy: Path, output: Path, shard: str) -> int:
    _binding()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    code = 1
    try:
        # Separate interpreters are essential: conftest reads SKIP_WARMUP on import.
        for operation in (["prepare", "warmup"] if stage == "prepare" else [stage]):
            args = [
                sys.executable,
                str(Path(__file__).resolve()),
                operation,
                "--execute-hidden",
                "--suite",
                str(suite),
                "--policy",
                str(policy),
                "--output",
                str(output),
                "--shard",
                shard,
            ]
            with (output / "stage.log").open("ab") as log:
                result = subprocess.run(
                    args,
                    stdout=log,
                    stderr=log,
                    stdin=subprocess.DEVNULL,
                    timeout=1500,
                    check=False,
                )
            code = result.returncode
            if code:
                break
    finally:
        # Do not upload plaintext, even when collection, warmup or a shard fails.
        label = f"shard-{shard}" if stage == "shard" else stage
        _seal(output, destination=Path("encrypted") / f"{label}.bin")
    return code


def _failure_summaries(output: Path, *, stage: str, shard: str) -> dict:
    # Only redacted one-line summaries; full tracebacks stay in encrypted stage.log.
    path = output / (f"shard-{shard}.json" if stage == "shard" else "acceptance.json")
    if stage not in {"shard", "aggregate"} or not path.exists():
        return {}

    record = runner._read(path)
    if stage == "aggregate":
        summaries = record.get("failure_summaries", {})
    else:
        summaries = {
            nodeid: case["failures"]
            for nodeid, case in record.get("cases", {}).items()
            if case.get("failures")
        }
    return summaries


@click.command()
@click.argument(
    "stage", type=click.Choice(["prepare", "warmup", "shard", "aggregate", "restore"])
)
@click.option(
    "--suite",
    type=click.Path(path_type=Path),
    default="tests/inference/hosted_platform_tests",
)
@click.option(
    "--policy",
    type=click.Path(path_type=Path),
    default=".github/hosted-e2e",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="evidence",
)
@click.option(
    "--source",
    type=click.Path(path_type=Path),
    default="incoming",
)
@click.option(
    "--shard",
    type=click.Choice(list("ABCD")),
    default="A",
)
@click.option(
    "--execute-hidden",
    is_flag=True,
    hidden=True,
)
def main(stage, suite, policy, output, source, shard, execute_hidden):
    """Execute manual hosted stages without emitting raw test output.

    Args:
        stage: Stage to execute or encrypted evidence restoration.
        suite: Pytest suite directory.
        policy: Reviewed skip and optional timing policy directory.
        output: Local plaintext evidence directory, never uploaded.
        source: Directory containing encrypted input bundles.
        shard: Shard A, B, C or D.
        execute_hidden: Internal subprocess mode requiring captured output.
    """
    os.umask(0o077)
    try:
        if execute_hidden:
            code = _execute(
                stage, suite=suite, policy=policy, output=output, shard=shard
            )
        elif stage == "restore":
            _restore(source, output=output)
            code = 0
        else:
            code = _run(stage, suite=suite, policy=policy, output=output, shard=shard)
        if not execute_hidden:
            click.echo(f"Hosted E2E {stage}: {'passed' if code == 0 else 'failed'}")
            if code != 0:
                summaries = _failure_summaries(output, stage=stage, shard=shard)
                for nodeid, phases in sorted(summaries.items()):
                    for phase, summary in sorted(phases.items()):
                        click.echo(f"  {nodeid} [{phase}]: {summary}")
            if stage == "aggregate" and code == 0:
                plan = runner._read(output / "plan.json")
                result = runner._read(output / "acceptance.json")
                click.echo(
                    f"Coverage verified: {len(plan['manifest'])} cases "
                    "across 4 shards; "
                    f"{len(result['retried'])} retried cases. "
                    "Deployment identity not verified."
                )
        raise SystemExit(code)
    except (Exception, KeyboardInterrupt):
        click.echo("Hosted E2E stage failed; detailed output is not public.", err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
