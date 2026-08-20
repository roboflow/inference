#!/usr/bin/env python3
"""Materialize standalone processor job files from the benchmark corpus."""

import argparse
import json
import re
import shlex
from pathlib import Path

SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")


def load_corpus(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open() as source:
        manifest = json.load(source)
    profiles = {}
    for profile in manifest.get("profiles") or []:
        profile_id = profile.get("id")
        if not SAFE_ID.fullmatch(str(profile_id or "")) or profile_id in profiles:
            raise ValueError(f"invalid or duplicate profile id: {profile_id!r}")
        spec_path = manifest_path.parent / profile["spec"]
        with spec_path.open() as source:
            specification = json.load(source)
        profiles[profile_id] = {**profile, "specification": specification}
    return profiles


def build_jobs(profiles, selections, source_url_template, mode, repeat):
    jobs = []
    ordinal = 0
    for profile_id in selections:
        if profile_id not in profiles:
            raise ValueError(f"unknown workflow profile: {profile_id}")
        profile = profiles[profile_id]
        for copy_index in range(repeat):
            ordinal += 1
            job_id = f"bench-{profile_id}-{copy_index + 1}-{ordinal}"
            source_url = source_url_template.format(
                index=ordinal,
                profile=profile_id,
                stream=f"bench-source-{ordinal:04d}",
            )
            jobs.append(
                {
                    "id": job_id,
                    "sourceUrl": source_url,
                    "mode": mode,
                    "imageOutput": profile.get("imageOutput"),
                    "workflowSpecification": profile["specification"],
                    "benchmarkProfile": profile_id,
                }
            )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=Path(__file__).with_name("workflows") / "manifest.json",
    )
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--mode", choices=["stream", "batch"], default="stream")
    parser.add_argument(
        "--source-url-template",
        default="rtsp://127.0.0.1:8554/{stream}",
        help="supports {stream}, {profile}, and {index}",
    )
    parser.add_argument("--output-dir", default="generated-jobs")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    profiles = load_corpus(args.manifest)
    try:
        jobs = build_jobs(
            profiles,
            args.profile,
            args.source_url_template,
            args.mode,
            args.repeat,
        )
    except (KeyError, ValueError) as error:
        parser.error(str(error))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for job in jobs:
        path = output_dir / f"{job['id']}.json"
        with path.open("w") as output:
            json.dump(job, output, indent=2, sort_keys=True)
            output.write("\n")
        paths.append(path)

    command = [
        "python",
        "development/video_poc/processor/processor.py",
        "--max-jobs",
        str(len(paths)),
    ]
    for path in paths:
        command.extend(["--job-file", str(path)])
    print(shlex.join(command))


if __name__ == "__main__":
    main()
