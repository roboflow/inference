# Activate hosted E2E sharding in inference

## Problem and requested behavior

The staging and production manual workflows still execute one serial pytest
job. The merged private reusable workflow has no configured caller. A passing
unit test or a merge does not demonstrate that users can run four shards.

The repository maintainer selected the existing inference repository for this
activation. Keep the current manual workflow names and target selection. Wire
them to preparation, a shared warmup, parallel serial-pytest shards A–D, and
aggregation. Keep whole files together and preserve retry and skip accounting.

## Execution and evidence

Restrict credentialed jobs to manual execution of this repository's default
branch or an approved release fast-track head. Pin the checkout to the dispatch commit across every job. Serialize
hosted runs across both targets and retain the existing skip-warmup option.
Allow only the five cases explicitly disabled in the reviewed test source;
missing provider credentials must fail rather than silently expanding skips.

Capture pytest output locally. Transfer only authenticated encrypted evidence,
including frozen dependency versions, between jobs. Derive a run-and-attempt
specific key from the existing high-entropy hosted API credential using HKDF;
use AES-GCM, random nonces, and context authentication. This avoids adding a
private repository or another required credential. Retain ciphertext for three
days. Publish only fixed status messages and aggregate counts. Do not print
raw exception messages or emit plaintext artifacts.

These are the existing hosted functional checks, not proof of a particular
deployed image. Record that distinction in evidence. Preserve the separate
private workflow's stronger deployment verification contract.

## Validation and completion

Prove the old entrypoints fail the wiring regression. Exercise the actual stage
CLI in subprocesses, including encrypted transfer between isolated directories,
four shards, retries, deliberate skips, missing evidence, tampering, wrong
attempt/key, provider misconfiguration, and output containing synthetic secrets.
Run existing accounting regressions, workflow checks, and formatting locally.

Live completion requires reviewed code on main or an approved fast-track, an actual staging
dispatch, all four shard jobs, and successful aggregation with complete coverage.
Record the run link and results; keep rollout incomplete until that evidence
exists. Measure load and balance during staging before using production. Never
automatically deploy or publish as part of this workflow.
