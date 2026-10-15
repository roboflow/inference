# Hosted platform E2E sharding implementation plan

## Problem and behavior

The hosted platform suite currently runs serially, with session warmups and
credentialed test output in public-repository Actions runs. The target is four
independent shards, each running serial pytest, with complete coverage accounting.

Implement a private-caller reusable workflow: validate the caller and trusted
test revision, collect the full manifest, verify deployment identity, warm up
once, run A/B/C/D, and aggregate even after a shard fails. Use private timing
profiles from the latest release's matching suite, separately for each target.
Keep tests from the same file together to avoid splitting known shared writes.
Assign longer files first to the least-loaded shard. Freeze the manifest,
assignment, dependency versions and identity for a run. Unexpected skips,
collection drift, missing reports and failed prerequisites prevent acceptance.

The public repository contains generic orchestration only. Credentials come from
the private caller; detailed evidence stays in its private run. Only a fixed
acceptance status may be exported. Deployment and release remain external.

## Compatibility and rollout

This is opt-in infrastructure. Existing manual workflows remain unchanged until
the private caller is configured and the migration is approved; they must not
be used for the new parallel run. Four shards are the target. Live use requires
explicit load validation and a private deployment-verification adapter.

## Validation

Use synthetic suites with no credentials or network access to test deterministic
disjoint coverage, one warmup, retries, skips, fixture failures, collection drift,
missing/duplicate reports, identity mismatch and sanitized public status. Check
workflow syntax, caller guards and unconditional aggregation. No hosted run is
authorized by these checks and no speedup is claimed.

## Decisions and integration prerequisites

- The private caller repository has not been selected. Its protected manual
  workflow must pin this reusable workflow and provide the verification adapter.
- Deployment identity cannot be inferred from the test checkout. The private
  adapter must verify serving identity and readiness against a trusted source,
  and the deployment must remain stable during acceptance.
- Shared resources and provider limits need live load validation. File grouping
  and a serialized orchestration run do not prove independent external quotas.
- The shared-CI-infrastructure approach has been agreed within the team, as
  confirmed by a repository maintainer. Private-caller configuration, deployment
  verification and load/isolation validation remain prerequisites for live rollout.
