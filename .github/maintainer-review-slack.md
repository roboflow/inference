# Maintainer review handoff to Slack

The existing Claude reviewer remains the first gate. A separate, non-agent job
collects its exact sign-off into a revision-specific JSON artifact. A default-branch
`workflow_run` listener validates the run, artifact, bot comment, and current PR
revision, then asks maintainers to review in Slack. A human PR comment beginning
`/maintainer-review <reason>` uses the same notifier through `issue_comment`.

No notification is sent for findings, skipped reviews, failed/cancelled runs,
drafts, closed PRs, stale revisions, or an unauthorized escalation. Automatic
handoffs respect `skip-claude-review`; explicit human escalations still work.
An old pass is superseded as soon as a newer agent review starts.

## One-time setup

1. Create a Slack app with bot scope **`chat:write`** only. Install it in the
   Roboflow workspace and invite the bot to the desired release channel.
   No Slack event subscriptions, incoming endpoint, or hosting is needed.
2. In GitHub repository **Settings → Environments**, create
   **`maintainer-review-slack`**. Under deployment branches and tags, select
   **Selected branches and tags**, then allow the **branch `main` only**.
   Do not allow PR refs, wildcard branches, or tags. This rule prevents a PR's
   workflow changes from requesting the Slack credential.
3. Add the environment secret **`SLACK_REVIEW_BOT_TOKEN`** (the installed app's
   bot token). Do **not** put it in repository/organization secrets or pass it to
   the Claude workflow. The agent job never references this environment.
4. Add these environment variables:

   | Variable | Value |
   | --- | --- |
   | `SLACK_REVIEW_CHANNEL_ID` | The channel's `C…` ID, not its name |
   | `SLACK_REVIEW_MAINTAINER_IDS` | Comma-separated Slack user IDs, e.g. `U123,U456` |

   Only these configured maintainers are mentioned. Select the reviewer rotation
   or individuals the team wants notified. The bot does not assign GitHub reviewers.
5. After merging the code and completing setup, add the **repository variable**
   `MAINTAINER_REVIEW_SLACK_ENABLED=true`. It must be repository-level because
   the job condition is evaluated before environment variables are available.
   Remove it or set it to `false` to pause all notifications.

The workflow currently runs only in `roboflow/inference`. The channel is deliberately
configurable: verify the intended destination rather than inferring it from a name.
Maintainers must already have access to that channel; a mention does not grant access.

## Behavior and credential boundary

- `Claude PR Review` runs the existing agent. A fresh collector job reads GitHub
  comments and writes `maintainer-review-result-<attempt>/result.json`, containing
  repository, PR, reviewed SHA, run ID/attempt, verdict, and sign-off comment ID.
  It does not receive any Slack secret. The pass wording and review prompt are unchanged.
- Both commands use Click, pinned in the script's PEP 723 metadata and adjacent
  `uv` lockfile. Dependencies are synchronized from trusted code before credentials
  are exposed, then commands run directly from that environment without invoking
  dependency resolution. No repository
  application package or PR dependency is installed in either job.
- `Maintainer Review Slack Handoff` uses `workflow_run` because comments created
  with `GITHUB_TOKEN` do not trigger another `issue_comment` workflow. Human-written
  escalation comments do trigger it.
- The notifier uses trusted default-branch code on a fresh GitHub-hosted runner.
  It never checks out the PR, installs its dependencies, runs an agent, or executes
  downloaded artifacts. Artifact JSON is bounded, parsed in memory, and revalidated
  against GitHub. Signed download redirects never receive the GitHub credential.
- Only the posting step receives the environment's Slack token. Its destination
  and maintainer mentions come from configuration; PR text is sent as plain text.
- One root Slack message is used per PR. Subsequent handoffs reply in its thread.
  A GitHub comment saves the Slack channel/message ID and delivered event keys.
  Because the agent shares the `github-actions[bot]` identity, this saved state is
  HMAC-signed using the notifier's Slack token; a copied or forged bot comment
  cannot redirect a notification or suppress delivery for another PR.
- Events are serialized with `queue: max` (up to 100 pending runs). Identical
  approvals for a SHA and reruns of an escalation comment are deduplicated.
  Review reruns which intentionally produce no new sign-off create no artifact.
- Slack discussion does not create GitHub reviews, dismiss findings, change labels,
  approve a PR, or merge it. A maintainer still records the final decision on GitHub.

## Verification after activation

1. On an open non-draft test PR, have its author post
   `/maintainer-review Testing the maintainer handoff; no review needed.`
2. Verify the configured channel receives one message mentioning the intended
   maintainers, and the PR gets a link to that thread.
3. Rerun the same notifier run: it should not post a duplicate. A second escalation
   comment should post a reply in the existing thread.
4. On the next real agent pass, verify a handoff appears for the reviewed SHA.
   Pushing a different commit before the notifier runs must suppress that handoff.

Local network-free tests run with:

```bash
uv venv work/handoff-tests
uv export --locked --script .github/scripts/maintainer_review.py --format requirements-txt --output-file work/handoff-requirements.txt
uv pip install --python work/handoff-tests/bin/python --require-hashes --requirements work/handoff-requirements.txt
work/handoff-tests/bin/python -m unittest discover -s .github/scripts -p 'test_*.py'
```

They also run in `Validate GitHub Actions`. No Slack messages are sent by these tests.

## Recovery and limits

- A normal failed run can be retried from Actions; successful recorded deliveries
  are no-ops. Artifacts are retained for seven days. Older runs without the collector
  are not backfilled; use a human escalation to request an existing PR's review.
- Slack and GitHub are separate services: a process failure after Slack accepts a
  message but before GitHub saves state can leave a duplicate on retry. A stable
  `client_msg_id` is included, but this is not an exactly-once guarantee. Inspect the
  Slack thread and the logged channel/message timestamp before retrying an ambiguous
  delivery. Failed state writes fail the Action rather than silently claiming success.
- Rotating the Slack token invalidates saved state signatures. The next handoff
  starts a new thread. Likewise, changing the configured channel starts a new thread
  there. Preserve old discussion links when announcing either change.
- A push after a handoff has already been posted does not update the old message.
  Every message names the reviewed SHA; review the current GitHub revision before
  approval and request another Claude review when appropriate.
- The queue holds at most 100 pending runs; GitHub cancels additional runs beyond
  that limit. Monitor failed/cancelled runs during an unusual burst.

Platform references: [GitHub workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows),
[environment protection](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments),
[queued concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency),
and [Slack messaging](https://docs.slack.dev/reference/methods/chat.postMessage/).
