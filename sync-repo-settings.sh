#!/bin/bash

# Script to sync repository settings from roboflow/inference to roboflow/inference-private
# Usage: ./sync-repo-settings.sh

set -e

echo "🔧 Syncing repository settings from upstream to private repo..."

# Repository settings
echo "📋 Syncing general repository settings..."
upstream_settings=$(gh repo view roboflow/inference --json deleteBranchOnMerge,mergeCommitAllowed,squashMergeAllowed,rebaseMergeAllowed)

delete_branch=$(echo "$upstream_settings" | jq -r '.deleteBranchOnMerge')
merge_commit=$(echo "$upstream_settings" | jq -r '.mergeCommitAllowed')
squash_merge=$(echo "$upstream_settings" | jq -r '.squashMergeAllowed')
rebase_merge=$(echo "$upstream_settings" | jq -r '.rebaseMergeAllowed')

# Apply repository settings
if [ "$delete_branch" = "true" ]; then
    gh repo edit roboflow/inference-private --delete-branch-on-merge
fi

if [ "$merge_commit" = "true" ]; then
    gh repo edit roboflow/inference-private --enable-merge-commit
fi

if [ "$squash_merge" = "true" ]; then
    gh repo edit roboflow/inference-private --enable-squash-merge
fi

if [ "$rebase_merge" = "true" ]; then
    gh repo edit roboflow/inference-private --enable-rebase-merge
fi

echo "✅ Repository settings synced"

# Branch protection rules
echo "🛡️  Syncing branch protection rules..."

# Get upstream branch protection
upstream_protection=$(gh api repos/roboflow/inference/branches/main/protection 2>/dev/null || echo "{}")

if [ "$upstream_protection" != "{}" ]; then
    # Extract key settings (excluding specific status checks that might not exist in private repo)
    enforce_admins=$(echo "$upstream_protection" | jq -r '.enforce_admins.enabled // false')
    dismiss_stale=$(echo "$upstream_protection" | jq -r '.required_pull_request_reviews.dismiss_stale_reviews // false')
    require_code_owners=$(echo "$upstream_protection" | jq -r '.required_pull_request_reviews.require_code_owner_reviews // false')
    require_last_push=$(echo "$upstream_protection" | jq -r '.required_pull_request_reviews.require_last_push_approval // false')
    required_reviewers=$(echo "$upstream_protection" | jq -r '.required_pull_request_reviews.required_approving_review_count // 1')
    strict_checks=$(echo "$upstream_protection" | jq -r '.required_status_checks.strict // true')
    
    # Create protection payload (start with empty status checks - add manually as needed)
    protection_payload=$(cat <<EOF
{
  "required_status_checks": {
    "strict": $strict_checks,
    "contexts": []
  },
  "enforce_admins": $enforce_admins,
  "required_pull_request_reviews": {
    "required_approving_review_count": $required_reviewers,
    "dismiss_stale_reviews": $dismiss_stale,
    "require_code_owner_reviews": $require_code_owners,
    "require_last_push_approval": $require_last_push
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
    )
    
    # Apply branch protection
    echo "$protection_payload" | gh api -X PUT repos/roboflow/inference-private/branches/main/protection --input -
    echo "✅ Branch protection rules synced"
else
    echo "ℹ️  No branch protection found on upstream"
fi

echo ""
echo "🎉 Repository settings sync complete!"
echo ""
echo "📝 Note: Status check contexts are not automatically synced."
echo "   Add them manually as your CI/CD workflows are configured:"
echo "   gh api -X PATCH repos/roboflow/inference-private/branches/main/protection/required_status_checks \\"
echo "     --field contexts[]='your-check-name'"
