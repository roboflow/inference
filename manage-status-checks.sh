#!/bin/bash

# Helper script to manage required status checks for branch protection
# Usage: 
#   ./manage-status-checks.sh add "check-name"
#   ./manage-status-checks.sh remove "check-name" 
#   ./manage-status-checks.sh list
#   ./manage-status-checks.sh sync-from-upstream

set -e

ACTION="$1"
CHECK_NAME="$2"

case "$ACTION" in
    "add")
        if [ -z "$CHECK_NAME" ]; then
            echo "❌ Usage: $0 add \"check-name\""
            exit 1
        fi
        echo "➕ Adding required status check: $CHECK_NAME"
        gh api -X PATCH repos/roboflow/inference-private/branches/main/protection/required_status_checks \
            --field contexts[]="$CHECK_NAME"
        echo "✅ Added: $CHECK_NAME"
        ;;
    
    "remove")
        if [ -z "$CHECK_NAME" ]; then
            echo "❌ Usage: $0 remove \"check-name\""
            exit 1
        fi
        echo "➖ Removing required status check: $CHECK_NAME"
        # Get current contexts, remove the specified one, and update
        current_contexts=$(gh api repos/roboflow/inference-private/branches/main/protection/required_status_checks --jq '.contexts | map(select(. != "'"$CHECK_NAME"'"))')
        gh api -X PATCH repos/roboflow/inference-private/branches/main/protection/required_status_checks \
            --field contexts="$current_contexts"
        echo "✅ Removed: $CHECK_NAME"
        ;;
    
    "list")
        echo "📋 Current required status checks:"
        gh api repos/roboflow/inference-private/branches/main/protection/required_status_checks --jq '.contexts[]' || echo "  None"
        ;;
    
    "sync-from-upstream")
        echo "🔄 Syncing status checks from upstream..."
        upstream_contexts=$(gh api repos/roboflow/inference/branches/main/protection/required_status_checks --jq '.contexts' 2>/dev/null || echo '[]')
        
        if [ "$upstream_contexts" != "[]" ]; then
            gh api -X PATCH repos/roboflow/inference-private/branches/main/protection/required_status_checks \
                --field contexts="$upstream_contexts"
            echo "✅ Synced status checks from upstream:"
            echo "$upstream_contexts" | jq -r '.[]' | sed 's/^/  - /'
        else
            echo "ℹ️  No status checks found on upstream"
        fi
        ;;
    
    *)
        echo "Usage: $0 {add|remove|list|sync-from-upstream} [check-name]"
        echo ""
        echo "Commands:"
        echo "  add \"check-name\"      Add a required status check"
        echo "  remove \"check-name\"   Remove a required status check"
        echo "  list                  List current required status checks"
        echo "  sync-from-upstream    Copy all status checks from public repo"
        echo ""
        echo "Examples:"
        echo "  $0 add \"build-dev-test\""
        echo "  $0 add \"CodeQL\""
        echo "  $0 list"
        echo "  $0 sync-from-upstream"
        exit 1
        ;;
esac
