#!/bin/bash

# Script to sync inference-private with roboflow/inference
# Usage: ./sync-with-upstream.sh

set -e

echo "🔄 Syncing inference-private with upstream roboflow/inference..."

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check if we're on main branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo "❌ Please switch to main branch first (currently on: $current_branch)"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

# Fetch latest from upstream
echo "📥 Fetching latest changes from upstream..."
git fetch upstream main

# Show what we're about to sync
commits_behind=$(git rev-list --count HEAD..upstream/main)
echo "📊 Local main is $commits_behind commits behind upstream/main"

if [ "$commits_behind" -eq 0 ]; then
    echo "✅ Already up to date!"
    exit 0
fi

# Create backup branch with timestamp
backup_branch="backup-$(date +%Y%m%d-%H%M%S)"
echo "💾 Creating backup branch: $backup_branch"
git branch "$backup_branch"

# Reset to upstream/main
echo "🔄 Resetting to upstream/main..."
git reset --hard upstream/main

# Push to origin
echo "📤 Pushing to origin..."
git push origin main --force

echo "✅ Sync complete!"
echo "📋 Summary:"
echo "   - Synced $commits_behind commits from upstream"
echo "   - Created backup branch: $backup_branch"
echo "   - Updated origin/main"
