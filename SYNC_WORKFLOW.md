# Inference Private Sync Workflow

This repo is a private fork of [roboflow/inference](https://github.com/roboflow/inference) used for developing features under NDA before they can be contributed back to the public repo.

## Setup (Already Done)

The repo is configured with two remotes:
- `origin`: Points to `roboflow/inference-private` (this private repo)
- `upstream`: Points to `roboflow/inference` (the public repo)

## Keeping Up to Date

### Automatic Sync (Recommended)

Use the provided script to sync with the latest upstream changes:

```bash
./sync-with-upstream.sh
```

This script will:
1. Check that you're on the main branch with no uncommitted changes
2. Fetch the latest changes from upstream
3. Create a timestamped backup branch
4. Reset main to match upstream/main
5. Force push to update the private repo

### Manual Sync

If you prefer to do it manually:

```bash
# Fetch latest from upstream
git fetch upstream main

# Create backup (optional but recommended)
git branch backup-$(date +%Y%m%d-%H%M%S)

# Reset to upstream
git reset --hard upstream/main

# Push to private repo
git push origin main --force
```

## Development Workflow

1. **Start with latest code**: Run `./sync-with-upstream.sh` to ensure you're up to date
2. **Create feature branch**: `git checkout -b feature/your-feature-name`
3. **Develop your feature**: Make your changes in the feature branch
4. **Test thoroughly**: Ensure your changes work as expected
5. **Merge to main**: When ready, merge your feature branch to main
6. **Sync before contributing back**: Before creating a PR to the public repo, sync again to ensure compatibility

## Contributing Back to Public Repo

When your NDA-protected feature is ready to be public:

1. Ensure your changes are on a clean feature branch
2. Push the feature branch to the public repo: `git push upstream feature/your-feature-name`
3. Create a PR from the public repo interface

## Important Notes

- **Always sync before starting new work** to avoid conflicts
- **Feature branches are preserved** during syncs (only main is reset)
- **Backup branches are created** automatically for safety
- **Force push is required** since we're rewriting history on main
