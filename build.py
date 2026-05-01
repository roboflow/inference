import os
import subprocess


# Directory containing npm tooling (package.json, tailwind.config.js, node_modules).
_BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theme_build")

# Source files that, when changed, require a rebuild of theme assets.
_SOURCE_GLOBS = (
    "../theme/assets/tailwind.css",
    "../theme/assets/home.js",
    "tailwind.config.js",
    "package.json",
)

# Output files produced by `npm run build`.
_OUTPUT_FILES = (
    "../docs/static/styles.css",
    "../docs/static/dist/home.js",
)


def _needs_rebuild(build_dir):
    """Return True if source files are newer than output files."""
    try:
        oldest_output = min(
            os.path.getmtime(os.path.join(build_dir, p))
            for p in _OUTPUT_FILES
            if os.path.exists(os.path.join(build_dir, p))
        )
    except ValueError:
        # No output files exist yet – must build.
        return True

    for pattern in _SOURCE_GLOBS:
        path = os.path.join(build_dir, pattern)
        if os.path.exists(path) and os.path.getmtime(path) > oldest_output:
            return True
    return False


def on_pre_build(config):
    """Run npm build before site build, but only when source files changed."""
    build_dir = os.path.normpath(_BUILD_DIR)

    # Run npm install if node_modules doesn't exist
    if not os.path.exists(os.path.join(build_dir, "node_modules")):
        subprocess.run(["npm", "install"], cwd=build_dir, check=True)

    if _needs_rebuild(build_dir):
        print("Building theme assets...")
        subprocess.run(["npm", "run", "build"], cwd=build_dir, check=True)
    else:
        print("Theme assets up to date, skipping build.")
