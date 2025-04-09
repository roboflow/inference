import os
import subprocess


def on_pre_build(config):
    """Run npm build before mkdocs build"""
    print("Building theme assets...")
    theme_dir = os.path.dirname(os.path.abspath(__file__))

    # Run npm install if node_modules doesn't exist
    if not os.path.exists(os.path.join(theme_dir, 'node_modules')):
        subprocess.run(['npm', 'install'], cwd=theme_dir, check=True)

    # Run the build
    subprocess.run(['npm', 'run', 'build'], cwd=theme_dir, check=True)