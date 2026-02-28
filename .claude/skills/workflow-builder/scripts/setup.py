#!/usr/bin/env python3
"""Manage Roboflow Workflow Builder configuration."""

import argparse
import json
import os
import sys

CONFIG_PATH = os.path.expanduser("~/.roboflow/workflow_skill_config.json")


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"workspaces": {}, "default_workspace": None, "api_url": "https://serverless.roboflow.com"}


def save_config(config):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def get_api_key(config=None):
    """Get the API key, checking env var first, then config."""
    env_key = os.environ.get("ROBOFLOW_API_KEY")
    if env_key:
        return env_key
    if config is None:
        config = load_config()
    default_ws = config.get("default_workspace")
    if default_ws and default_ws in config.get("workspaces", {}):
        return config["workspaces"][default_ws].get("api_key")
    # Return first workspace key if only one exists
    workspaces = config.get("workspaces", {})
    if len(workspaces) == 1:
        return next(iter(workspaces.values())).get("api_key")
    return None


def get_api_url(config=None):
    """Get the inference server URL."""
    env_url = os.environ.get("ROBOFLOW_API_URL")
    if env_url:
        return env_url
    if config is None:
        config = load_config()
    return config.get("api_url", "https://serverless.roboflow.com")


def cmd_status(args):
    config = load_config()
    api_key = get_api_key(config)
    api_url = get_api_url(config)

    if not api_key:
        print("NOT CONFIGURED")
        print()
        print("No API key found. Set one with:")
        print("  python3 setup.py set --api-key <KEY> --workspace <NAME>")
        print()
        print("Or set the ROBOFLOW_API_KEY environment variable.")
        sys.exit(1)

    print("CONFIGURED")
    print(f"  API URL:           {api_url}")
    print(f"  Default workspace: {config.get('default_workspace', '(auto)')}")
    print(f"  API key:           {api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else f"  API key:           {api_key}")

    workspaces = config.get("workspaces", {})
    if len(workspaces) > 1:
        print(f"  Workspaces:        {', '.join(workspaces.keys())}")


def cmd_set(args):
    config = load_config()

    if args.api_url:
        config["api_url"] = args.api_url.rstrip("/")

    if args.api_key and args.workspace:
        if "workspaces" not in config:
            config["workspaces"] = {}
        config["workspaces"][args.workspace] = {"api_key": args.api_key}
        config["default_workspace"] = args.workspace
    elif args.api_key:
        # No workspace specified — use "default"
        if "workspaces" not in config:
            config["workspaces"] = {}
        config["workspaces"]["default"] = {"api_key": args.api_key}
        config["default_workspace"] = "default"

    save_config(config)
    print("Configuration saved.")
    cmd_status(argparse.Namespace())


def cmd_switch(args):
    config = load_config()
    if args.workspace not in config.get("workspaces", {}):
        print(f"Error: workspace '{args.workspace}' not found.")
        print(f"Available: {', '.join(config.get('workspaces', {}).keys())}")
        sys.exit(1)
    config["default_workspace"] = args.workspace
    save_config(config)
    print(f"Switched to workspace: {args.workspace}")


def main():
    parser = argparse.ArgumentParser(description="Roboflow Workflow Skill Config")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show current configuration")

    set_parser = subparsers.add_parser("set", help="Set configuration")
    set_parser.add_argument("--api-key", help="Roboflow API key")
    set_parser.add_argument("--workspace", help="Workspace name")
    set_parser.add_argument("--api-url", help="Inference server URL (default: https://serverless.roboflow.com)")

    switch_parser = subparsers.add_parser("switch", help="Switch default workspace")
    switch_parser.add_argument("workspace", help="Workspace name to switch to")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "set":
        cmd_set(args)
    elif args.command == "switch":
        cmd_switch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
