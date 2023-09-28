import argparse


def main():
    parser = argparse.ArgumentParser(description="Inference CLI coming soon!")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    serve_parser = subparsers.add_parser("serve", help="Start the inference server.")
    serve_parser.add_argument(
        "--port",
        type=int,
        default=9001,
        help="Port to run the inference server on.",
    )

    infer_parser = subparsers.add_parser("infer", help="Run inference.")

    args = parser.parse_args()

    if args.subcommand == "serve":
        print(f"Starting inference server")
    elif args.subcommand == "infer":
        print(f"Running inference")
    else:
        parser.print_help()
