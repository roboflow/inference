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
    infer_parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="URL of image to run inference on.",
    )
    infer_parser.add_argument(
        "-p",
        "--project_id",
        type=str,
        help="Project to run inference with.",
    )
    infer_parser.add_argument(
        "-v",
        "--model_version",
        type=str,
        help="Version of model to run inference with.",
    )
    infer_parser.add_argument(
        "-a",
        "--api_key",
        type=str,
        help="Path to save output image to.",
    )
    infer_parser.add_argument(
        "-h",
        "--host",
        type=str,
        default="https://localhost:9001",
        help="Host to run inference on.",
    )

    args = parser.parse_args()

    if args.subcommand == "serve":
        print(f"Starting inference server")
    elif args.subcommand == "infer":
        print(f"Running inference")
    else:
        parser.print_help()
