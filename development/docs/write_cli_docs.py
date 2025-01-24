import subprocess
import os 

FILENAME = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "docs",
        "inference_helpers",
        "cli_commands",
        "reference.md"
    )
)

def write_file(path: str, content: str) -> None:
    path = os.path.abspath(path)
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def main():
    cmd = f"typer inference_cli.main utils docs --name inference"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    print(result.stdout)
    write_file(FILENAME, result.stdout)


if __name__ == "__main__":
    main()
