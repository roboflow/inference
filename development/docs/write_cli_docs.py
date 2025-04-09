import os
import subprocess

DOCS_ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "docs",
    )
)

filename = os.path.join(DOCS_ROOT_DIR, "inference_helpers", "cli_commands", "reference.md")

def main():
    cmd = f"typer inference_cli/main.py utils docs --name inference"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    content = result.stdout
    error = result.stderr
    status = result.returncode

    print("CONTENT length: ", len(content))
    print("CONTENT type: ", type(content))
    print("ERROR length: ", len(error))
    print("ERROR: ", error)
    print("STATUS: ", status)

    print("Writing CLI reference to ", filename)
    
    with open(filename, 'w', encoding='utf-8') as f:
       f.write(content)


if __name__ == "__main__":
    main()
