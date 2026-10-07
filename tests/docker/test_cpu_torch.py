"""Keep CPU image dependencies on CPU wheels without flattening layers."""

import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILES = [
    ROOT / f"docker/dockerfiles/Dockerfile.onnx.{suffix}"
    for suffix in ("cpu", "cpu.dev", "cpu.slim", "cpu.parallel")
]


def _instructions(contents):
    """Join Docker continuations, ignoring full-line comments even within them."""
    pending = ""
    for line in contents.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        continued = stripped.endswith("\\")
        pending += stripped[:-1] + " " if continued else stripped
        if continued:
            continue

        instruction, _, arguments = pending.partition(" ")
        yield instruction.upper(), arguments.strip()
        pending = ""

    assert not pending, "Unterminated Docker instruction"


def _shell_commands(arguments):
    """Split shell lists without treating quoted separators as commands."""
    lexer = shlex.shlex(arguments, posix=True, punctuation_chars=";&|\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    command = []
    for token in lexer:
        if token and all(character in ";&|\n" for character in token):
            if command:
                yield command
                command = []
        else:
            command.append(token)

    if command:
        yield command


def _assert_cpu_recipe(contents):
    """Track the constraint through instructions and every subsequent pip install."""
    constraint_path = "/requirements.torch-cpu.txt"
    constraint = None
    copied = False
    installed = False
    for instruction, arguments in _instructions(contents):
        if instruction == "FROM":
            constraint = None
            copied = installed = False
        elif instruction == "COPY":
            copied |= shlex.split(arguments) == [
                "requirements/requirements.torch-cpu.txt",
                constraint_path,
            ]
        elif instruction in {"ARG", "ENV"}:
            fields = shlex.split(arguments)
            if fields and fields[0] == "PIP_CONSTRAINT":
                constraint = " ".join(fields[1:])
            else:
                for field in fields:
                    if field.startswith("PIP_CONSTRAINT="):
                        constraint = field.partition("=")[2]
        elif instruction == "RUN":
            shell_constraint = constraint
            for command in _shell_commands(arguments):
                if "unset" in command and "PIP_CONSTRAINT" in command:
                    shell_constraint = None

                overrides = [
                    token.partition("=")[2]
                    for token in command
                    if token.startswith("PIP_CONSTRAINT=")
                ]
                effective_constraint = overrides[-1] if overrides else shell_constraint
                pip_index = next(
                    (
                        index
                        for index, token in enumerate(command)
                        if token in {"pip", "pip3"}
                        and command[index + 1 : index + 2] == ["install"]
                    ),
                    None,
                )
                if pip_index is None:
                    if overrides:
                        shell_constraint = overrides[-1]
                    continue

                # Accept only executable pip invocations, including python -m pip.
                prefix = command[:pip_index]
                assert not any(token in {"echo", "printf"} for token in prefix)
                options = command[pip_index + 2 :]
                cpu_install = (
                    "--index-url" in options
                    and options[options.index("--index-url") + 1]
                    == "https://download.pytorch.org/whl/cpu"
                    and "-r" in options
                    and options[options.index("-r") + 1] == constraint_path
                )
                if cpu_install:
                    assert copied, "CPU requirements must be copied before installing"
                    installed = True

                if installed:
                    assert (
                        effective_constraint == constraint_path
                    ), f"Unconstrained pip install: {' '.join(command)}"
                else:
                    # Only the existing pip/wheel bootstrap may precede CPU wheels.
                    packages = [token for token in options if not token.startswith("-")]
                    assert packages and all(
                        token == "pip" or token.startswith("wheel>=")
                        for token in packages
                    ), "Dependencies resolved before CPU wheels were installed"

    assert installed, "Missing executable CPU-index install"


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_cpu_torch_install_precedes_dependency_resolution(dockerfile):
    """Install CPU wheels and constrain every later pip dependency resolution.

    Args:
        dockerfile (Path): CPU image recipe to check.
    """
    _assert_cpu_recipe(dockerfile.read_text())


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_cpu_images_preserve_dependency_layers(dockerfile):
    """Retain base layers so Docker can share and pull them independently.

    Args:
        dockerfile (Path): CPU image recipe to check.
    """
    contents = dockerfile.read_text()
    assert "FROM scratch" not in contents
    assert "COPY --from=base / /" not in contents


def test_cpu_torch_constraints_preserve_resolved_versions():
    """Pin the existing image versions and retain the required NVML binding."""
    constraints = ROOT / "requirements/requirements.torch-cpu.txt"
    assert constraints.read_text().splitlines() == [
        "torch==2.14.0+cpu",
        "torchvision==0.29.0+cpu",
    ]
    assert (
        "nvidia-ml-py<13.0.0"
        in (ROOT / "requirements/requirements.cpu.txt").read_text()
    )


CPU_INSTALL = (
    "RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "
    "-r /requirements.torch-cpu.txt"
)
CONSTRAINT = "ARG PIP_CONSTRAINT=/requirements.torch-cpu.txt"


@pytest.mark.parametrize(
    "mutation",
    [
        "remove_install",
        "comment_install",
        "comment_constraint",
        "clear_constraint",
        "cuda_reinstall",
        "env_clear",
        "unset_constraint",
        "python_reinstall",
        "triton_reinstall",
        "nvidia_reinstall",
    ],
)
def test_cpu_recipe_rejects_disabled_protection(tmp_path, mutation):
    """Reject recipes that disable CPU wheel protection.

    Args:
        tmp_path (Path): Temporary directory for the mutated recipe.
        mutation (str): Disabled protection to exercise.
    """
    contents = DOCKERFILES[0].read_text()
    replacements = {
        "remove_install": (CPU_INSTALL, ""),
        "comment_install": (CPU_INSTALL, "# " + CPU_INSTALL),
        "comment_constraint": (CONSTRAINT, "# " + CONSTRAINT),
        "clear_constraint": (CPU_INSTALL, CPU_INSTALL + "\nARG PIP_CONSTRAINT="),
        "cuda_reinstall": (
            CPU_INSTALL,
            CPU_INSTALL + "\nRUN PIP_CONSTRAINT= pip3 install --force-reinstall "
            "torch==2.14.0 torchvision==0.29.0",
        ),
        "env_clear": (CPU_INSTALL, CPU_INSTALL + "\nENV PIP_CONSTRAINT="),
        "unset_constraint": (
            CPU_INSTALL,
            CPU_INSTALL + "\nRUN unset PIP_CONSTRAINT && pip3 install torch",
        ),
        "python_reinstall": (
            CPU_INSTALL,
            CPU_INSTALL + "\nRUN PIP_CONSTRAINT= python3 -m pip install torch",
        ),
        "triton_reinstall": (
            CPU_INSTALL,
            CPU_INSTALL + "\nRUN PIP_CONSTRAINT= pip3 install triton",
        ),
        "nvidia_reinstall": (
            CPU_INSTALL,
            CPU_INSTALL + "\nRUN PIP_CONSTRAINT= pip3 install nvidia-cublas-cu12",
        ),
    }
    old, new = replacements[mutation]
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(contents.replace(old, new))

    with pytest.raises(AssertionError):
        test_cpu_torch_install_precedes_dependency_resolution(dockerfile)


def test_cpu_recipe_accepts_comments_and_continuations():
    """Parse continued instructions without interpreting comments as commands."""
    contents = (
        DOCKERFILES[0]
        .read_text()
        .replace(
            CPU_INSTALL,
            "# RUN PIP_CONSTRAINT= pip3 install torch\n"
            "RUN pip3 install --no-cache-dir \\\n"
            "    # A comment within a continued instruction.\n"
            "    --index-url https://download.pytorch.org/whl/cpu \\\n"
            "    -r /requirements.torch-cpu.txt",
        )
    )
    _assert_cpu_recipe(contents)
