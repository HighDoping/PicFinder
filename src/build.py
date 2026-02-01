import os
import subprocess
import sys
from pathlib import Path


def get_main_directory():
    return Path(__file__).parent.resolve()


def get_version():
    with open(get_main_directory() / "pyproject.toml") as f:
        for line in f:
            if "version" in line:
                return line.split("=")[1].strip().strip('"')


def get_commit_hash():
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .strip()
        .decode("utf-8")
    )


def save_version():
    with open(get_main_directory() / "VERSION", "w") as f:
        f.write(get_version())
        f.write("\n")
        f.write(get_commit_hash())


def build_nuitka():
    main_dir = get_main_directory()
    os_name = os.name  # 'nt' for Windows, 'posix' for Unix/Linux

    # Base Nuitka options
    nuitka_options = [
        "--standalone",
        "--enable-plugin=pyside6",
        f"--include-data-files={main_dir}/VERSION=VERSION",
        f"--include-data-files={main_dir}/icon.ico=icon.ico",
        f"--windows-icon-from-ico={main_dir}/icon.ico",
        f"--include-data-dir={main_dir}/models=models",
        f"--include-data-dir={main_dir}/.venv/Lib/site-packages/rapidocr_onnxruntime=rapidocr_onnxruntime",
        "--windows-console-mode=disable",
        "--product-name=PicFinder",
        f"--file-version={get_version()}",
        f"--product-version={get_version()}",
        "--copyright=Copyright © 2024-2026 HighDoping",
    ]

    # OS-specific options
    if sys.platform.startswith("win"):
        # Windows-specific data directories and files
        windows_dir = main_dir / "backend/libsimple/libsimple-windows-x64"
        nuitka_options.extend(
            [
                f"--include-data-dir={windows_dir}=lib/backend/libsimple/libsimple-windows-x64",
                f"--include-data-files={windows_dir}/simple.dll=lib/backend/libsimple/libsimple-windows-x64/simple.dll",
            ]
        )
    elif sys.platform.startswith("linux"):
        # Linux-specific data directories and files
        linux_dir = main_dir / "backend/libsimple/libsimple-linux-ubuntu-latest"
        nuitka_options.extend(
            [
                f"--include-data-dir={linux_dir}=lib/backend/libsimple/libsimple-linux-ubuntu-latest",
                f"--include-data-files={linux_dir}/simple.so=lib/backend/libsimple/libsimple-linux-ubuntu-latest/simple.so",
            ]
        )
    else:
        print(f"Unsupported OS: {sys.platform}")
        sys.exit(1)

    # Define the main Python script to compile
    main_script = main_dir / "main.py"  # Replace with your actual main script

    # Construct the Nuitka command
    command = (
        ["poetry", "run", "python", "-m", "nuitka"]
        + nuitka_options
        + [str(main_script)]
    )

    print("Executing Nuitka command:")
    print(" ".join(command))

    # Execute the Nuitka build
    subprocess.check_call(command)


if __name__ == "__main__":
    save_version()
    build_nuitka()
