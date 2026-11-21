"""Validate built distributions and exercise an installed wheel in isolation."""

from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def fail(message: str) -> None:
    raise SystemExit(message)


def main() -> None:
    wheels = list(DIST.glob("oribi-*.whl"))
    sdists = list(DIST.glob("oribi-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        fail(f"Expected one wheel and one sdist in {DIST}; found {wheels!r}, {sdists!r}")

    forbidden = (".ipynb", ".env", "__pycache__", ".git/", ".venv/", "tests/")
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = archive.namelist()
        if not any(name.startswith("oribi/") and name.endswith(".py") for name in wheel_names):
            fail("Wheel does not contain the oribi Python package")
        for name in wheel_names:
            if any(marker in name for marker in forbidden):
                fail(f"Unexpected development or sensitive file in wheel: {name}")
        metadata_files = [name for name in wheel_names if name.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            fail("Wheel must contain exactly one METADATA file")
        metadata_lines = archive.read(metadata_files[0]).decode().splitlines()
        metadata_fields = dict(line.split(": ", 1) for line in metadata_lines if ": " in line)
        expected_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
        if metadata_fields.get("Name", "").lower() != "oribi":
            fail("Wheel metadata is missing the expected project name or version")
        if metadata_fields.get("Version") != expected_version:
            fail(
                f"Wheel version {metadata_fields.get('Version')!r} does not match "
                f"pyproject.toml version {expected_version!r}"
            )

    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_names = archive.getnames()
        if not any("/oribi/__init__.py" in name for name in sdist_names):
            fail("Source distribution does not contain the oribi package")
        for name in sdist_names:
            normalized = name.replace("\\", "/")
            if any(marker in normalized for marker in forbidden):
                fail(f"Unexpected development or sensitive file in sdist: {name}")

    with tempfile.TemporaryDirectory(prefix="oribi-install-") as temp:
        env_dir = Path(temp) / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-m", "pip", "install", str(wheels[0])],
            check=True,
            cwd=temp,
        )
        probe = (
            "import oribi; from oribi import inspect; "
            "assert callable(inspect); print(oribi.__file__)"
        )
        result = subprocess.run(
            [str(python), "-c", probe],
            check=True,
            text=True,
            capture_output=True,
            cwd=temp,
        )
        installed_path = Path(result.stdout.strip()).resolve()
        if ROOT in installed_path.parents:
            fail(f"Import resolved to the source checkout instead of the venv: {installed_path}")

    print(f"Validated wheel and sdist: {wheels[0].name}, {sdists[0].name}")


if __name__ == "__main__":
    main()
