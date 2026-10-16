"""Fresh-process checks that the installed package exposes the documented task name.

Each test installs the package into a new virtual environment with uv, then runs
the documented `inspect eval biomysterybench/bio_mystery_bench` command from a
directory outside the repository. An invalid `source` argument makes the task
raise its own ValueError, which proves Inspect resolved the name to this task
without downloading the dataset or calling a model.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK = "biomysterybench/bio_mystery_bench"

pytestmark = [
    pytest.mark.install,
    pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv on PATH"),
]


def _run(
    *args: str | Path, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(a) for a in args], cwd=cwd, capture_output=True, text=True, timeout=600
    )


def _venv(tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    result = _run("uv", "venv", "--python", sys.executable, venv)
    assert result.returncode == 0, result.stderr
    return venv


def _install(venv: Path, *target: str | Path) -> None:
    result = _run("uv", "pip", "install", "--python", venv / "bin" / "python", *target)
    assert result.returncode == 0, result.stderr


def _assert_task_resolves(venv: Path, cwd: Path) -> None:
    result = _run(
        venv / "bin" / "inspect",
        "eval",
        TASK,
        "-T",
        "source=not-a-source",
        "--model",
        "mockllm/model",
        "--log-dir",
        cwd / "logs",
        cwd=cwd,
    )
    output = result.stdout + result.stderr
    assert "not found" not in output, output
    assert "source must be 'full' or 'preview'" in output, output


def test_task_resolves_after_editable_install(tmp_path):
    venv = _venv(tmp_path)
    _install(venv, "-e", REPO_ROOT)
    _assert_task_resolves(venv, tmp_path)


def test_task_resolves_after_wheel_install(tmp_path):
    dist = tmp_path / "dist"
    result = _run("uv", "build", "--wheel", "--out-dir", dist, REPO_ROOT)
    assert result.returncode == 0, result.stderr
    [wheel] = dist.glob("*.whl")
    venv = _venv(tmp_path)
    _install(venv, wheel)
    _assert_task_resolves(venv, tmp_path)
