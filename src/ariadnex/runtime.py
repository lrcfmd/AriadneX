"""Single-environment Conda routing and encoder subprocess support."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

Command = Sequence[str]
Executor = Callable[..., subprocess.CompletedProcess[str]]


def current_environment_name(environ: Mapping[str, str] | None = None) -> str | None:
    values = os.environ if environ is None else environ
    explicit = values.get("CONDA_DEFAULT_ENV")
    if explicit:
        return explicit
    prefix = values.get("CONDA_PREFIX")
    return Path(prefix).name if prefix else None


@dataclass(frozen=True)
class RuntimeSpec:
    """Conda environment and executable used by the complete workflow."""

    env_name: str = "ariadnex"
    conda_exe: str = "conda"

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "RuntimeSpec":
        values = os.environ if environ is None else environ
        return cls(
            env_name=values.get("ARIADNEX_ENV", "ariadnex"),
            conda_exe=values.get("CONDA_EXE", "conda"),
        )

    def is_current(self, environ: Mapping[str, str] | None = None) -> bool:
        return current_environment_name(environ) == self.env_name


class CondaRunner:
    """Run commands in the one configured AriadneX environment."""

    def __init__(
        self,
        runtime: RuntimeSpec | None = None,
        *,
        executor: Executor | None = None,
        environ: Mapping[str, str] | None = None,
        process_env: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if process_env is not None and env is not None:
            raise ValueError("pass only one of process_env or env")
        self.runtime = runtime or RuntimeSpec.from_environment(environ)
        self.executor = executor or subprocess.run
        self.environ = environ
        selected_env = env if env is not None else process_env
        self.process_env = None if selected_env is None else dict(selected_env)

    def command(self, args: Iterable[str]) -> list[str]:
        inner = [str(arg) for arg in args]
        if not inner:
            raise ValueError("a subprocess command cannot be empty")
        if self.runtime.is_current(self.environ):
            return inner
        return [
            self.runtime.conda_exe,
            "run",
            "--no-capture-output",
            "-n",
            self.runtime.env_name,
            *inner,
        ]

    def run(
        self,
        args: Iterable[str],
        *,
        check: bool = True,
        cwd: str | Path | None = None,
        capture_output: bool = True,
        text: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = self.command(args)
        kwargs: dict[str, object] = {
            "check": check,
            "capture_output": capture_output,
            "shell": False,
            "text": text,
        }
        if cwd is not None:
            kwargs["cwd"] = str(cwd)
        if env is not None:
            kwargs["env"] = dict(env)
        elif self.process_env is not None:
            kwargs["env"] = dict(self.process_env)
        return self.executor(command, **kwargs)

    def python_command(self, module: str, args: Iterable[str] = ()) -> list[str]:
        return self.command(["python", "-m", module, *[str(arg) for arg in args]])

    def run_module(
        self,
        module: str,
        args: Iterable[str] = (),
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return self.run(["python", "-m", module, *[str(arg) for arg in args]], **kwargs)


class EncoderRunner:
    """Run checkpoint inference in a clean child process of the same environment."""

    def __init__(self, runner: CondaRunner | None = None) -> None:
        self.runner = runner or CondaRunner()

    @staticmethod
    def worker_environment(
        *,
        repo_src: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        values = dict(os.environ if environ is None else environ)
        values["PYTHONNOUSERSITE"] = "1"
        paths = [] if repo_src is None else [str(Path(repo_src).expanduser())]
        if paths:
            values["PYTHONPATH"] = os.pathsep.join(paths)
        else:
            values.pop("PYTHONPATH", None)
        return values

    def command(
        self,
        module: str = "ariadnex.library.embedding_worker",
        args: Iterable[str] = (),
    ) -> list[str]:
        return self.runner.python_command(module, args)

    def run_module(
        self,
        module: str = "ariadnex.library.embedding_worker",
        args: Iterable[str] = (),
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if "env" not in kwargs:
            kwargs["env"] = self.worker_environment()
        return self.runner.run_module(module, args, **kwargs)


def launcher_command(
    argv: Iterable[str],
    *,
    runtime: RuntimeSpec | None = None,
    environ: Mapping[str, str] | None = None,
    python_executable: str | None = None,
) -> list[str]:
    """Construct the command used by the top-level workflow launcher."""

    selected = runtime or RuntimeSpec.from_environment(environ)
    runner = CondaRunner(selected, environ=environ)
    python = (
        (python_executable or sys.executable or "python")
        if selected.is_current(environ)
        else "python"
    )
    return runner.command([python, "-m", "ariadnex.cli", *[str(arg) for arg in argv]])


__all__ = [
    "CondaRunner",
    "EncoderRunner",
    "Executor",
    "RuntimeSpec",
    "current_environment_name",
    "launcher_command",
]
