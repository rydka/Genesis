"""Shared GPU lock of the main repository (rule .agents/rules/gpu-et-environnement.md).

The lock is always `<main repository>/data/.gpu-lock`, whatever the working copy a tool runs from: a
git worktree (for example under .claude/worktrees/) shares the GPU with the main checkout, so it must
see the same lock. The main repository is found with `git rev-parse --git-common-dir`; the path can
also be given explicitly (`--gpu-lock`) or through the GENESIS_GPU_LOCK environment variable.

Usage:
    lock = GpuLock(agent="fit_silhouette", task="...")
    if lock.acquire():          # False: another live process holds it -> stay on the CPU
        try: ...
        finally: lock.release()
or as a context manager (`with GpuLock(...) as held:`), `held` telling whether it was taken.

An existing lock is respected by default even when the PID it names no longer runs: agents often
write the PID of a short-lived shell while the job they launched keeps the GPU (seen on 2026-11-06),
so a dead PID does not prove the GPU is free. `take_stale=True` (option --take-stale-lock) takes over
a lock whose process is gone, for a human who checked. The file is created atomically, so two tools
started at the same time cannot both believe they hold it.
"""

import atexit
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent


def main_repo_root(start: Optional[Path] = None) -> Path:
    """Root of the main checkout (parent of the common .git directory); the working copy otherwise."""
    start = Path(start) if start else HERE
    try:
        out = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                             cwd=str(start), capture_output=True, text=True, timeout=10, check=True)
        common = Path(out.stdout.strip())
        if common.name == ".git":
            return common.parent
    except (OSError, subprocess.SubprocessError):
        pass
    return HERE.parents[1]


def default_lock_path() -> Path:
    env = os.environ.get("GENESIS_GPU_LOCK")
    return Path(env) if env else main_repo_root() / "data" / ".gpu-lock"


def pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True,
                             text=True, check=False)
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_lock(path: Path) -> dict:
    """Fields of a lock file ("key: value" lines); {} when absent or unreadable."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    fields = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z_]+)\s*:\s*(.*)$", line)
        if m:
            fields[m.group(1).lower()] = m.group(2).strip()
    return fields


def lock_holder_alive(path: Path) -> bool:
    """True when the lock exists and names a running process (or names none: assume alive)."""
    if not path.exists():
        return False
    fields = read_lock(path)
    try:
        return pid_running(int(fields.get("pid", "")))
    except ValueError:
        return True


class GpuLock:
    """The GPU lock file, taken with the agent name, PID, start time and task (rule format)."""

    def __init__(self, agent: str, task: str, path: Optional[Path] = None, take_stale: bool = False):
        self.agent, self.task, self.take_stale = agent, task, take_stale
        self.path = Path(path) if path else default_lock_path()
        self.held = False

    def describe_holder(self) -> str:
        f = read_lock(self.path)
        desc = ", ".join(f"{k} {v}" for k, v in f.items()) or "unreadable"
        return desc + ("" if lock_holder_alive(self.path) else "; its PID no longer runs")

    def acquire(self) -> bool:
        if self.held:
            return True
        if self.path.exists():
            if not self.take_stale or lock_holder_alive(self.path):
                return False
            self.path.unlink(missing_ok=True)      # stale, and taking it over was asked for
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = (f"agent: {self.agent}\npid: {os.getpid()}\n"
                f"start: {time.strftime('%Y-%m-%dT%H:%M:%S')}\ntask: {self.task}\n")
        try:
            with open(self.path, "x", encoding="utf-8") as fh:   # atomic: fails if created meanwhile
                fh.write(body)
        except FileExistsError:
            return False
        self.held = True
        atexit.register(self.release)
        return True

    def release(self):
        if not self.held:
            return
        self.held = False
        # Only remove our own lock
        if read_lock(self.path).get("pid") == str(os.getpid()):
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


def choose_device(requested: str, agent: str, task: str, lock_path: Optional[Path] = None,
                  log=print, take_stale: bool = False):
    """Device to use and the lock held for it (None on the CPU).

    "cpu" -> cpu. "cuda" or "auto" -> cuda if CUDA is available and the lock can be taken, otherwise
    the CPU (the lock of another live job is respected, never overridden).
    """
    if requested == "cpu":
        return "cpu", None
    import torch
    if not torch.cuda.is_available():
        if requested != "auto":
            log("CUDA is not available: running on the CPU")
        return "cpu", None
    lock = GpuLock(agent, task, lock_path, take_stale)
    if not lock.acquire():
        log(f"GPU lock {lock.path} held ({lock.describe_holder()}): running on the CPU")
        return "cpu", None
    log(f"GPU lock {lock.path} taken")
    return (requested if requested.startswith("cuda") else "cuda"), lock
