"""GPU lock helper (tools/bench/gpu_lock.py): main repository path, take, respect, release."""

import os
import subprocess
import sys

from tools.bench import gpu_lock


def test_lock_is_in_the_main_repository():
    # From a worktree, git-common-dir points to the main checkout's .git
    root = gpu_lock.main_repo_root()
    assert (root / ".git").exists()
    assert gpu_lock.default_lock_path() == root / "data" / ".gpu-lock" or \
        "GENESIS_GPU_LOCK" in os.environ


def test_take_and_release(tmp_path):
    path = tmp_path / ".gpu-lock"
    lock = gpu_lock.GpuLock("test", "unit test", path)
    assert lock.acquire()
    fields = gpu_lock.read_lock(path)
    assert fields["agent"] == "test" and fields["pid"] == str(os.getpid())
    assert "start" in fields and fields["task"] == "unit test"
    # A second taker sees a live holder
    assert not gpu_lock.GpuLock("other", "x", path).acquire()
    lock.release()
    assert not path.exists()


def test_live_and_stale_locks(tmp_path):
    path = tmp_path / ".gpu-lock"
    # Live holder: a child process that is still running
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        path.write_text(f"agent: other\npid: {child.pid}\nstart: now\ntask: busy\n", encoding="utf-8")
        device, lock = gpu_lock.choose_device("cuda", "test", "x", path, log=lambda *_: None)
        assert device == "cpu" and lock is None
        assert path.exists()
    finally:
        child.kill()
        child.wait()
    # Dead PID: still respected by default (the PID may be a shell that launched the GPU job)...
    assert not gpu_lock.GpuLock("test", "x", path).acquire()
    assert path.exists()
    # ... and taken over only on request
    lock = gpu_lock.GpuLock("test", "x", path, take_stale=True)
    assert lock.acquire()
    lock.release()
    assert not path.exists()


def test_release_never_removes_someone_elses_lock(tmp_path):
    path = tmp_path / ".gpu-lock"
    lock = gpu_lock.GpuLock("test", "x", path)
    assert lock.acquire()
    path.write_text("agent: other\npid: 1\n", encoding="utf-8")
    lock.release()
    assert path.exists()


def test_cpu_request_never_touches_the_lock(tmp_path):
    path = tmp_path / ".gpu-lock"
    assert gpu_lock.choose_device("cpu", "test", "x", path) == ("cpu", None)
    assert not path.exists()
