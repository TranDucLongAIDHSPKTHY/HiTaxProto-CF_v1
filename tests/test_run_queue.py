"""Queue lock: one queue at a time; a lock left by a killed queue is replaced, a live one is respected."""
import os
import subprocess
import sys

import pytest

import tools.experiments.run_queue as rq


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_pid_alive():
    assert rq.pid_alive(os.getpid())
    assert not rq.pid_alive(_dead_pid())


def test_stale_lock_is_replaced_and_live_lock_respected(tmp_path, monkeypatch):
    monkeypatch.setattr(rq, "LOCK", tmp_path / "queue" / "queue.lock")
    rq.LOCK.parent.mkdir(parents=True)
    rq.LOCK.write_text(str(_dead_pid()), encoding="utf-8")          # queue killed without clean-up
    rq.acquire_lock()
    assert rq.LOCK.read_text(encoding="utf-8") == str(os.getpid())
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        rq.LOCK.write_text(str(live.pid), encoding="utf-8")
        with pytest.raises(SystemExit):
            rq.acquire_lock()
        assert rq.LOCK.read_text(encoding="utf-8") == str(live.pid)
    finally:
        live.kill(); live.wait()
