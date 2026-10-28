"""Run a plan of training runs strictly ONE AT A TIME (never two runs on the GPU at once).

A plan is a JSON file: {"name": ..., "runs": [["--model", "LightGCN", "--dataset", "yelp2018", ...], ...]}.
Each entry is passed to main.py with --resume; main.py skips a run already completed with the same configuration and
resumes an interrupted one, so a queue can be restarted safely. A lock file prevents a second queue from starting.

    python -m tools.experiments.run_queue plans/pilot_yelp2018.json
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import MODEL_OUTPUT_DIR  # noqa: E402

LOCK = MODEL_OUTPUT_DIR / "queue" / "queue.lock"


def pid_alive(pid):
    """True if a process with this id exists (never signals it: os.kill(pid, 0) TERMINATES the process on Windows)."""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x1000, False, int(pid))        # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
        k32.CloseHandle(handle)
        return bool(ok) and code.value == 259                     # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock():
    """Create the lock; a lock left by a queue that was killed (its process no longer exists) is replaced."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            text = LOCK.read_text(encoding="utf-8").strip()
            if text.isdigit() and not pid_alive(int(text)):
                print("removing stale lock {} (queue process {} no longer exists)".format(LOCK, text), flush=True)
                LOCK.unlink(missing_ok=True)
                continue
            raise SystemExit("another queue is running (lock {}, process {}); runs are executed one at a time"
                             .format(LOCK, text))
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return
    raise SystemExit("could not acquire the queue lock {}".format(LOCK))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit(__doc__)
    plan = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    acquire_lock()
    status_path = MODEL_OUTPUT_DIR / "queue" / "{}.status.json".format(plan["name"])
    status = {"plan": plan["name"], "started_at_utc": datetime.now(timezone.utc).isoformat(), "runs": []}
    try:
        for k, args in enumerate(plan["runs"], start=1):
            t0 = time.time()
            print("[{}/{}] python main.py {}".format(k, len(plan["runs"]), " ".join(args)), flush=True)
            rc = subprocess.run([sys.executable, str(ROOT / "main.py"), *args, "--resume"], cwd=ROOT).returncode
            status["runs"].append({"args": args, "returncode": rc, "seconds": round(time.time() - t0, 1)})
            with open(status_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(status, f, indent=1); f.write("\n")
        status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        with open(status_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(status, f, indent=1); f.write("\n")
    finally:
        LOCK.unlink(missing_ok=True)
    return 0 if all(r["returncode"] == 0 for r in status["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
