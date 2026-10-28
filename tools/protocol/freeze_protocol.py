"""Freeze the protocol: after this file exists, the sealed test split may be opened ONCE per selected run.

Input: protocol/protocol_draft.json, written by hand and reviewed before freezing:
  {"hypotheses": ..., "endpoint": ..., "families": ..., "margin": ..., "selection_rule": ...,
   "seeds": [...], "datasets": ["amazon-book/2026", ...], "runs": ["log/<study>/<dataset>/<model>/<setting>", ...]}
The frozen file adds: the sealed test SHA-256 of every dataset, the SHA-256 of every selected run's resolved
configuration and selected checkpoint, the SHA-256 of protocol/protocol_plan.md, the code hash
(utility.utility_function.tools.code_hash, as in every run's environment.json) and the git commit, and "frozen": true.
Freezing is refused when tracked files differ from the git commit (the commit would not describe the code). An existing
frozen protocol is never overwritten.

    python -m tools.protocol.freeze_protocol
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import DATASET_VERIFY_DIR, PROJECT_ROOT  # noqa: E402
from utility.utility_function.tools import code_hash as _code_hash  # noqa: E402

DRAFT, FROZEN = PROJECT_ROOT / "protocol" / "protocol_draft.json", PROJECT_ROOT / "protocol" / "frozen_protocol.json"
SEEDS = (42, 0, 1, 2, 3)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash():
    """Code and default configurations of this codebase (same hash as environment.json of every run)."""
    return _code_hash(ROOT)


def main():
    if FROZEN.exists():
        raise SystemExit("{} already exists: a frozen protocol is never overwritten".format(FROZEN))
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    frozen = dict(draft)
    frozen["datasets"] = {}
    for key in draft["datasets"]:
        ds, seed = key.split("/")
        m = json.loads((DATASET_VERIFY_DIR / ds / seed / "split_manifest.json").read_text(encoding="utf-8"))
        actual = sha(DATASET_VERIFY_DIR / ds / seed / m["sealed_test"]["file"])
        if actual != m["sealed_test"]["sha256"]:
            raise SystemExit("sealed test of {} changed since the contract was built".format(key))
        frozen["datasets"][key] = {"sealed_test_sha256": actual, "interactions_sha256": m["sha256"]["interactions.csv"]}
    frozen["runs"] = {}
    for run in draft["runs"]:
        seeds = {}
        for s in draft.get("seeds", SEEDS):
            d = PROJECT_ROOT / run / "seed{}".format(s)
            man = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
            if man.get("status") != "completed":
                raise SystemExit("{} is not completed".format(d))
            seeds[str(s)] = {"config_hash": man["config_hash"], "checkpoint_sha256": sha(d / "best_validation_model.pt"),
                             "selected_epoch": man["selected_epoch"]}
        frozen["runs"][run] = seeds
    try:
        git = lambda *a: subprocess.run(["git", *a], cwd=PROJECT_ROOT, check=True, capture_output=True,
                                        text=True).stdout.strip()
        commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    if dirty:
        raise SystemExit("tracked files differ from git commit {}: commit them before freezing".format(commit))
    plan = PROJECT_ROOT / "protocol" / "protocol_plan.md"
    frozen.update(frozen=True, frozen_at_utc=datetime.now(timezone.utc).isoformat(), code_sha256=code_hash(),
                  git_commit=commit, protocol_plan_sha256=sha(plan) if plan.is_file() else None)
    with open(FROZEN, "w", encoding="utf-8", newline="\n") as f:
        json.dump(frozen, f, indent=1, sort_keys=True); f.write("\n")
    print("frozen:", FROZEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
