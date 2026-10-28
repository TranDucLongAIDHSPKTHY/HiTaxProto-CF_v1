"""Record the SHA-256 of every sealed test file (and the role of every dataset) in protocol/sealed_test_hashes.json.

Run after building the contracts and before any pilot run; the file is committed so that any later change of a sealed
test split is detectable. It does not unseal anything.

    python -m tools.protocol.record_locks
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_path.config_path import DATASET_VERIFY_DIR, PROJECT_ROOT  # noqa: E402
from utility.utility_data.data_loader import _sha256  # noqa: E402


def main():
    out = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    for manifest_path in sorted(DATASET_VERIFY_DIR.glob("*/*/split_manifest.json")):
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        d = manifest_path.parent
        sealed = d / m["sealed_test"]["file"]
        actual = _sha256(sealed)
        if actual != m["sealed_test"]["sha256"]:
            raise SystemExit("sealed test of {} differs from its manifest".format(d))
        out["datasets"]["{}/{}".format(d.parent.name, d.name)] = {
            "role": m.get("role"), "sealed_test_sha256": actual, "interactions_sha256": m["sha256"]["interactions.csv"],
            "test_rows": m["counts"]["test"]}
    target = PROJECT_ROOT / "protocol" / "sealed_test_hashes.json"
    target.parent.mkdir(exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2, sort_keys=True); f.write("\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
