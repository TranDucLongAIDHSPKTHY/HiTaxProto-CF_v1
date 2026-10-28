"""Independent audit of the protocol.

1. Statistics vs brute-force definitions: sign-flip p-value vs exact enumeration of all 2^n sign vectors (small n);
   Holm vs its step-down definition; non-inferiority on D' vs the ratio statement; percentile bootstrap coverage of
   the two-level interval on simulated seed x user data with a known difference.
2. Locks on the REAL contracts: the test split of every contract is refused without a frozen protocol, interactions.csv
   holds no test row, sealed hashes equal protocol/sealed_test_hashes.json, the confirmation dataset is refused by the
   runner, the validation halves are disjoint, cover all validation users and are reproducible.

    python -m tools.protocol.audit_protocol
"""
import collections
import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as runner  # noqa: E402
from config_path.config_path import DATASET_VERIFY_DIR, PROJECT_ROOT  # noqa: E402
from utility.utility_data.data_loader import ContractData, SealedTestError, _sha256  # noqa: E402
from utility.utility_function.statistics import (holm, noninferiority, signflip_p, split_validation_users,  # noqa
                                                 two_level_bootstrap)

RESULTS = collections.OrderedDict()


def check(name, ok, detail=""):
    RESULTS.setdefault(name, []).append((bool(ok), detail))


def statistics_checks():
    rng = np.random.default_rng(11)
    for n in (6, 9, 12):
        for _ in range(3):
            D = rng.normal(0.3, 1.0, n)
            obs = abs(D.mean())
            exact = np.mean([abs((np.array(s) * D).mean()) >= obs - 1e-15
                             for s in itertools.product((-1.0, 1.0), repeat=n)])
            p = signflip_p(D, B=200_000, seed=1)
            check("signflip_vs_exact", abs(p - exact) < 0.01, "n={} p={:.4f} exact={:.4f}".format(n, p, exact))
    for _ in range(20):
        p = rng.uniform(0, 0.2, 9)
        order = np.argsort(p); m = len(p); ref = np.empty(m); run = 0.0
        for k, i in enumerate(order):
            run = max(run, min(1.0, (m - k) * p[i])); ref[i] = run
        check("holm_definition", np.allclose(holm(p), ref))
    base = rng.uniform(0.05, 0.15, (5, 2000))
    for factor, should_pass in ((1.0, True), (0.995, True), (0.97, False)):
        r = noninferiority(base * factor, base, delta=0.01, n_boot=1000)
        check("noninferiority", r["pass"] == should_pass, "factor {}".format(factor))
    cover, trials = 0, 300                                   # standard error of the coverage estimate ~0.013
    for t in range(trials):
        seed_eff = rng.normal(0, 0.003, (5, 1))
        user = rng.uniform(0, 0.2, (1, 800))
        A = user + 0.01 + seed_eff + rng.normal(0, 0.05, (5, 800))
        B = user + rng.normal(0, 0.003, (5, 1)) + rng.normal(0, 0.05, (5, 800))
        ci = two_level_bootstrap(A, B, n_boot=400, seed=t)["ci_abs"]
        cover += ci[0] <= 0.01 <= ci[1]
    check("bootstrap_coverage", cover / trials >= 0.90, "coverage {:.3f} of nominal 0.95".format(cover / trials))


def lock_checks():
    record = json.loads((PROJECT_ROOT / "protocol" / "sealed_test_hashes.json").read_text(encoding="utf-8"))
    for manifest in sorted(DATASET_VERIFY_DIR.glob("*/*/split_manifest.json")):
        ds, seed = manifest.parent.parent.name, int(manifest.parent.name)
        key = "{}/{}".format(ds, seed)
        m = json.loads(manifest.read_text(encoding="utf-8"))
        try:
            ContractData(ds, seed, splits=("train", "val", "test"))
            check("test_locked", False, key)
        except SealedTestError:
            check("test_locked", True, key)
        with open(manifest.parent / "interactions.csv", encoding="utf-8") as f:
            check("no_test_rows", not any(line.rstrip().endswith(",test") for line in f), key)
        actual = _sha256(manifest.parent / m["sealed_test"]["file"])
        check("sealed_hash_recorded", record["datasets"].get(key, {}).get("sealed_test_sha256") == actual, key)
        if m.get("role") == "confirmation":                 # the refused run writes into a temporary folder only
            original = runner.run_dir
            with tempfile.TemporaryDirectory() as tmp:
                runner.run_dir = lambda st, d, mo, se, s: Path(tmp) / st / d / mo / se / "seed{}".format(s)
                try:
                    rc = runner.main(["--model", "LightGCN", "--dataset", ds, "--split_seed", str(seed), "--epochs",
                                      "1", "--device", "cpu", "--study", "dev", "--setting", "audit-lock"])
                finally:
                    runner.run_dir = original
                    runner.configure_logging(None)          # close training.log before the folder is removed
            check("confirmation_locked", rc == 1, key)
        data = ContractData(ds, seed)
        users = set(data.by_user("val"))
        s1, a1 = split_validation_users(users, runner.VALIDATION_SPLIT_SEED)
        s2, a2 = split_validation_users(users, runner.VALIDATION_SPLIT_SEED)
        check("validation_halves", not (s1 & a1) and (s1 | a1) == users and s1 == s2 and a1 == a2 and
              abs(len(s1) - len(a1)) <= 1, key)


def main():
    statistics_checks()
    lock_checks()
    report = {k: {"checks": len(v), "failures": sum(not ok for ok, _ in v),
                  "examples": [d for ok, d in v if not ok][:3], "details": [d for _, d in v][:6]}
              for k, v in RESULTS.items()}
    out = PROJECT_ROOT / "preprocessed" / "protocol_checks" / "audit_protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1); f.write("\n")
    for k, v in report.items():
        print("{:22s} {:3d}/{:3d} {}".format(k, v["checks"] - v["failures"], v["checks"], v["examples"] or v["details"][:2]))
    return 0 if all(v["failures"] == 0 for v in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
