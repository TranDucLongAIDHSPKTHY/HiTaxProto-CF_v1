"""Train one configuration over one or more seeds; validation only (the test split is never loaded here).

    python main.py --model LightGCN --dataset yelp2018 --study convergence --seeds 42

Outputs per seed in log/<study>/<dataset>/<model>/<setting>/seed<seed>/ (same file names as the TaxPro-CL
repository): training.log, config_resolved.json, environment.json, validation_metrics.json (every epoch, every item
group), best_validation_model.pt (checkpoint chosen by the selection rule), last_model.pt (resumable),
run_manifest.json (status, selected epoch, feasibility, hit_cap, timings). The test split is evaluated once, later,
by the sealed test evaluation.
"""
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import Parser
from config_path.config_path import PROJECT_ROOT, RESULT_DIR, model_config_file, run_dir
from utility.utility_data.data_graph import exact_propagation
from utility.utility_data.data_loader import ContractData
from utility.utility_data.sampler import make_sampler
from utility.utility_function.tools import code_hash, config_hash, read_configuration
from utility.utility_train import model_factory
from utility.utility_function.statistics import split_validation_users
from utility.utility_train.group_evaluator import ENDPOINT_GROUP, WITHIN_GROUPS, evaluate
from utility.utility_train.trainer import EarlyStopper, train

VAL_GROUPS = ("overall", "near_cold", "mid_tail", "long_tail", "warm")
VAL_KEYS = VAL_GROUPS + tuple(g + "_within" for g in WITHIN_GROUPS)   # + within-group rankings (endpoint)
VALIDATION_SPLIT_SEED = 2026                     # fixed selection / assessment halves of the validation users
logger = logging.getLogger("hitax")


def enable_determinism():
    """Bitwise-reproducible runs on one machine: deterministic kernels (index_add / scatter on CUDA otherwise use
    atomics whose summation order varies) and a fixed cuBLAS workspace. Called before the first CUDA operation."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested, gpu_id):
    if requested == "cpu" or (requested == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device("cuda:{}".format(gpu_id))


def git_commit():
    """(HEAD commit, True if tracked files differ from it) or (None, None) outside a git checkout."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=PROJECT_ROOT, check=True, capture_output=True,
                                        text=True).stdout.strip()
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return None, None


def environment(device):
    commit, dirty = git_commit()
    env = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "numpy": np.__version__, "device": str(device), "git_commit": commit,
           "git_dirty": dirty,
           "code_sha256": code_hash(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
        env["gpu_memory_bytes"] = int(torch.cuda.get_device_properties(device).total_memory)
    return env


def write_json(path, value):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(tmp, path)


def save_torch(path, payload):
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def configure_logging(log_path=None):
    """Same line layout as the TaxPro-CL repository: time | level | module | message."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h); h.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(module)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt); root.addHandler(h)


def save_per_user(path, res):
    """Per-user metrics of one evaluation (input of the paired statistics): <group>/users, <group>/<metric>."""
    arrays = {}
    for g, d in res.items():
        arrays[g + "/users"] = np.asarray(d["users"], dtype=np.int64)
        for k, v in d.items():
            if k.startswith(("Recall", "NDCG")):
                arrays["{}/{}".format(g, k)] = np.asarray(v, dtype=np.float64)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _cpu(state):
    return None if state is None else {k: v.detach().cpu() for k, v in state.items()}


def reference_for_seed(base_model, args, seed):
    """Selection threshold base of one run: None for the reference LightGCN; otherwise the validation Overall
    Recall@20 of the reference run of the SAME seed (--reference_selection, 'per_seed') or one value for every seed
    (--reference_overall, 'single')."""
    if model_factory.is_reference(base_model):
        return None, None
    if args.reference_selection is not None and args.reference_overall is not None:
        raise ValueError("give --reference_selection or --reference_overall, not both")
    if args.reference_selection is not None:
        path = Path(args.reference_selection)
        path = path if path.is_absolute() else RESULT_DIR / "selection" / path
        per_seed = json.loads(path.read_text(encoding="utf-8"))["chosen_metrics"]["per_seed"]
        if str(seed) not in per_seed:
            raise ValueError("{} has no reference run for seed {}".format(path, seed))
        return float(per_seed[str(seed)]["val_Overall"]), "per_seed"
    if args.reference_overall is None:
        raise ValueError("--reference_selection or --reference_overall (validation Overall Recall@20 of the "
                         "reference LightGCN) is required for every model other than the reference (selection rule)")
    return float(args.reference_overall), "single"


def run_seed(base_model, config, default_config, cli_overrides, seed, args, device):
    started = time.time()
    name = model_factory.display_name(base_model, config)
    setting = args.setting or model_factory.setting_name(base_model, config)
    reference, reference_mode = reference_for_seed(base_model, args, seed)
    out = run_dir(args.study, config["dataset"], name, setting, seed)
    out.mkdir(parents=True, exist_ok=True)
    h = config_hash(config)
    previous = out / "config_resolved.json"
    if previous.is_file() and config_hash(json.loads(previous.read_text(encoding="utf-8"))) != h:
        raise FileExistsError("{} holds a different configuration; choose another --setting".format(out))
    done = out / "run_manifest.json"
    if done.is_file():
        old = json.loads(done.read_text(encoding="utf-8"))
        if old.get("validation_users", "all") != args.validation_users:
            raise FileExistsError("{} was run with validation_users={}".format(out, old.get("validation_users")))
        if old.get("reference_overall") != reference:     # the selection threshold is not part of the configuration
            raise FileExistsError("{} was run with reference_overall={}".format(out, old.get("reference_overall")))
        if old.get("status") == "completed" and old.get("config_hash") == h:
            print("skip (already completed with the same configuration): {}".format(out))
            return 0
    configure_logging(out / "training.log")
    manifest_path = out / "run_manifest.json"
    manifest = {"status": "running", "model": name, "study": args.study, "setting": setting, "seed": seed,
                "dataset": config["dataset"], "config_hash": h, "reference_overall": reference,
                "reference_mode": reference_mode,
                "validation_users": args.validation_users, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "selection": "selection rule on validation (docs/GLOSSARY.md); test evaluated later, once"}
    write_json(out / "config_resolved.json", config)
    write_json(out / "environment.json", environment(device))
    write_json(manifest_path, manifest)
    try:
        torch.set_default_dtype(torch.float32)          # every real run is float32 (configure/profiles/gpu.json)
        logger.info("Start Training: model=%s dataset=%s seed=%d", name, config["dataset"], seed)
        logger.info("Training log: %s", out / "training.log")
        for key, (value, default) in sorted(cli_overrides.items()):
            logger.info("CLI configuration override: %s=%s (default=%s)", key, value, default)
        set_seed(seed)
        logger.info("Random seed: %d (enabled=True)", seed)
        data = ContractData(config["dataset"], config["split_seed"], splits=("train", "val"))
        if data.manifest.get("role") == "confirmation":
            key = "{}/{}".format(config["dataset"], config["split_seed"])
            proto = json.loads(Path(args.protocol).read_text(encoding="utf-8")) if args.protocol else {}
            if proto.get("frozen") is not True or key not in proto.get("datasets", {}):
                raise PermissionError("{} is the locked confirmation dataset: it is run only under the frozen "
                                      "protocol (--protocol protocol/frozen_protocol.json)".format(key))
        logger.info("Dataset statistics: users=%d items=%d train=%d validation=%d (test split not loaded)",
                    data.n_users, data.n_items, len(data.edges["train"]), len(data.edges["val"]))
        logger.info("Dataset Loaded: %s", data.dir)
        model, epoch_callback, extras = model_factory.build(base_model, config, data, device, seed)
        manifest.update(extras)
        for key, value in sorted(extras.items()):
            logger.info("Resolved %s=%s", key, value)
        logger.info("Model Initialized: %s on %s", name, device)
        for key in sorted(config):
            logger.info("Configuration %s=%s", key, "none" if config[key] is None else config[key])
        logger.info("Model parameters: {:,}".format(sum(p.numel() for p in model.parameters())))
        sampler = make_sampler(data.edges["train"], data.n_items, seed=seed)
        seen, truth = data.seen_for("val"), data.by_user("val")
        assessment = None
        if args.validation_users == "selection":       # pilot: choose on one half, judge once on the other
            sel, ass = split_validation_users(truth, VALIDATION_SPLIT_SEED)
            assessment = {u: truth[u] for u in sorted(ass)}
            truth = {u: truth[u] for u in sorted(sel)}
            logger.info("Validation users: selection half %d, assessment half %d (seed %d)", len(truth),
                        len(assessment), VALIDATION_SPLIT_SEED)
        deg = data.train_degree
        ks = tuple(int(k) for k in config["ks"])
        sel_k = int(config["selection_k"])
        history = []

        def val_fn(m):
            with torch.no_grad():
                eu, ei, _ = m.representations()
            res = evaluate(eu, ei, seen, truth, deg, ks=ks, groups=VAL_GROUPS, within=WITHIN_GROUPS)
            val_fn.last = {g: {**res[g]["mean"], "n_users": res[g]["n_users"]} for g in VAL_KEYS}
            return (val_fn.last[ENDPOINT_GROUP]["Recall@%d" % sel_k], val_fn.last["overall"]["Recall@%d" % sel_k])

        stopper = EarlyStopper(reference, delta_sel=float(config.get("delta_sel", 0.005)),
                               patience=int(config["patience"]))
        if reference is not None:
            logger.info("Reference Overall Recall@%d: %.8f (%s)", sel_k, reference,
                        "reference run of the same seed" if reference_mode == "per_seed" else "one value for every seed")
        opt = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
        logger.info("Optimizer Initialized: Adam lr=%s", config["lr"])
        logger.info("Validation interval: every 1 epoch(s); selection: %s",
                    "validation Overall Recall@%d" % sel_k if reference is None else
                    "validation %s Recall@%d with Overall >= %.6f" % (ENDPOINT_GROUP, sel_k, stopper.thr))
        start_epoch = 0
        if args.resume and (out / "last_model.pt").is_file():
            ck = torch.load(out / "last_model.pt", map_location="cpu")    # RNG states must stay CPU ByteTensors;
            # load_state_dict copies model and optimizer tensors to the device of the parameters
            if ck["config_hash"] != h:
                raise ValueError("resume: configuration hash differs")
            if ck.get("reference_overall") != reference:
                raise ValueError("resume: reference_overall differs ({} vs {})".format(ck.get("reference_overall"),
                                                                                     reference))
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
            sampler.rng.bit_generator.state = ck["sampler_rng"]
            stopper.best_feas, stopper.best_over, stopper.bad = ck["stopper"]
            torch.set_rng_state(ck["torch_rng"]); np.random.set_state(ck["numpy_rng"]); random.setstate(ck["py_rng"])
            if device.type == "cuda" and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
            if (out / "validation_metrics.json").is_file():
                history = json.loads((out / "validation_metrics.json").read_text(encoding="utf-8"))
                history = [r for r in history if r["epoch"] <= ck["epoch"] + 1]
            start_epoch = ck["epoch"] + 1
            logger.info("Resumed training from %s at epoch %d", out / "last_model.pt", start_epoch + 1)

        def checkpoint(ep, optimizer):
            bf, bo = stopper.best_feas, stopper.best_over
            save_torch(out / "last_model.pt", {
                "epoch": ep, "config_hash": h, "reference_overall": reference, "model": _cpu(model.state_dict()),
                "optimizer": optimizer.state_dict(), "sampler_rng": sampler.rng.bit_generator.state,
                "stopper": ((bf[0], bf[1], _cpu(bf[2])), (bo[0], bo[1], _cpu(bo[2])), stopper.bad),
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "py_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None})

        def on_epoch_end(ep, optimizer, rec):
            v = val_fn.last
            history.append({"epoch": ep + 1, "loss": rec["loss"], "loss_terms": rec["loss_terms"],
                            "training_time": rec["train_s"], "validation_time": rec["val_s"], "metrics": v})
            write_json(out / "validation_metrics.json", history)
            state, sel_epoch, feasible = stopper.result()
            if sel_epoch == ep:
                value = stopper.best_over[0] if (reference is None or not feasible) else stopper.best_feas[0]
                logger.info("Best Validation Updated: epoch=%d %s=%.8f", ep + 1,
                            "recall@%d_overall" % sel_k if (reference is None or not feasible)
                            else "recall@%d_%s" % (sel_k, ENDPOINT_GROUP), value)
                save_torch(out / "best_validation_model.pt", {"model": _cpu(state), "epoch": ep + 1,
                                                              "feasible": feasible, "config_hash": h})
                logger.info("Saving Checkpoint: best_validation_model=%s", out / "best_validation_model.pt")
            if rec["stop"]:
                logger.info("Early stopping condition reached: best_epoch=%d patience=%d", sel_epoch + 1,
                            stopper.patience)
            if rec["stop"] or (ep + 1) % args.checkpoint_every == 0 or ep == int(config["epochs"]) - 1:
                checkpoint(ep, optimizer)
                logger.info("Saving Checkpoint: last_model=%s", out / "last_model.pt")
            o = v["overall"]
            logger.info("Validation Metrics: epoch=%d recall@%s=%s ndcg@%s=%s", ep + 1, list(ks),
                        [round(o["Recall@%d" % k], 8) for k in ks], list(ks), [round(o["NDCG@%d" % k], 8) for k in ks])
            logger.info("Validation Groups: epoch=%d recall@%d %s", ep + 1, sel_k,
                        " ".join("%s=%.8f" % (g, v[g]["Recall@%d" % sel_k]) for g in VAL_KEYS[1:]))

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, info = train(model, sampler, len(data.edges["train"]), int(config["epochs"]),
                            int(config["batch_size"]), reg=float(config["reg"]), val_fn=val_fn, stopper=stopper,
                            optimizer=opt, start_epoch=start_epoch, on_epoch_end=on_epoch_end,
                            epoch_callback=epoch_callback)
        state, _, _ = stopper.result()                   # always rewrite the selected checkpoint (also after resume)
        sel_row = next(r for r in history if r["epoch"] == info["epoch"] + 1)
        manifest["selected_validation_training"] = sel_row["metrics"]      # as measured during training (cuSPARSE)
        model.eval()
        with torch.no_grad(), exact_propagation(model):    # final numbers: bitwise reproducible from the checkpoint
            eu, ei, _ = model.representations()
        res_v = evaluate(eu, ei, seen, truth, deg, ks=ks, groups=VAL_GROUPS, within=WITHIN_GROUPS)
        save_per_user(out / "validation_per_user.npz", res_v)
        manifest["selected_validation"] = {g: {**res_v[g]["mean"], "n_users": res_v[g]["n_users"]} for g in VAL_KEYS}
        if assessment is not None:                       # evaluated ONCE, on the selected checkpoint
            res_a = evaluate(eu, ei, seen, assessment, deg, ks=ks, groups=VAL_GROUPS, within=WITHIN_GROUPS)
            save_per_user(out / "assessment_per_user.npz", res_a)
            manifest["assessment"] = {g: {**res_a[g]["mean"], "n_users": res_a[g]["n_users"]} for g in VAL_KEYS}
            logger.info("Assessment half (evaluated once): recall@%d %s", sel_k, " ".join(
                "%s=%.8f" % (g, res_a[g]["mean"]["Recall@%d" % sel_k]) for g in VAL_KEYS))
        save_torch(out / "best_validation_model.pt", {"model": _cpu(state), "epoch": info["epoch"] + 1,
                                                      "feasible": info["feasible"], "config_hash": h})
        sel = stopper.best_over if (reference is None or not info["feasible"]) else stopper.best_feas
        logger.info("Training loop completed: best_epoch=%s selection_value=%.8f feasible=%s hit_cap=%s",
                    None if info["epoch"] is None else info["epoch"] + 1, sel[0], info["feasible"], info["hit_cap"])
        logger.info("Test evaluation deferred: sealed test protocol (the test split is evaluated once, later)")
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        duration_seconds=round(time.time() - started, 1),
                        selected_epoch=None if info["epoch"] is None else info["epoch"] + 1,
                        feasible=info["feasible"], completed_epochs=info["last_epoch"] + 1, hit_cap=info["hit_cap"],
                        stopped_early=info["stopped_early"], selection_value=sel[0],
                        peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda" else None)
        write_json(manifest_path, manifest)
        logger.info("Training Finished")
        return 0
    except Exception:
        logger.exception("Training failed; seed=%d", seed)
        manifest.update(status="failed", failed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(manifest_path, manifest)
        return 1
    finally:
        logger.info("Seed %d Training Time: %.3f seconds", seed, time.time() - started)


def main(argv=None):
    base_model = Parser.parse_model(argv)
    default_config = read_configuration(model_config_file(base_model))
    args = Parser.parse_args(default_config, argv)
    cli = Parser.overrides(default_config, args)
    config = Parser.apply_overrides(default_config, args)
    enable_determinism()
    device = resolve_device(args.device, args.gpu_id)
    failed = [s for s in args.seeds if run_seed(base_model, config, default_config, cli, s, args, device) != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
