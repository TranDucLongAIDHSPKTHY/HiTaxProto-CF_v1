"""Training loop, early stopping / checkpoint selection, and the configuration selection rule.

Selection rule (docs/GLOSSARY.md, "selection rule"): among configurations whose validation Overall Recall@20 is at
least (1 - delta_sel) times that of the reference LightGCN, choose the one with the highest validation long-tail
Recall@20; delta_sel = 0.5%, tighter than the 1% non-inferiority margin tested on the test split.
"""
import logging
import time

import torch

logger = logging.getLogger("hitax")


class EarlyStopper:
    """Checkpoint selection and early stopping consistent with the selection rule.

    Keeps TWO checkpoints: (1) best FEASIBLE epoch = max validation long-tail Recall@20 among epochs whose validation
    Overall Recall@20 >= (1 - delta_sel) * reference; (2) best OVERALL epoch (declared fallback).
    Patience counts consecutive epochs in which NEITHER checkpoint improved — so a run that has not yet reached the
    Overall threshold keeps training as long as its Overall still improves. The reference LightGCN itself
    (reference_overall=None) is selected on validation Overall only.
    result() -> (state, epoch, feasible): the feasible checkpoint if any, else the best-Overall checkpoint with
    feasible=False (reported as "infeasible")."""

    def __init__(self, reference_overall=None, delta_sel=0.005, patience=20):
        self.thr = None if reference_overall is None else (1 - delta_sel) * reference_overall
        self.patience = patience
        self.best_feas = (-float("inf"), None, None)      # (val long-tail, epoch, state)
        self.best_over = (-float("inf"), None, None)      # (val Overall, epoch, state)
        self.bad = 0

    def step(self, epoch, val_lt, val_overall, state_fn):
        """Call once per epoch; state_fn() returns a detached copy of the model state. Returns True to stop."""
        improved = False
        if val_overall == val_overall and val_overall > self.best_over[0]:      # NaN-safe
            self.best_over = (val_overall, epoch, state_fn()); improved = True
        if self.thr is not None and val_overall == val_overall and val_overall >= self.thr \
                and val_lt == val_lt and val_lt > self.best_feas[0]:
            self.best_feas = (val_lt, epoch, state_fn()); improved = True
        self.bad = 0 if improved else self.bad + 1
        return self.bad >= self.patience

    def result(self):
        if self.thr is None:                                # reference LightGCN: Overall-selected
            return self.best_over[2], self.best_over[1], True
        if self.best_feas[2] is not None:
            return self.best_feas[2], self.best_feas[1], True
        return self.best_over[2], self.best_over[1], False


def _fmt(seconds):
    s = max(0, int(round(seconds)))
    return "{:02d}:{:02d}:{:02d}".format(s // 3600, s % 3600 // 60, s % 60)


def train(model, sampler, n_edges, epochs, batch_size, lr=1e-3, reg=1e-4, val_fn=None, stopper=None, log=None,
          epoch_callback=None, with_replacement=False, optimizer=None, start_epoch=0, on_epoch_end=None):
    """Adam + BPR. val_fn(model) -> (val long-tail Recall@20, val Overall Recall@20) on VALIDATION (never test).
    stopper: REQUIRED when val_fn is given — EarlyStopper(reference_overall=None) for the reference LightGCN,
    otherwise EarlyStopper(reference_overall=<its validation Overall>) (no silent Overall-only default).
    In the pilot, val_fn must use the SELECTION half of the validation users only.
    epoch_callback(model, epoch) runs at the START of every epoch (K-means source: cluster assignments; NCL: E-step).
    Each epoch is one pass over a permutation of the train edges unless with_replacement=True or the sampler has no
    .epoch() method (then n_edges // batch_size draws with replacement).
    `optimizer` and `start_epoch` resume a checkpointed run; on_epoch_end(epoch, opt, record) receives
    {epoch, loss, loss_terms, train_s, val_s, stop} after every epoch (logging, checkpoints).
    Log lines follow the TaxPro-CL repository: Start Epoch / Train Loss / End Epoch / Epoch Progress.
    Returns (model loaded with the selected checkpoint, info dict)."""
    log = log or logger.info
    opt = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=lr)
    if val_fn is not None and stopper is None:
        raise ValueError("pass an EarlyStopper explicitly (reference_overall=None only for the reference LightGCN)")
    device = next(model.parameters()).device
    steps = max(1, n_edges // batch_size)
    snap = lambda: {k: t.detach().clone() for k, t in model.state_dict().items()}
    terms_fn = getattr(model, "loss_terms", None)
    stopped, last, loop_start = False, start_epoch - 1, time.time()
    if stopper is not None and start_epoch > 0 and stopper.bad >= stopper.patience:
        stopped = True                                  # resumed after the stopping epoch: no extra epoch
        log("Early stopping condition already reached before epoch %d (resumed run)" % (start_epoch + 1))
    for ep in range(start_epoch, start_epoch if stopped else epochs):
        t0 = time.time()
        log("Start Epoch: %d" % (ep + 1))
        if epoch_callback is not None:
            epoch_callback(model, ep)
        model.train()
        if with_replacement or not hasattr(sampler, "epoch"):
            batches = (sampler(batch_size) for _ in range(steps))
        else:
            batches = sampler.epoch(batch_size)
        totals, n_batches = None, 0
        for u, p, n in batches:
            u, p, n = u.to(device), p.to(device), n.to(device)
            parts = terms_fn(u, p, n, reg=reg) if terms_fn is not None else (model.bpr_loss(u, p, n, reg=reg),)
            loss = sum(parts)
            opt.zero_grad(); loss.backward(); opt.step()
            vals = torch.stack([x.detach() for x in parts])
            totals = vals if totals is None else totals + vals
            n_batches += 1
        terms = [float(x) / max(1, n_batches) for x in totals]
        mean_loss = sum(terms)
        t1 = time.time()
        log("Train Loss: epoch=%d loss=%.6f = %s" % (ep + 1, mean_loss, " + ".join("%.6g" % x for x in terms)))
        log("End Epoch: %d training_time=%.3f" % (ep + 1, t1 - t0))
        if val_fn is not None:
            model.eval()
            v_lt, v_all = val_fn(model)
            stopped = stopper.step(ep, v_lt, v_all, snap)
        last = ep
        if on_epoch_end is not None:
            on_epoch_end(ep, opt, {"epoch": ep, "loss": mean_loss, "loss_terms": terms, "train_s": t1 - t0,
                                   "val_s": time.time() - t1, "stop": stopped})
        elapsed = time.time() - loop_start
        done = ep + 1 - start_epoch
        log("Epoch Progress: %d/%d elapsed=%s ETA=%s" % (ep + 1, epochs, _fmt(elapsed),
                                                          _fmt(elapsed / done * (epochs - ep - 1))))
        if stopped:
            break
    info = {"epoch": None, "feasible": None, "last_epoch": last, "stopped_early": stopped,
            "hit_cap": (not stopped) and last == epochs - 1}
    if val_fn is not None:
        state, ep_sel, feas = stopper.result()
        if state is not None:
            model.load_state_dict(state)
        info.update(epoch=ep_sel, feasible=feas)
        if not feas:
            log("WARNING: no feasible epoch; best-Overall checkpoint kept (report as infeasible)")
    return model, info


def select_config(candidates, reference_overall, delta_sel=0.005):
    """Selection rule: maximise validation long-tail Recall@20 subject to validation Overall Recall@20 >=
    (1 - delta_sel) * reference. The reference is the LightGCN configuration with the highest validation Overall
    Recall@20 (mean over tuning seeds). candidates: dicts with 'name', 'val_LT', 'val_Overall' (means over tuning
    seeds). A configuration is feasible iff the MEAN over tuning seeds of validation Overall at the selected
    checkpoints is >= (1 - delta_sel) * reference; infeasible seed checkpoints enter that mean unchanged.
    Returns (chosen, feasible). If no candidate is feasible, the declared fallback is the candidate with the highest
    validation Overall, reported with feasible = False (delta_sel is never relaxed)."""
    ok = [c for c in candidates if c["val_Overall"] >= (1 - delta_sel) * reference_overall]
    if ok:
        return max(ok, key=lambda c: c["val_LT"]), True
    return max(candidates, key=lambda c: c["val_Overall"]), False
