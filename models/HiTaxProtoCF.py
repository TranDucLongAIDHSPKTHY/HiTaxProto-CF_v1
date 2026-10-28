"""HiTaxProto-CF: leave-one-out taxonomy prototypes fused into the scoring representation of LightGCN.

    e*_i = g_i * e_i + (1 - g_i) * p_i,   p_i = stop-gradient( leave-one-out prototype of i )

Gate types (see docs/GLOSSARY.md):
    none            LightGCN, no fusion
    global          GlobalGate          g_i = sigmoid(b), one learned weight shared by every item
    degree          DegreeGate          g_i = sigmoid(a * log(1 + d_i) + b), a = softplus(.) >= 0 (monotone in degree)
    degree_content  DegreeContentGate   DegreeGate + w_e^T e_i + w_p^T p_i
GlobalGate is DegreeGate with a = 0, so DegreeGate - GlobalGate isolates the degree term.
AncestorPrototypeMemory builds prototypes over multi-level ancestor sets (equal weights or attention).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GATE_TYPES = ("none", "global", "degree", "degree_content")
DISPLAY_NAMES = {"none": "LightGCN", "global": "GlobalGate", "degree": "DegreeGate",
                 "degree_content": "DegreeContentGate"}


class PrototypeMemory(nn.Module):
    """Leave-one-out prototypes from FINAL item embeddings of training items (stop-gradient).

    pairs: LongTensor [P, 2] of (item, node) — direct labels Λ(i), or a kept ancestor set A(i).
    is_train_item: BoolTensor [I] (train degree >= 1). Only training items are members of a node.
    leave_one_out=False (the self-inclusive variant) keeps the item itself in its own prototype.
    """

    def __init__(self, n_items, n_nodes, pairs, is_train_item, leave_one_out=True):
        super().__init__()
        self.n_items, self.n_nodes, self.leave_one_out = n_items, n_nodes, leave_one_out
        pairs = torch.unique(pairs[:, :2], dim=0)               # each item counted ONCE per node (duplicate rows
        assert len(torch.unique(pairs, dim=0)) == len(pairs)    # from several paths/releases would leak e_i into p_i)
        self.register_buffer("pair_item", pairs[:, 0].clone())
        self.register_buffer("pair_node", pairs[:, 1].clone())
        member = is_train_item[pairs[:, 0]]
        self.register_buffer("pair_member", member.clone())
        cnt = torch.zeros(n_nodes).scatter_add_(0, pairs[:, 1], member.to(torch.get_default_dtype()))
        self.register_buffer("node_count", cnt)

    @torch.no_grad()
    def forward(self, item_emb):
        e = item_emb.detach()                                   # stop-gradient
        d = e.shape[1]
        m = self.pair_member.to(e.dtype).unsqueeze(1)           # [P,1]
        S = torch.zeros(self.n_nodes, d, dtype=e.dtype, device=e.device)
        S.index_add_(0, self.pair_node, e[self.pair_item] * m)  # S_n = sum of members (each item once per node)
        if self.leave_one_out:
            num = S[self.pair_node] - e[self.pair_item] * m     # leave-one-out numerator
            den = self.node_count[self.pair_node].to(e.dtype) - m.squeeze(1)
        else:
            num = S[self.pair_node]
            den = self.node_count[self.pair_node].to(e.dtype)
        valid = den > 0                                         # singleton under leave-one-out -> undefined
        p_pair = num[valid] / den[valid].unsqueeze(1)
        items_v = self.pair_item[valid]
        P = torch.zeros(self.n_items, d, dtype=e.dtype, device=e.device).index_add_(0, items_v, p_pair)
        k = torch.zeros(self.n_items, dtype=e.dtype, device=e.device).index_add_(
            0, items_v, torch.ones_like(items_v, dtype=e.dtype))
        has = k > 0
        P[has] = P[has] / k[has].unsqueeze(1)                   # equal-weight mean over defined nodes
        return P, has


class Gate(nn.Module):
    """mode: 'global' (one learned b), 'degree' (a, b), 'degree_content' (a, b, w_e, w_p),
    'fixed' (frozen alpha; worked example only)."""

    def __init__(self, mode, dim, alpha=0.5, a_init=1.0, b_init=0.0):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        if mode in ("global", "degree", "degree_content"):
            self.b = nn.Parameter(torch.tensor(float(b_init)))
        if mode in ("degree", "degree_content"):
            # a = softplus(a_raw) >= 0 guarantees monotonicity in degree; softplus^-1 needs a_init > 0
            a_init = max(float(a_init), 1e-6)
            self.a_raw = nn.Parameter(torch.tensor(math.log(math.expm1(a_init))))
        if mode == "degree_content":
            self.w_e = nn.Parameter(torch.zeros(dim))
            self.w_p = nn.Parameter(torch.zeros(dim))

    def forward(self, log1p_deg, e, p):
        if self.mode == "fixed":
            return torch.full_like(log1p_deg, self.alpha)
        if self.mode == "global":
            return torch.sigmoid(self.b).expand_as(log1p_deg)
        z = F.softplus(self.a_raw) * log1p_deg + self.b
        if self.mode == "degree_content":
            z = z + e @ self.w_e + p @ self.w_p
        return torch.sigmoid(z)


class HiTaxProtoCF(nn.Module):
    """gate_type in GATE_TYPES. The prototype source (taxonomy, shuffled, global, kmeans) is decided by the
    (item, node) pairs given to the memory; every source fuses exactly the items of the shared `fuse_mask`."""

    def __init__(self, backbone, gate_type, item_degree, memory=None, gate=None, fuse_mask=None):
        super().__init__()
        if gate_type not in GATE_TYPES:
            raise ValueError("gate_type must be one of {}".format(GATE_TYPES))
        self.backbone, self.gate_type = backbone, gate_type
        self.register_buffer("log1p_deg", torch.log1p(item_degree.to(torch.get_default_dtype())))
        self.memory, self.gate = memory, gate
        # fuse_mask: BoolTensor [I] shared by all sources (items with a defined taxonomy prototype); None = own `has`
        self.register_buffer("fuse_mask", None if fuse_mask is None else fuse_mask.clone(), persistent=False)

    def representations(self):
        eu, ei = self.backbone()
        if self.gate_type == "none":
            return eu, ei, None
        P, has = self.memory(ei)
        act = has if self.fuse_mask is None else (has & self.fuse_mask)
        if self.fuse_mask is not None and not bool(has[self.fuse_mask].all()):
            raise RuntimeError("prototype source lacks a prototype for %d masked items — check label resolution, "
                               "shuffle strata or the K-means minimum cluster size" % int((self.fuse_mask & ~has).sum()))
        g = self.gate(self.log1p_deg[act], ei[act], P[act])
        ei_star = ei.clone()
        ei_star[act] = g.unsqueeze(1) * ei[act] + (1 - g).unsqueeze(1) * P[act]
        return eu, ei_star, (g, act)

    def loss_terms(self, users, pos, neg, reg=1e-4):
        """(BPR, weighted L2[, backbone term]) for one batch; the objective is their sum. No taxonomy-defined loss;
        a contrastive backbone (SimGCL) adds its own loss through backbone.extra_loss."""
        eu, ei, _ = self.representations()
        x = (eu[users] * ei[pos]).sum(1) - (eu[users] * ei[neg]).sum(1)
        terms = (-F.logsigmoid(x).mean(), reg * self.backbone.reg_terms(users, pos, neg))
        if hasattr(self.backbone, "extra_loss"):
            terms = terms + (self.backbone.extra_loss(users, pos),)
        return terms

    def bpr_loss(self, users, pos, neg, reg=1e-4):
        return sum(self.loss_terms(users, pos, neg, reg))


def make_fuse_mask(tax_has, item_degree):
    """Shared fuse mask of every prototype source: items with a defined taxonomy prototype AND train degree >= 1."""
    return tax_has.bool() & (item_degree >= 1)


class AncestorPrototypeMemory(nn.Module):
    """Prototypes over the kept ancestor set A(i) (multi-level taxonomy).

    pairs: LongTensor [P, 3] of (item, node, hop) from ancestor_sets.csv (kept == True).
    variant 'mean': equal weights over defined nodes (equals PrototypeMemory on the same pairs).
    variant 'attention': w_{i,n} = softmax_n( q^T p_n^(-i) + gamma[hop] ), hop capped at H.
    Leave-one-out prototypes are stop-gradient; q and gamma are trainable.
    """

    def __init__(self, n_items, n_nodes, pairs, is_train_item, dim, variant="mean", H=3):
        super().__init__()
        # deduplicate (item, node), keeping the SHORTEST hop
        key = pairs[:, 0] * n_nodes + pairs[:, 1]
        order = torch.argsort(key * (H + 10_000) + pairs[:, 2])
        pairs = pairs[order]; key = key[order]
        first = torch.ones(len(pairs), dtype=torch.bool); first[1:] = key[1:] != key[:-1]
        pairs = pairs[first]
        self.base = PrototypeMemory(n_items, n_nodes, pairs[:, :2], is_train_item)
        self.register_buffer("hop", pairs[:, 2].clamp(max=H))
        if variant not in ("mean", "attention"):
            raise ValueError("variant must be 'mean' or 'attention'")
        self.variant = variant
        if variant == "attention":
            self.q = nn.Parameter(torch.zeros(dim))
            self.gamma = nn.Parameter(torch.zeros(H + 1))

    def forward(self, item_emb):
        b = self.base
        with torch.no_grad():
            e = item_emb.detach(); d = e.shape[1]
            m = b.pair_member.to(e.dtype).unsqueeze(1)
            S = torch.zeros(b.n_nodes, d, dtype=e.dtype, device=e.device).index_add_(0, b.pair_node, e[b.pair_item] * m)
            num = S[b.pair_node] - e[b.pair_item] * m
            den = b.node_count[b.pair_node].to(e.dtype) - m.squeeze(1)
            valid = den > 0
            p_pair = num[valid] / den[valid].unsqueeze(1)
        items_v, hop_v = b.pair_item[valid], self.hop[valid]
        if self.variant == "mean":
            s = torch.zeros(len(items_v), dtype=e.dtype, device=e.device)
        else:
            s = p_pair @ self.q + self.gamma[hop_v]
        # softmax within each item (scatter): subtract per-item max for stability
        mx = torch.full((b.n_items,), -float("inf"), dtype=e.dtype, device=e.device).scatter_reduce(
            0, items_v, s.detach(), "amax")                    # constant shift: detach (softmax is shift-invariant)
        w = torch.exp(s - mx[items_v])
        Z = torch.zeros(b.n_items, dtype=e.dtype, device=e.device).index_add(0, items_v, w)
        w = w / Z[items_v]
        P = torch.zeros(b.n_items, d, dtype=e.dtype, device=e.device).index_add(0, items_v, w.unsqueeze(1) * p_pair)
        has = Z > 0
        return P, has
