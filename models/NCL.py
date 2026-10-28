"""NCL and NCL+Taxonomy re-implemented on the LightGCN backbone of this codebase.

NCL: Z. Lin, C. Tian, Y. Hou, W. X. Zhao, "Improving graph collaborative filtering with neighborhood-enriched
contrastive learning," WWW 2022 (arXiv 2202.06200). Reference implementation: github.com/RUCAIBox/NCL
(ncl.py, trainer.py, properties/*.yaml). Both variants share EVERY component; they differ ONLY in the item-side
grouping of the prototype loss, so NCL - NCL+Taxonomy is not confounded by implementation.

    L = L_BPR + lam_s * (L_S^U + alpha * L_S^I) + lam_p * (L_P^U + L_P^I) + reg * L_reg       (as in the repo code:
        alpha weights ONLY the item STRUCTURE term; the prototype term is unweighted)

Reduction: the repo SUMS the InfoNCE terms over the batch (-log(...).sum()) while BPR is a batch mean. Here every
InfoNCE term is (batch_ref / B) * sum = batch_ref * mean, batch_ref = 4096 (repo train_batch_size): identical to the
repo at B = 4096; at any other B it has the same EXPECTED objective and the same InfoNCE : BPR ratio as the repo at
B = 4096 (negatives are all nodes, not in-batch), so lam_s, lam_p stay in REPO units.
L_S (structure neighbours): positive (z_v^(2h), z_v^(0)), negatives z_w^(0) for ALL nodes w of the same type;
     z = L2-normalised; temperature tau; h = hyper_layers = 1.
L_P (semantic neighbours): normalised layer-0 (ego) embedding vs L2-normalised centroids; positive = own group.
     users: K-means groups (both variants).  items: K-means groups (NCL) or valid taxonomy nodes (NCL+Taxonomy).
     Multi-label item (NCL+Taxonomy): MEAN over its valid labels of the single-positive InfoNCE (candidates = all valid
     nodes); items without a valid label contribute 0 (excluded). Empty clusters / nodes without train members are
     masked out of the softmax denominator.
E-step: every epoch (repo m_step = 1), at the START of the epoch, from RAW ego embeddings; constant during the epoch.
     One k for users and items (repo num_clusters). Here: torch Lloyd, n_init=1, iters=25 (faiss default niter = 25).
Warm-up: the prototype loss is OFF for epochs < warm_up (repo warm_up_step = 20); L_S is on from epoch 0.
Group mean INCLUDES the item itself (as an NCL centroid; leave-one-out belongs to HiTaxProto-CF, not to NCL).
Declared deviations from the repo: (i) L_reg follows the backbone convention of this codebase; (ii) K-means is torch Lloyd +
k-means++ instead of faiss, centroids are the means of the FINAL assignment, item K-means runs on TRAIN items only,
empty clusters are masked out of the softmax; (iii) embeddings are initialised as in this backbone, N(0, 0.1^2);
(iv) data, split, evaluation and early stopping are those of this codebase.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.utility_data.controls import kmeans_labels
from utility.utility_data.data_graph import spmm

REPO_DEFAULTS = {          # RucAIBox/NCL properties/*.yaml (read 25/10/2026)
    "NCL.yaml":          dict(tau=0.1,  lam_s=1e-7, alpha=1.0, lam_p=8e-8, k=1000, reg=1e-4, hyper=1, warm_up=20),
    "yelp.yaml":         dict(tau=0.05, lam_s=1e-6, alpha=1.5, lam_p=1e-7, k=2000, reg=1e-4),
    "amazon-books.yaml": dict(tau=0.05, lam_s=1e-6, alpha=0.8, lam_p=1e-7, k=2000, reg=1e-6),
}


class NCL(nn.Module):
    def __init__(self, backbone, item_groups="kmeans", item_label_pairs=None, n_item_nodes=None, is_train_item=None,
                 k=1000, lam_s=1e-7, lam_p=8e-8, alpha=1.0, tau=0.1, hyper=1, warm_up=20, batch_ref=4096,
                 kmeans_init=1, kmeans_iters=25):
        super().__init__()
        assert item_groups in ("kmeans", "taxonomy")
        self.bb, self.item_groups = backbone, item_groups
        self.lam_s, self.lam_p, self.alpha, self.tau, self.hyper = lam_s, lam_p, alpha, tau, hyper
        self.warm_up, self.batch_ref = warm_up, batch_ref
        self.kmeans_init, self.kmeans_iters = kmeans_init, kmeans_iters
        nu, ni = backbone.n_users, backbone.n_items
        d = backbone.emb.weight.shape[1]
        dev = backbone.emb.weight.device
        self.k_users = min(k, nu)
        tr = (is_train_item if is_train_item is not None else torch.ones(ni, dtype=torch.bool)).to(dev)
        self.register_buffer("is_train_item", tr.clone())
        if item_groups == "taxonomy":
            pairs = torch.unique(item_label_pairs[:, :2].to(dev), dim=0)
            self.n_item_groups = int(n_item_nodes)
            # padded positive lists [n_items, max_labels] (-1 = padding), precomputed once
            cnt = torch.bincount(pairs[:, 0], minlength=ni)
            width = max(1, int(cnt.max()) if len(pairs) else 1)
            pos = torch.full((ni, width), -1, dtype=torch.long, device=dev)
            order = torch.argsort(pairs[:, 0], stable=True)
            it, nd = pairs[order, 0], pairs[order, 1]
            start = torch.cumsum(cnt, 0) - cnt
            slot = torch.arange(len(it), device=dev) - start[it]
            pos[it, slot] = nd
            self.register_buffer("item_pos", pos)
            self.register_buffer("lp_item", it.clone())
            self.register_buffer("lp_node", nd.clone())
            # every node used as a label must have a TRAIN member (run label resolution first)
            has_tr = torch.zeros(self.n_item_groups, dtype=torch.bool, device=dev)
            has_tr[nd[tr[it]]] = True
            assert bool(has_tr[nd].all()), "taxonomy node without train member: run valid_labels first"
        else:
            self.n_item_groups = min(k, int(tr.sum()))
        # fixed-size buffers so that EarlyStopper snapshots always reload
        self.register_buffer("user_c", torch.zeros(self.k_users, d, device=dev))
        self.register_buffer("user_ok", torch.zeros(self.k_users, dtype=torch.bool, device=dev))
        self.register_buffer("user_a", torch.zeros(nu, dtype=torch.long, device=dev))
        self.register_buffer("item_c", torch.zeros(self.n_item_groups, d, device=dev))
        self.register_buffer("item_ok", torch.zeros(self.n_item_groups, dtype=torch.bool, device=dev))
        self.register_buffer("item_a", torch.full((ni,), -1, dtype=torch.long, device=dev))
        self.epoch = -1          # no prototype loss before the first e_step, whatever warm_up is

    # ---------- propagation keeping every layer ----------
    def layers(self):
        h = self.bb.layer0(); out = [h]
        for _ in range(max(self.bb.K, 2 * self.hyper)):
            h = spmm(self.bb.adj, h); out.append(h)
        return out

    def forward(self):
        L = self.layers()
        m = torch.stack(L[: self.bb.K + 1], 0).mean(0)
        return m[: self.bb.n_users], m[self.bb.n_users:]

    def representations(self):
        """Scoring representations (same interface as HiTaxProtoCF): plain LightGCN propagation."""
        eu, ei = self.forward()
        return eu, ei, None

    # ---------- E-step (call at the START of every epoch: train(..., epoch_callback=lambda m, ep: m.e_step(ep))) ----
    @torch.no_grad()
    def e_step(self, epoch, seed=None):
        self.epoch = int(epoch)
        seed = self.epoch if seed is None else seed
        e0 = self.bb.layer0().detach()
        nu, dev = self.bb.n_users, e0.device
        eu, ei = e0[:nu], e0[nu:]
        lab_u, _, _, _ = kmeans_labels(eu, list(range(nu)), self.k_users, seed, n_init=self.kmeans_init,
                                       iters=self.kmeans_iters, min_size=1)
        au = torch.tensor([int(lab_u[u][0][1:]) for u in range(nu)], dtype=torch.long, device=dev)
        self.user_a.copy_(au)
        c, ok = self._means(eu, au, self.k_users)
        self.user_c.copy_(F.normalize(c, dim=1)); self.user_ok.copy_(ok)
        if self.item_groups == "kmeans":
            tr = self.is_train_item.nonzero().flatten().tolist()
            lab_i, _, _, _ = kmeans_labels(ei, tr, self.n_item_groups, seed + 7919, n_init=self.kmeans_init,
                                           iters=self.kmeans_iters, min_size=1)
            ai = torch.full((self.bb.n_items,), -1, dtype=torch.long, device=dev)
            for i, v in lab_i.items():
                ai[i] = int(v[0][1:])
            self.item_a.copy_(ai)
            c, ok = self._means(ei[ai >= 0], ai[ai >= 0], self.n_item_groups)
        else:
            mem = self.is_train_item[self.lp_item]
            c, ok = self._means(ei[self.lp_item[mem]], self.lp_node[mem], self.n_item_groups)
        self.item_c.copy_(F.normalize(c, dim=1)); self.item_ok.copy_(ok)

    @staticmethod
    def _means(X, a, k):
        S = torch.zeros(k, X.shape[1], dtype=X.dtype, device=X.device).index_add_(0, a, X)
        cnt = torch.bincount(a, minlength=k)
        return S / cnt.clamp(min=1).unsqueeze(1).to(X.dtype), cnt > 0

    # ---------- losses (each returns batch_ref * batch MEAN = repo's batch SUM at B = batch_ref) ----------
    def _structure(self, L, idx, offset, n_type):
        z0_all = F.normalize(L[0][offset: offset + n_type], dim=1)
        zk = F.normalize(L[2 * self.hyper][offset + idx], dim=1)
        return self.batch_ref * F.cross_entropy(zk @ z0_all.T / self.tau, idx)

    def _logits(self, e, C, ok):
        return (e @ C.T / self.tau).masked_fill(~ok.unsqueeze(0), float("-inf"))

    def proto_loss_users(self, users):
        e = F.normalize(self.bb.layer0()[users], dim=1)
        return self.batch_ref * F.cross_entropy(self._logits(e, self.user_c, self.user_ok), self.user_a[users])

    def proto_loss_items(self, items):
        e = F.normalize(self.bb.layer0()[self.bb.n_users + items], dim=1)
        logp = F.log_softmax(self._logits(e, self.item_c, self.item_ok), dim=1)       # [B, groups]
        if self.item_groups == "kmeans":
            a = self.item_a[items]
            per = torch.where(a >= 0, -logp.gather(1, a.clamp(min=0).unsqueeze(1)).squeeze(1),
                              torch.zeros((), dtype=logp.dtype, device=logp.device))
        else:
            P = self.item_pos[items]                                                    # [B, w], -1 = pad
            m = P >= 0
            nll = torch.where(m, -logp.gather(1, P.clamp(min=0)), torch.zeros((), dtype=logp.dtype, device=logp.device))
            per = nll.sum(1) / m.sum(1).clamp(min=1)                                    # 0 for unlabelled items
        return self.batch_ref * per.sum() / len(items)

    def bpr_loss(self, users, pos, neg, reg=1e-4):
        L = self.layers()
        m = torch.stack(L[: self.bb.K + 1], 0).mean(0)
        nu = self.bb.n_users
        eu, ei = m[:nu], m[nu:]
        x = (eu[users] * ei[pos]).sum(1) - (eu[users] * ei[neg]).sum(1)
        l_s = self._structure(L, users, 0, nu) + self.alpha * self._structure(L, pos, nu, self.bb.n_items)
        loss = -F.logsigmoid(x).mean() + self.lam_s * l_s + reg * self.bb.reg_terms(users, pos, neg)
        if self.epoch >= self.warm_up:
            assert bool(self.user_ok.any()) and bool(self.item_ok.any()), "call e_step(epoch) before the prototype loss"
            loss = loss + self.lam_p * (self.proto_loss_users(users) + self.proto_loss_items(pos))
        return loss
