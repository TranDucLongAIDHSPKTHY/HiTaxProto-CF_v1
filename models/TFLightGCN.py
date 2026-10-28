"""TF-LightGCN: same-information comparator adapting Kanagal et al. (PVLDB 2012) to LightGCN."""
import torch
import torch.nn as nn

from models.LightGCN import LightGCN

class TFLightGCN(LightGCN):
    """Constructed comparator: adaptation of Kanagal et al. (PVLDB 2012) to LightGCN.
    Layer-0 item vector  h_i = x_i + AGG_{n in N(i)} z_n  with learnable node vectors z_n.
      direct-label variant: N(i) = valid direct labels Λ(i), AGG = mean (multi-label adaptation; no growth with |Λ(i)|);
      ancestor variant: N(i) = kept ancestor set A(i), AGG = sum (Kanagal's additive offsets; DAG: deduplicated set).
    Regularisation follows Kanagal's Gaussian prior on EVERY parameter vector: L2 on x_i and on each z_n separately.
    (L2 on the combined h_i only would make the model a mere re-parameterisation of LightGCN: any change of z could be
    absorbed by the free x_i with identical predictions and penalty.) With separate L2, a sparse item's own offset x_i
    is shrunk towards 0, so h_i is pulled towards its nodes' vectors: the taxonomy prior."""

    def __init__(self, n_users, n_items, dim, n_layers, adj, item_node_pairs, n_nodes, init_std=0.1, agg="mean"):
        super().__init__(n_users, n_items, dim, n_layers, adj, init_std)
        self.z = nn.Embedding(n_nodes, dim)
        nn.init.zeros_(self.z.weight)   # starts identical to LightGCN
        pairs = torch.unique(item_node_pairs[:, :2], dim=0)
        self.register_buffer("pi", pairs[:, 0].clone())
        self.register_buffer("pn", pairs[:, 1].clone())
        cnt = torch.zeros(n_items).index_add_(0, pairs[:, 0], torch.ones(len(pairs)))
        w = (1.0 / cnt.clamp(min=1))[pairs[:, 0]] if agg == "mean" else torch.ones(len(pairs))
        self.register_buffer("pw", w)                       # weight of z_n in h_i for each (item, node) pair
        self.agg = agg

    def layer0(self):
        w = self.emb.weight
        add = torch.zeros(self.n_items, w.shape[1], dtype=w.dtype, device=w.device).index_add(
            0, self.pi, self.z.weight[self.pn] * self.pw.to(w.dtype).unsqueeze(1))
        return torch.cat([w[: self.n_users], w[self.n_users:] + add], 0)

    def reg_terms(self, users, pos, neg):
        """Kanagal-style: L2 on user vectors, on the items' own offsets x_i, and on every node vector z_n used by the
        batch items (each (item, node) occurrence counted, averaged over the batch)."""
        base = super().reg_terms(users, pos, neg)
        zz = self.z.weight.pow(2).sum(1)
        per_item = torch.zeros(self.n_items, dtype=zz.dtype, device=zz.device).index_add(0, self.pi, zz[self.pn])
        return base + 0.5 * (per_item[pos] + per_item[neg]).mean()
