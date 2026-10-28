"""LightGCN backbone (He et al., SIGIR 2020): layer 0 included, uniform layer weights."""
import torch.nn as nn

from utility.utility_data.data_graph import SegmentAdjacency, spmm


class LightGCN(nn.Module):
    """LightGCN with layer 0 INCLUDED and uniform layer weights 1/(K+1) (He et al., SIGIR 2020)."""

    def __init__(self, n_users, n_items, dim, n_layers, adj, init_std=0.1):
        super().__init__()
        self.n_users, self.n_items, self.K = n_users, n_items, n_layers
        self.emb = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.emb.weight, std=init_std)
        if isinstance(adj, SegmentAdjacency):
            self.adj = adj                                     # submodule with non-persistent buffers
        else:
            self.register_buffer("adj", adj, persistent=False)   # not copied into state_dict snapshots

    def layer0(self):
        """Layer-0 vectors entering propagation (users then items)."""
        return self.emb.weight

    def reg_terms(self, users, pos, neg):
        """L_reg = (1/2|B|) sum (||e_u||^2 + ||e_i+||^2 + ||e_i-||^2) on the PARAMETERS of the batch nodes."""
        w = self.emb.weight; nu = self.n_users
        return 0.5 * (w[users].pow(2).sum(1) + w[nu + pos].pow(2).sum(1) + w[nu + neg].pow(2).sum(1)).mean()

    def propagate(self, h):
        acc = h
        for _ in range(self.K):
            h = spmm(self.adj, h)
            acc = acc + h
        out = acc / (self.K + 1)
        return out[: self.n_users], out[self.n_users:]

    def forward(self):
        return self.propagate(self.layer0())
