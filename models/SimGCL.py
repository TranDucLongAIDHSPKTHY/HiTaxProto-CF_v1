"""SimGCL backbone (Yu et al., SIGIR 2022), following the official implementation (SELFRec SimGCL):

    propagation WITHOUT layer 0: e = mean_{k=1..K} e^(k); perturbed view: after every layer
        e^(k) <- e^(k) + sign(e^(k)) * normalize(U[0, 1)) * eps
    contrastive loss: InfoNCE between two perturbed views over the UNIQUE users and items of the batch (in-batch
        negatives, cosine similarity, temperature tau), weighted by cl_lambda; scoring uses the unperturbed view.

Used as the SimGCL baseline (no gate) and as the host of the transfer study (fusion on top of SimGCL).
Declared deviation: the L2 term follows the convention of this codebase (layer-0 parameters of the batch).
"""
import torch
import torch.nn.functional as F

from models.LightGCN import LightGCN
from utility.utility_data.data_graph import spmm


def info_nce(view1, view2, tau):
    """Official SimGCL InfoNCE: -mean log softmax of the diagonal of cos(view1, view2) / tau."""
    v1, v2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    return -torch.diag(F.log_softmax(v1 @ v2.T / tau, dim=1)).mean()


class SimGCL(LightGCN):
    def __init__(self, n_users, n_items, dim, n_layers, adj, init_std=0.1, eps=0.1, tau=0.2, cl_lambda=0.5):
        super().__init__(n_users, n_items, dim, n_layers, adj, init_std)
        self.eps, self.tau, self.cl_lambda = eps, tau, cl_lambda

    def propagate(self, h, perturbed=False):
        layers = []
        for _ in range(self.K):
            h = spmm(self.adj, h)
            if perturbed:
                h = h + torch.sign(h) * F.normalize(torch.rand_like(h), dim=-1) * self.eps
            layers.append(h)
        out = torch.stack(layers, 0).mean(0)
        return out[: self.n_users], out[self.n_users:]

    def forward(self):
        return self.propagate(self.layer0())

    def extra_loss(self, users, pos):
        """cl_lambda * (InfoNCE over batch users + InfoNCE over batch positive items)."""
        u, i = torch.unique(users), torch.unique(pos)
        u1, i1 = self.propagate(self.layer0(), perturbed=True)
        u2, i2 = self.propagate(self.layer0(), perturbed=True)
        return self.cl_lambda * (info_nce(u1[u], u2[u], self.tau) + info_nce(i1[i], i2[i], self.tau))
