"""Normalised bipartite adjacency built from TRAIN edges only (COO, CSR or deterministic segment layout)."""
import contextlib
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

def build_norm_adj(n_users, n_items, edges, dtype=None, layout="coo"):
    """Symmetric-normalised bipartite adjacency (TRAIN edges only).

    edges: LongTensor [E, 2] of (user, item) pairs.  Node order: users 0..U-1, items U..U+I-1.
    layout: 'coo' or 'csr' (same values; CSR spmm is ~1.9x faster on GPU), or 'segment': the CSR values multiplied
    by gather + segment sum (SegmentAdjacency), bitwise reproducible on CUDA (cuSPARSE spmm is not) but ~5x slower.
    """
    dtype = dtype or torch.get_default_dtype()
    u, i = edges[:, 0], edges[:, 1] + n_users
    n = n_users + n_items
    row = torch.cat([u, i]); col = torch.cat([i, u])
    deg = torch.zeros(n, dtype=dtype).scatter_add_(0, row, torch.ones_like(row, dtype=dtype))
    inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    val = inv_sqrt[row] * inv_sqrt[col]
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), val, (n, n), check_invariants=True).coalesce()
    if layout == "csr":
        return adj.to_sparse_csr()
    if layout == "segment":
        return SegmentAdjacency(adj.to_sparse_csr())
    if layout != "coo":
        raise ValueError("layout must be 'coo', 'csr' or 'segment'")
    return adj


class SegmentAdjacency(nn.Module):
    """CSR adjacency whose product is computed as segment_sum(values * h[col]) over the rows: every output row is
    reduced in a fixed order, so forward and backward are bitwise reproducible on CUDA."""

    def __init__(self, csr):
        super().__init__()
        self.shape = tuple(csr.shape)
        self.register_buffer("crow", csr.crow_indices().clone(), persistent=False)
        self.register_buffer("col", csr.col_indices().clone(), persistent=False)
        self.register_buffer("val", csr.values().clone(), persistent=False)

    def mm(self, h):
        return torch.segment_reduce(h.index_select(0, self.col) * self.val.unsqueeze(1), "sum", offsets=self.crow)


def spmm(adj, h):
    """adj @ h for every adjacency layout of build_norm_adj."""
    return adj.mm(h) if isinstance(adj, SegmentAdjacency) else torch.sparse.mm(adj, h)


@contextlib.contextmanager
def exact_propagation(model):
    """Temporarily replace every sparse adjacency buffer `adj` of the model by its SegmentAdjacency (same values),
    so that the representations computed inside are bitwise reproducible on CUDA (final evaluations of a checkpoint)."""
    swapped = []
    for m in model.modules():
        a = m._buffers.get("adj") if hasattr(m, "_buffers") else None
        if isinstance(a, torch.Tensor) and a.layout in (torch.sparse_coo, torch.sparse_csr):
            seg = SegmentAdjacency(a.to_sparse_csr() if a.layout == torch.sparse_coo else a).to(a.device)
            del m._buffers["adj"]
            m.add_module("adj", seg)
            swapped.append((m, a))
    try:
        yield model
    finally:
        for m, a in swapped:
            del m._modules["adj"]
            m.register_buffer("adj", a, persistent=False)
