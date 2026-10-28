"""Reader of the frozen data contract dataset_verify/<dataset>/<split_seed>/.

The test split is SEALED: its rows live apart from the training contract (sealed_test/test_interactions.csv) and
are returned only when a frozen protocol file is given whose recorded SHA-256 matches both the manifest and the
file on disk. Training, early stopping and every configuration choice therefore use train and validation only.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from config_path.config_path import contract_dir


class SealedTestError(PermissionError):
    """Raised when the test split is requested without a valid frozen protocol."""


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path, keep):
    rows = {s: [] for s in keep}
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader) != ["user_id", "item_id", "split"]:
            raise ValueError("unexpected header in {}".format(path))
        for u, i, s in reader:
            if s == "test" and path.name == "interactions.csv":
                raise SealedTestError("{} still contains test rows: rebuild the contract "
                                      "(python -m tools.data.build_interactions --force)".format(path))
            if s in keep:
                rows[s].append((int(u), int(i)))
    return rows


class ContractData:
    def __init__(self, dataset, split_seed, splits=("train", "val"), protocol=None):
        self.dataset, self.split_seed = dataset, int(split_seed)
        self.dir = Path(contract_dir(dataset, split_seed))
        self.manifest = json.loads((self.dir / "split_manifest.json").read_text(encoding="utf-8"))
        self.n_users = int(self.manifest["counts"]["users"])
        self.n_items = int(self.manifest["counts"]["items"])
        keep = set(splits)
        rows = _read_rows(self.dir / "interactions.csv", keep - {"test"})
        if "test" in keep:
            rows["test"] = _read_rows(self._unseal(protocol), {"test"})["test"]
        self.edges = {s: np.asarray(v, dtype=np.int64).reshape(-1, 2) for s, v in rows.items()}
        if "train" in keep:
            tr = self.edges["train"]
            self.train_degree = torch.as_tensor(np.bincount(tr[:, 1], minlength=self.n_items))

    def _unseal(self, protocol):
        sealed = self.manifest.get("sealed_test")
        if sealed is None:
            raise SealedTestError("the contract has no sealed test file; rebuild it")
        if protocol is None:
            raise SealedTestError("the test split is sealed: it is read only by the sealed test evaluation, with a "
                                  "frozen protocol file (protocol/frozen_protocol.json)")
        proto = json.loads(Path(protocol).read_text(encoding="utf-8"))
        entry = proto.get("datasets", {}).get("{}/{}".format(self.dataset, self.split_seed))
        path = self.dir / sealed["file"]
        if proto.get("frozen") is not True or entry is None:
            raise SealedTestError("protocol {} is not frozen or does not cover {}/{}".format(
                protocol, self.dataset, self.split_seed))
        if not (entry.get("sealed_test_sha256") == sealed["sha256"] == _sha256(path)):
            raise SealedTestError("sealed test hash mismatch between protocol, manifest and file")
        return path

    def label_pairs(self, name="item_labels.csv"):
        """(item, node) pairs of a label file of the contract (item_labels.csv, shuffled_labels_s<seed>.csv)."""
        with open(self.dir / name, encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader) != ["item_id", "node_id"]:
                raise ValueError("unexpected header in {}".format(name))
            rows = [(int(i), int(n)) for i, n in reader]
        return torch.tensor(rows, dtype=torch.long).reshape(-1, 2)

    def n_nodes(self):
        with open(self.dir / "taxonomy_nodes.csv", encoding="utf-8") as stream:
            return sum(1 for _ in stream) - 1

    def by_user(self, split):
        out = {}
        for u, i in self.edges[split]:
            out.setdefault(int(u), []).append(int(i))
        return out

    def seen_for(self, split):
        """Items excluded from the ranking: train at validation, train + val at test."""
        seen = self.by_user("train")
        if split == "test":
            for u, items in self.by_user("val").items():
                seen.setdefault(u, []).extend(items)
        return seen
