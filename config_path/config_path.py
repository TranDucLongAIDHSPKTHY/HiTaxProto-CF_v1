"""Portable path contract for the HiTaxProto-CF codebase.

Raw sources are downloaded by tools/data/download_data.py into dataset/. A local copy of the TaxPro-CL repository holds
the same files (Amazon/Yelp metadata dumps, ~38 GB, same SHA-256): set HITAX_TAXPROCL_ROOT to its folder to read them
in place instead of downloading them. Everything produced here stays under PROJECT_ROOT.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURE_DIR = PROJECT_ROOT / "configure"
DATASET_DIR = PROJECT_ROOT / "dataset"                  # raw interaction files (LightGCN format, ML-1M, Amazon 2018)
DATASET_VERIFY_DIR = PROJECT_ROOT / "dataset_verify"    # frozen data contract: <dataset>/<split_seed>/
METADATA_DIR = PROJECT_ROOT / "metadata"                # category paths per release, Yelp hierarchy
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"        # data checks and protocol artifacts
RESULT_DIR = Path(os.environ.get("HITAX_RESULT_ROOT") or (PROJECT_ROOT / "results")).resolve()  # selection, pilot, ...
DOCS_DIR = PROJECT_ROOT / "docs"
MODEL_OUTPUT_DIR = Path(os.environ.get("HITAX_OUTPUT_ROOT") or (PROJECT_ROOT / "log")).resolve()

TAXPROCL_ROOT = Path(os.environ["HITAX_TAXPROCL_ROOT"]) if os.environ.get("HITAX_TAXPROCL_ROOT") else None  # optional
RAW_METADATA_DIR = DATASET_DIR / "raw" / "metadata"     # amazon/{2014,2018,2023}, yelp/{2018,2021,2022}


def taxprocl_file(rel):
    """File `rel` of the TaxPro-CL repository copy (read-only), or None if HITAX_TAXPROCL_ROOT is not set."""
    return None if TAXPROCL_ROOT is None else TAXPROCL_ROOT / rel


def metadata_source(rel):
    """Raw metadata file `rel` (e.g. amazon/2014/meta_Books.json): the downloaded copy if present, else the same file
    of the TaxPro-CL repository copy if one is set (identical files, SHA-256 in tools/data/sources.py)."""
    local = RAW_METADATA_DIR / rel
    alt = taxprocl_file("metadata/" + rel)
    return alt if not local.is_file() and alt is not None and alt.is_file() else local

ATOMIC_TEMPORARY_SUFFIX = ".tmp"


def contract_dir(dataset, split_seed):
    return DATASET_VERIFY_DIR / str(dataset) / str(int(split_seed))


def model_config_file(model_name):
    return CONFIGURE_DIR / "{}.txt".format(model_name)


def run_dir(study, dataset, model, setting, seed):
    """log/<study>/<dataset>/<model>/<setting>/seed<seed>/, e.g. log/convergence/yelp2018/LightGCN/lr0.001_reg0.0001/seed42"""
    return MODEL_OUTPUT_DIR / str(study) / str(dataset) / str(model) / str(setting) / "seed{}".format(int(seed))


def temporary_file(path):
    path = Path(path)
    return path.with_name(path.name + ATOMIC_TEMPORARY_SUFFIX)
