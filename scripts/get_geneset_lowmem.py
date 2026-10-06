#!/usr/bin/env python3
"""Memory-light drop-in for src/preprocess/get_geneset.py (HMHVG method), identical outputs.

The upstream script loads every slide of a group into RAM, builds a dense spots x genes pandas table
over all of them and copies it; on the 47-slide WUSTL-BRCA group that exceeds 83 GB and the process
is OOM-killed. The selection it computes is: per-gene mean and (ddof=1) std of RAW counts over all
spots, restricted to the union of per-slide 2000-gene HVG sets, ranked (descending, method='min'),
ranks summed, top-N. Every piece of that streams one slide at a time.

Same CLI as upstream; writes <output_dir>/total_<N>genes.json and <output_dir>/hmhvg_<n>genes.json.
Only --method HMHVG is implemented (the benchmark's gene_type); other methods fall through to upstream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm

EXCLUDE_PREFIXES = ("NegControlCodeword", "NegControlProbe", "UnassignedCodeword")
DROP_PREFIXES = ("MT", "mt", "RPS", "RPL")


def file_list(st_dir, id_path):
    if id_path:
        ids = pd.read_csv(id_path)["sample_id"].dropna().astype(str).tolist()
        files = [f"{st_dir}/{sid}.h5ad" for sid in ids]
    else:
        from glob import glob
        files = sorted(glob(f"{st_dir}/*.h5ad"))
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError("missing ST files: " + ", ".join(missing[:10]))
    return files


def common_gene_set(files):
    common = None
    for f in tqdm(files, desc="Reading var_names"):
        names = {g for g in sc.read_h5ad(f, backed="r").var_names if not g.startswith(EXCLUDE_PREFIXES)}
        common = names if common is None else common & names
    return common


def per_sample_hvgs(adata, common_sorted):
    tmp = adata[:, common_sorted].copy()
    sc.pp.filter_cells(tmp, min_genes=1)
    sc.pp.filter_genes(tmp, min_cells=1)
    sc.pp.normalize_total(tmp, inplace=True)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=2000)
    return set(tmp.var_names[tmp.var["highly_variable"]])


def column_sums(X):
    """(sum, sum of squares) per column for sparse or dense X, as float64 1-D arrays."""
    if sp.issparse(X):
        X = X.tocsr().astype(np.float64)
        return np.asarray(X.sum(axis=0)).ravel(), np.asarray(X.multiply(X).sum(axis=0)).ravel()
    X = np.asarray(X, dtype=np.float64)
    return X.sum(axis=0), (X * X).sum(axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--st_dir", required=True)
    p.add_argument("--n_top_hvg", type=int, default=50)
    p.add_argument("--n_top_heg", type=int, default=1000)
    p.add_argument("--n_top_hmhvg", type=int, default=500)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--method", default="HMHVG", choices=["HVG", "HEG", "HMHVG", "ALL"])
    p.add_argument("--id_path", default=None)
    args = p.parse_args()
    if args.method != "HMHVG":
        sys.exit("only --method HMHVG is implemented here; use src/preprocess/get_geneset.py for the others")

    files = file_list(args.st_dir, args.id_path)
    common = common_gene_set(files)
    common_sorted = sorted(common)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/total_{len(common_sorted)}genes.json", "w") as f:
        json.dump({"genes": common_sorted}, f)

    # pass 1: union of per-slide HVGs (upstream: tqdm "Computing per-sample HVGs")
    union_hvg = set()
    for f in tqdm(files, desc="Computing per-sample HVGs"):
        union_hvg |= per_sample_hvgs(sc.read_h5ad(f), common_sorted)
    union_hvg = sorted(g for g in union_hvg if not g.startswith(DROP_PREFIXES))

    # pass 2: streaming per-gene sum / sum-of-squares of RAW counts over every spot of every slide.
    # Upstream: pd.concat of dense frames then .mean() and .std() (ddof=1). union_hvg is a subset of
    # the common genes, so every slide has every column and upstream's fillna(0) is a no-op.
    s = np.zeros(len(union_hvg)); ss = np.zeros(len(union_hvg)); n = 0
    for f in tqdm(files, desc="Accumulating count statistics"):
        adata = sc.read_h5ad(f)
        X = adata[:, union_hvg].X
        cs, css = column_sums(X)
        s += cs; ss += css; n += adata.shape[0]
    mean = s / n
    var = (ss - n * mean ** 2) / (n - 1)
    std = np.sqrt(np.maximum(var, 0.0))

    mean_ranks = pd.Series(mean, index=union_hvg).rank(ascending=False, method="min")
    std_ranks = pd.Series(std, index=union_hvg).rank(ascending=False, method="min")
    combined = mean_ranks + std_ranks
    hmhvg = combined.sort_values().head(args.n_top_hmhvg).index.tolist()
    with open(f"{args.output_dir}/hmhvg_{len(hmhvg)}genes.json", "w") as f:
        json.dump({"genes": hmhvg}, f)
    print(f"{len(files)} slides, {n} spots, {len(common_sorted)} common genes, {len(union_hvg)} union HVGs -> {len(hmhvg)} HMHVG written to {args.output_dir}")


if __name__ == "__main__":
    main()
