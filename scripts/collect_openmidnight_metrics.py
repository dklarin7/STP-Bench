#!/usr/bin/env python3
"""Collect STP-Bench metrics from disk into one CSV, independent of the BenchmarkResult object.

Walks logs/<namespace>/<data>_om_<encoder>/<Model>/<timestamp>/fold<k>/ and reads the last row of
    eval/metrics.csv                         (internal k-fold)
    eval_external/<ns>/<data>/metrics.csv    (external evaluation)
Writes per-fold rows plus a per-(encoder, model, split) summary with mean, min, max over folds.

    python scripts/collect_openmidnight_metrics.py                  # -> results/openmidnight_metrics.csv
    python scripts/collect_openmidnight_metrics.py --metric test_PearsonCorrCoef --out my.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

FOLD_RE = re.compile(r"/(fold\d+)/")
DATA_RE = re.compile(r"^logs/([^/]+/[^/]+)_om_([A-Za-z0-9_]+)/([^/]+)/([^/]+)/fold")


def collect(logs_dir: str = "logs") -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "**", "fold*", "eval*", "**", "metrics.csv"), recursive=True)):
        rel = path.replace("\\", "/")
        m = DATA_RE.search(rel)
        if not m:
            continue
        data, encoder, model, timestamp = m.groups()
        fold = FOLD_RE.search(rel).group(1)
        if "/eval_external/" in rel:
            split = "external:" + rel.split("/eval_external/", 1)[1].rsplit("/metrics.csv", 1)[0]
        else:
            split = "internal"
        last = pd.read_csv(path).iloc[-1].to_dict()
        rows.append({"encoder": encoder, "train_data": data, "model": model, "timestamp": timestamp, "fold": fold, "split": split,
                     **{k: v for k, v in last.items() if isinstance(v, (int, float))}})
    if not rows:
        raise SystemExit(f"no metrics.csv found under {logs_dir}")
    df = pd.DataFrame(rows)
    # Two runs of the same (encoder, model, split, fold) produce identical numbers (deterministic);
    # keep the latest timestamp so re-runs do not double-count.
    df = df.sort_values("timestamp").drop_duplicates(subset=["encoder", "train_data", "model", "fold", "split"], keep="last")
    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["train_data", "split", "model", "encoder"])[metric]
    out = g.agg(mean="mean", min="min", max="max", folds="count").round(4).reset_index()
    return out.sort_values(["train_data", "split", "model", "mean"], ascending=[True, True, True, False])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", default="logs")
    p.add_argument("--metric", default="test_PearsonCorrCoef")
    p.add_argument("--out", default="results/openmidnight_metrics.csv")
    args = p.parse_args()
    df = collect(args.logs)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = summarize(df, args.metric)
    summary_path = args.out.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(f"\nper-fold rows: {len(df)} -> {args.out}\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
