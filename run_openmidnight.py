#!/usr/bin/env python3
"""Benchmark the OpenMidnight encoders on STP-Bench: NCCHE Xenium (internal 5-fold) -> HEST LUAD (external).

    .stpbench/bin/python run_openmidnight.py                       # both encoders
    .stpbench/bin/python run_openmidnight.py --tags teacher_300000 # one

Encoders are the data configs config/data/{ncche/xenium,hest/LUAD}_om_<tag>.yaml, which differ from
the originals only in DATA.data_dir and DATA.model_name (openmidnight_<tag>). Checkpoints resolve to
$OPENMIDNIGHT_CKPT_DIR/<tag>.pth, repo at $OPENMIDNIGHT_ROOT (see extract_img_features.py).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from stpbench import STPred  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tags", nargs="+", default=["teacher_300000", "template_300k"])
p.add_argument("--models", nargs="+", default=["LinearProb", "StFlow"])
p.add_argument("--internal", default="ncche/xenium")
p.add_argument("--external", default="hest/LUAD")
p.add_argument("--gpu_id", type=int, default=0)
args = p.parse_args()

for tag in args.tags:
    internal, external = f"{args.internal}_om_{tag}", f"{args.external}_om_{tag}"
    stp = STPred(models=args.models, gpu=1, gpu_id=args.gpu_id, repo_root=os.path.dirname(os.path.abspath(__file__)),
                 log_file=f"logs/openmidnight_{tag}.log")
    # benchmark() runs preprocessing (gene set, feature extraction) itself; check() only passes afterwards.
    result = stp.benchmark(internal_data=internal, external_data=external)
    stp.check(data=internal, strict=False)
    print(f"\n===== {tag} =====")
    print(result.summary())
    result.save(f"benchmark_openmidnight_{tag}.csv")
