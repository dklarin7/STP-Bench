#!/usr/bin/env python3
"""Run several internal->external pairs x encoders as a bounded queue, generating configs as needed.

    .stpbench/bin/python run_openmidnight_queue.py --concurrency 3

Defaults: the five remaining pairs x three encoders, LinearProb only. For each (pair, tag) it writes
config/data/<ns>/<name>_om_<tag>.yaml from the upstream config (DATA.data_dir, preprocess.input_dir,
preprocess.output_dir set to $STP_DATA, default ~/stp_data; DATA.model_name = openmidnight_<tag>) if
missing, then runs run_openmidnight.py for that pair with its own log under logs/queue/.
The LinearProb runs are CPU-bound; 3 concurrent on 12 vCPUs, 6-8 on 48.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PAIRS = [("wustl/BRCA", "hest/BRCA"), ("snu/GBM", "hest/GBM"), ("wustl/PDAC", "hest/PAAD"),
         ("wustl/RCC", "hest/CCRCC"), ("massey/TNBC", "hest/BRCA")]
TAGS = ["teacher_300000", "template_300k", "teacher_50000"]


def ensure_config(data: str, tag: str, stp_data: str) -> Path:
    ns, name = data.split("/")
    dst = ROOT / "config" / "data" / ns / f"{name}_om_{tag}.yaml"
    if dst.exists():
        return dst
    src = ROOT / "config" / "data" / ns / f"{name}.yaml"
    if not src.exists():
        # upstream ships ids.csv for some groups without a data config (hest/GBM): derive from a
        # sibling config in the same namespace, swapping the meta_dir.
        siblings = sorted(p for p in (ROOT / "config" / "data" / ns).glob("*.yaml") if "_om_" not in p.name)
        ids = ROOT / "input" / ns / name / "ids.csv"
        if not siblings or not ids.exists():
            raise FileNotFoundError(f"no config for {data} and nothing to derive it from")
        sib = siblings[0]
        text = sib.read_text().replace(f"input/{ns}/{sib.stem}", f"input/{ns}/{name}")
        print(f"derived {data} config from {sib.relative_to(ROOT)}")
        cfg = yaml.safe_load(text)
    else:
        cfg = yaml.safe_load(src.read_text())
    cfg["DATA"]["data_dir"] = stp_data
    cfg["DATA"]["model_name"] = f"openmidnight_{tag}"
    cfg.setdefault("preprocess", {})
    cfg["preprocess"]["input_dir"] = stp_data
    cfg["preprocess"]["output_dir"] = stp_data
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {dst.relative_to(ROOT)}")
    return dst


def prepare_geneset(data: str, stp_data: str) -> None:
    """Compute the group's HVG gene set once, serially. Each benchmark() would otherwise recompute
    it, and three concurrent copies on a 157K-spot group got OOM-killed. Same command the API runs."""
    ns, name = data.split("/")
    out_dir = ROOT / "input" / ns / name
    if (out_dir / "hmhvg_200genes.json").exists():
        return
    cmd = [sys.executable, str(ROOT / "src" / "preprocess" / "get_geneset.py"), "--st_dir", f"{stp_data}/st",
           "--output_dir", str(out_dir), "--id_path", str(out_dir / "ids.csv"),
           "--n_top_hvg", "50", "--n_top_heg", "1000", "--n_top_hmhvg", "200", "--method", "HMHVG"]
    print(f"{time.strftime('%H:%M:%S')} geneset {data}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", nargs="+", default=[f"{a}:{b}" for a, b in PAIRS], help="internal:external, e.g. wustl/BRCA:hest/BRCA")
    p.add_argument("--tags", nargs="+", default=TAGS)
    p.add_argument("--models", nargs="+", default=["LinearProb"])
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--gpus", default="0", help="GPU ids to spread jobs over, e.g. '0-7' or '0,2,4'; each job gets one")
    p.add_argument("--stp_data", default=os.environ.get("STP_DATA", os.path.expanduser("~/stp_data")))
    args = p.parse_args()

    jobs = []
    groups = []
    for pair in args.pairs:
        internal, external = pair.split(":")
        for g in (internal, external):
            if g not in groups:
                groups.append(g)
        for tag in args.tags:
            ensure_config(internal, tag, args.stp_data)
            ensure_config(external, tag, args.stp_data)
            jobs.append((internal, external, tag))
    for g in groups:  # one at a time: this is the RAM-heavy step
        prepare_geneset(g, args.stp_data)
    (ROOT / "logs" / "queue").mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs, {args.concurrency} at a time")

    gpus = []
    for part in args.gpus.split(","):
        lo, _, hi = part.partition("-")
        gpus.extend(range(int(lo), int(hi or lo) + 1))
    # A ViT-g extraction holds ~16 GB; keep at most 2 jobs per 96 GB card (4 per 40 GB card is too many).
    gpu_load = {g: 0 for g in gpus}
    running = []
    failures = []
    launched = 0
    while jobs or running:
        while jobs and len(running) < args.concurrency:
            internal, external, tag = jobs.pop(0)
            gpu = min(gpus, key=lambda g: gpu_load[g])  # least-loaded card
            gpu_load[gpu] += 1
            log = ROOT / "logs" / "queue" / f"{internal.replace('/', '_')}__{external.replace('/', '_')}__{tag}.out"
            # STP-Bench sets CUDA_VISIBLE_DEVICES itself from gpu_id (an ABSOLUTE card id) for both
            # extraction and training, so an env var here is overwritten; pass the card as --gpu_id.
            cmd = [sys.executable, str(ROOT / "run_openmidnight.py"), "--tags", tag, "--internal", internal,
                   "--external", external, "--models", *args.models, "--gpu_id", str(gpu)]
            proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=ROOT)
            running.append((proc, internal, external, tag, log, gpu))
            launched += 1
            print(f"{time.strftime('%H:%M:%S')} start  {internal} -> {external}  {tag}  gpu={gpu}  (log {log.name})", flush=True)
        for item in list(running):
            proc, internal, external, tag, log, gpu = item
            if proc.poll() is not None:
                running.remove(item)
                gpu_load[gpu] -= 1
                status = "done " if proc.returncode == 0 else f"FAIL({proc.returncode})"
                print(f"{time.strftime('%H:%M:%S')} {status} {internal} -> {external}  {tag}", flush=True)
                if proc.returncode != 0:
                    failures.append((internal, external, tag, str(log)))
        time.sleep(20)

    subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_openmidnight_metrics.py")], cwd=ROOT, check=False)
    if failures:
        print("\nFAILED jobs:")
        for f in failures:
            print("  ", *f)
        sys.exit(1)


if __name__ == "__main__":
    main()
