# OpenMidnight encoders on STP-Bench

Branch `openmidnight` of this fork adds three OpenMidnight (DINOv2 ViT-g/14, TCGA-pretrained) teacher
exports as patch encoders and benchmarks them with the stock predictors. Everything else is upstream.

| encoder name | checkpoint | PathoROB robustness (nanopath probe) |
|---|---|---|
| `openmidnight_teacher_50000` | Phase 0 run, step 50K (58M samples) | 0.754 |
| `openmidnight_teacher_300000` | Phase 0 run, step 300K (346M samples) | 0.567 |
| `openmidnight_template_300k` | 300K export + 4K-step robustifying fine-tune (synthetic acquisition shift) | 0.854 |

Features: CLS token, 1536-d, fp16, 224 px, ImageNet mean/std; same transforms as the other encoders.
Loaded through `openmidnight_probe.loader` from the OpenMidnight repo (`$OPENMIDNIGHT_ROOT`).

## Results: five cancer types, LinearProb, Pearson over 200 HMHVGs

Internal = k-fold CV on the paper's training set; external = the paper's held-out set for that
cancer type, trained on all internal folds. Runs 2026-10-05 (lung, 1x A100) and 2026-10-07/08
(the rest, 8x RTX PRO 6000 via `run_openmidnight_queue.py` on the `queue` branch). Per-fold values
from `scripts/collect_openmidnight_metrics.py`; raw logs and CSVs in `gs://wsi-brb/robustify/stp_bench/`.

### In-distribution (internal CV; mean over folds)

| training set | folds | 50K | 300K | template_300k | paper's UNIv2 linear baseline |
|---|---|---|---|---|---|
| NCCHE-LUAD-Xenium | 5 | 0.588 | 0.583 | 0.581 | 0.586 |
| HEST-CCRCC | 6 | 0.311 | 0.315 | 0.310 | ~0.31 |
| SNU-GBM | 5 | 0.270 | 0.280 | 0.281 | ~0.27 |
| WUSTL-BRCA | 5 | 0.349 | 0.357 | 0.352 | ~0.345 |
| WUSTL-PDAC | 5 | 0.296 | 0.299 | 0.302 | ~0.30 |
| HEST-PRAD | 2 | 0.377 | 0.364 | 0.377 | ~0.40 |

The three encoders are indistinguishable in-distribution, and all sit on the paper's UNIv2 linear
baseline (read off its Fig. 2 panels; Xenium is the one quoted in text) except PRAD, ~0.03 under.

### Out-of-distribution (external set; mean and fold range)

| train -> test | 50K | 300K | template_300k | template - 300K |
|---|---|---|---|---|
| NCCHE-LUAD-Xenium -> HEST-LUAD | 0.524 (0.494-0.554) | 0.578 (0.567-0.591) | **0.627 (0.615-0.637)** | +0.049, fold ranges disjoint |
| HEST-CCRCC -> WUSTL-RCC | 0.199 (0.173-0.236) | 0.177 (0.154-0.204) | **0.217 (0.204-0.232)** | +0.039, fold ranges disjoint |
| WUSTL-PDAC -> HEST-PAAD | 0.236 (0.138-0.364) | 0.269 (0.126-0.355) | **0.300 (0.186-0.405)** | +0.031, overlapping |
| SNU-GBM -> HEST-GBM | **0.266 (0.163-0.301)** | 0.204 (0.117-0.259) | 0.231 (0.152-0.289) | +0.026, overlapping |
| WUSTL-BRCA -> Massey-TNBC | 0.257 (0.237-0.275) | **0.259 (0.245-0.274)** | 0.247 (0.222-0.261) | -0.012, overlapping |

Reading: robustifying the 300K export (synthetic acquisition-shift fine-tune, 4K steps) leaves
in-distribution prediction unchanged and improves transfer to the external cohort in four of five
cancer types, by 0.03-0.05 Pearson, clearly in lung and kidney (every template fold above every
base fold), within fold noise in pancreas and brain, and within noise the other way in breast.
The 50K export, the most PathoROB-robust checkpoint of the original run, is not a better transfer
encoder: worst in lung and pancreas, best in brain (3 external slides). PRAD has no public external
set (MGB-PRAD is withheld).

StFlow (lung only): teacher_300000 0.475 internal / 0.553 external; template_300k 0.276 / 0.332 with
folds near zero. A predictor-side instability on raw 1536-d features, not an encoder result.

Caveats: one run per cell (no seed repeats); external sets are 2-10 slides; Visium PCCs are low across
the board, as in the paper. Two runs on the same encoder and data reproduce to four decimals.

## Reproduce

```bash
export OPENMIDNIGHT_ROOT=$HOME/OpenMidnight OPENMIDNIGHT_CKPT_DIR=$HOME/checkpoints   # <tag>.pth per encoder
# lung (single pair):
.stpbench/bin/python run_openmidnight.py --tags teacher_300000 template_300k teacher_50000 --models LinearProb
# the other pairs, all encoders, spread over GPUs (branch `queue`):
.stpbench/bin/python run_openmidnight_queue.py --pairs wustl/BRCA:massey/TNBC snu/GBM:hest/GBM wustl/PDAC:hest/PAAD hest/CCRCC:wustl/RCC --concurrency 12 --gpus 0-7
.stpbench/bin/python run_openmidnight_queue.py --pairs hest/PRAD:hest/PRAD --concurrency 3 --gpus 0-2
.stpbench/bin/python scripts/collect_openmidnight_metrics.py
```
Machine: 128 GB+ RAM (the gene-set step), many cores (the probes are CPU-bound), any GPU with
kernels for its torch build. See OPENMIDNIGHT_SETUP.md.
