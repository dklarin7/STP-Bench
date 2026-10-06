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

## Lung: train NCCHE Xenium (16 slides, 5-fold) -> test HEST LUAD (2 slides), 200 HVGs, Pearson

Run 2026-10-05/06 on one A100 40 GB. Per-fold metrics collected by `scripts/collect_openmidnight_metrics.py`
(upstream `BenchmarkResult.save()` wrote empty files); raw logs in `gs://wsi-brb/robustify/stp_bench/`.

| encoder | LinearProb internal (mean, min–max over 5 folds) | LinearProb external LUAD |
|---|---|---|
| teacher_50000 | 0.588 (0.547–0.633) | 0.524 (0.494–0.554) |
| teacher_300000 | 0.583 (0.543–0.632) | 0.578 (0.567–0.591) |
| **template_300k** | 0.581 (0.532–0.632) | **0.627 (0.615–0.637)** |

- In-distribution the three encoders are indistinguishable.
- Under the platform/lab shift to HEST LUAD the ranking follows PathoROB robustness exactly, and the
  fold ranges do not overlap: the robustified encoder's worst fold (0.615) beats the base encoder's
  best (0.591), which beats the 50K export's best (0.554).
- StFlow (teacher_300000: internal 0.475, external 0.553; template_300k: 0.276 / 0.332) diverged on
  individual folds (min -0.03) with the template encoder. A predictor-side instability on raw 1536-d
  features, not an encoder result; LinearProb on the same embeddings is the comparison above.

Only one cancer type so far. Remaining natural pairs: wustl/BRCA -> hest/BRCA, snu/GBM -> hest/GBM,
wustl/PDAC -> hest/PAAD, wustl/RCC -> hest/CCRCC, massey/TNBC -> hest/BRCA. Each is the same config
copy plus `run_openmidnight.py --internal <ns/name> --external <ns/name>`. The runs are CPU-bound
(per-spot h5 reads at ~6 batches/s with the GPU idle): rent cores, not GPUs, and run pairs in parallel.

## Reproduce

```bash
export OPENMIDNIGHT_ROOT=$HOME/OpenMidnight OPENMIDNIGHT_CKPT_DIR=$HOME/checkpoints   # <tag>.pth per encoder
.stpbench/bin/python run_openmidnight.py --tags teacher_300000 template_300k teacher_50000 --models LinearProb
.stpbench/bin/python scripts/collect_openmidnight_metrics.py
```
