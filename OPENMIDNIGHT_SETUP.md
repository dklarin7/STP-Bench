# Setting up a box for the OpenMidnight STP-Bench runs

Exactly what was run on 2026-10-05 on a GCP `a2-highgpu-1g` (1x A100 40 GB, 12 vCPU, Ubuntu minimal
26.04, 300 GB disk). One command per step; run them in order. Steps 1-2 are the GCP/26.04 driver
flow; skip them on an image that already has a driver. Pick a box with 48+ vCPUs next time: the
LinearProb runs are CPU-bound and the GPU is only busy during feature extraction.

## 1. Kernel headers and build tools

```bash
sudo apt-get update -qq && sudo apt-get install -y linux-headers-$(uname -r) build-essential
```

## 2. NVIDIA driver (open kernel module), then reboot

```bash
sudo apt-get install -y nvidia-driver-580-open && sudo reboot
```

After the reboot, `nvidia-smi` must list the GPU.

## 3. Tools: git, gh, zellij (with mouse mode and pane frames off)

```bash
sudo apt-get install -y git gh python3-venv && curl -sL https://github.com/zellij-org/zellij/releases/latest/download/zellij-x86_64-unknown-linux-musl.tar.gz | tar xz -C /tmp && sudo install /tmp/zellij /usr/local/bin/ && mkdir -p ~/.config/zellij && printf 'mouse_mode false\npane_frames false\n' > ~/.config/zellij/config.kdl && zellij --version
```

## 4. uv (their environment script expects it on PATH)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
```

## 5. GitHub login (device flow, no browser on the box)

```bash
BROWSER=echo gh auth login -h github.com -p https -w
```

Enter the printed code at https://github.com/login/device, then:

```bash
gh auth setup-git
```

## 6. Clone this fork on the `openmidnight` branch and build the environment

```bash
cd ~ && git clone -b openmidnight https://github.com/dklarin7/STP-Bench.git && cd STP-Bench && bash scripts/create_env.sh
```

Several minutes: Python 3.11, torch 2.3.1+cu118 (no Blackwell/sm_120 kernels; use A100/H100/L4/T4).

## 7. Fix two package issues their script leaves

```bash
~/.local/bin/uv pip install --python .stpbench/bin/python "torchmetrics>=1.6"
```

```bash
sudo apt-get install -y ca-certificates && sudo update-ca-certificates --fresh && ls -la /etc/ssl/certs/ca-certificates.crt
```

(torchmetrics < 1.5 lacks `MeanAbsoluteError(num_outputs=...)`, which their trainer calls. The
Hugging Face xet downloader is a Rust client that reads the system bundle at
`/etc/ssl/certs/ca-certificates.crt`; on Ubuntu minimal the package can be present without that
bundle ever having been generated, and the download then fails with "No CA certificates were
loaded from the system". `update-ca-certificates --fresh` writes it; expect a file of a few hundred
KB. If it still fails, prefix the download with `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`.)

## 8. Verify the environment sees the GPU

```bash
cd ~/STP-Bench && .stpbench/bin/python -c "import torch, trident; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 9. Hugging Face login (read token; the dataset is click-through gated)

```bash
cd ~/STP-Bench && .stpbench/bin/hf auth login
```

## 10. Download the benchmark data, without the 303 GB of WSIs (~140 GB, 1-2 h)

Start a zellij session first so it survives the SSH connection:

```bash
zellij -s stp
```

```bash
cd ~/STP-Bench && .stpbench/bin/hf download nexgem/STP-Bench --repo-type dataset --include "patches/*" "st/*" "metadata/*" --local-dir ~/stp_data
```

Rerunning the same command resumes. If xet still fails, prefix with `HF_HUB_DISABLE_XET=1`.

## 11. OpenMidnight repo (the loader the encoder uses)

```bash
cd ~ && git clone -b robustify https://github.com/dklarin7/openmidnight-phase0.git OpenMidnight
```

## 12. Checkpoints (one 4.4 GB file per encoder, names must match the encoder tags)

```bash
mkdir -p ~/checkpoints && gcloud storage cp gs://wsi-brb/phase0-24gpu/eval/training_300000/teacher_checkpoint.pth ~/checkpoints/teacher_300000.pth
```

```bash
gcloud storage cp gs://wsi-brb/robustify/template_300k/eval/training_4000/teacher_checkpoint.pth ~/checkpoints/template_300k.pth
```

```bash
gcloud storage cp gs://wsi-brb/phase0-24gpu/eval/training_50000/teacher_checkpoint.pth ~/checkpoints/teacher_50000.pth
```

## 13. Encoder load test (builds ViT-g from the export, two random images through it)

```bash
cd ~/STP-Bench && .stpbench/bin/python -c "
import sys, torch; sys.path.insert(0, 'src/preprocess')
from extract_img_features import OpenMidnightInferenceEncoder
e = OpenMidnightInferenceEncoder('openmidnight_teacher_300000').eval().cuda()
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16): print(e(torch.randn(2, 3, 224, 224).cuda()).shape)
"
```

Expect `dim=1536` and `torch.Size([2, 1536])`.

## 14. Run (one encoder per zellij pane; they share the GPU fine, the CPU is the limit)

```bash
cd ~/STP-Bench && mkdir -p logs && OPENMIDNIGHT_ROOT=$HOME/OpenMidnight OPENMIDNIGHT_CKPT_DIR=$HOME/checkpoints .stpbench/bin/python run_openmidnight.py --tags teacher_300000 --models LinearProb 2>&1 | tee logs/lp_300k.out
```

Other encoders: `--tags template_300k`, `--tags teacher_50000`. Other tissue pairs: add
`--internal wustl/BRCA --external hest/BRCA` (the `_om_<tag>` data configs must exist for both;
copy the lung ones and change the two paths and `model_name`). ~4.5 h per 16-slide Xenium group on
12 vCPUs; Visium groups are smaller.

## 15. Collect metrics and save them

```bash
cd ~/STP-Bench && .stpbench/bin/python scripts/collect_openmidnight_metrics.py
```

```bash
cd ~/STP-Bench && tar czf stp_openmidnight_logs.tgz logs results && gcloud storage cp stp_openmidnight_logs.tgz results/openmidnight_metrics.csv results/openmidnight_metrics_summary.csv gs://wsi-brb/robustify/stp_bench/
```

## Detaching a job that was started outside zellij

Press `Ctrl-Z`, then:

```bash
bg && disown -h %1 && echo detached
```

## Shutting the box down but keeping the data and environment

```bash
gcloud compute instances delete <instance> --zone <zone> --keep-disks=boot
```
