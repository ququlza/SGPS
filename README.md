# SGPS: SURE Guided Posterior Sampling

[![arXiv](https://img.shields.io/badge/arXiv-2512.23232-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.23232)

Code for **"SURE Guided Posterior Sampling: Trajectory Correction for
Diffusion-Based Inverse Problems"** (IEEE Access).

## Overview

SGPS is a training-free posterior sampling method for diffusion-based inverse
problems. At each noise level, it obtains a denoiser estimate, applies Langevin
measurement guidance, and corrects the guided sample using a Monte Carlo
Stein's unbiased risk estimate (SURE) objective. A patch-based PCA estimator
provides the plug-in scale for the correction, and its gradient is computed
through the frozen denoiser with respect to the guided sample.

The provided implementation and configurations support pixel-space sampling
with an FFHQ 256 × 256 DDPM prior across eight inverse problems.

## Installation

### 1. Environment

Reproduction-tested versions:

- Python: `3.8`
- PyTorch: `2.3.0`
- CUDA: `12.1`

```bash
conda create -n sgps python=3.8 -y
conda activate sgps
python -m pip install -r requirements.txt
python -m pip install gdown
```

### 2. Checkpoints

Create the checkpoint directory in the SGPS folder:

```bash
mkdir checkpoints
```

Required checkpoints:

- FFHQ DDPM: [`ffhq256.pt`](https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh)
- Nonlinear blur model: [`GOPRO_wVAE.pth`](https://drive.google.com/uc?id=1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy)

Download the FFHQ diffusion prior:

```bash
gdown "https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh" -O checkpoints/ffhq256.pt
```

For nonlinear deblurring, also download the BKSE checkpoint:

```bash
gdown "https://drive.google.com/uc?id=1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy" -O checkpoints/GOPRO_wVAE.pth
```

### 3. Dataset

FFHQ test images: [`test-ffhq.zip`](https://drive.google.com/uc?id=1i0oI8nt_b9XCHNPKM5KR92Y4t8ZVMDvR).

```bash
mkdir dataset
gdown "https://drive.google.com/uc?id=1i0oI8nt_b9XCHNPKM5KR92Y4t8ZVMDvR" -O dataset/test-ffhq.zip
python -m zipfile -e dataset/test-ffhq.zip dataset
```

Place the extracted RGB images under `dataset/test-ffhq/`. Preserve the numeric
source-ID filenames, such as `00000.png` and `00003.png`. For exact paper
reproduction, verify the source images against the original FFHQ manifest.
The checkpoint and dataset links above are the ones provided by the upstream
project acknowledged below; the assets are not bundled in this repository.

## Quick Start

Predefined commands for all tasks are provided in
[`configs_sgps_cli.txt`](configs_sgps_cli.txt).

Supported tasks:

- `down_sampling`: super-resolution ×4
- `inpainting`: box inpainting
- `inpainting_rand`: random inpainting
- `gaussian_blur`: Gaussian deblurring
- `motion_blur`: motion deblurring
- `phase_retrieval`: phase retrieval
- `nonlinear_blur`: nonlinear deblurring
- `hdr`: high dynamic range reconstruction

Supported space: **pixel**. Supported dataset/prior: **FFHQ 256 × 256**.

Example — super-resolution ×4, T=16:

```bash
python main.py --task down_sampling --T 16 --data dataset/test-ffhq --checkpoint checkpoints/ffhq256.pt --output results/sr4_T16 --evaluate
```

Example — super-resolution ×4, T=33:

```bash
python main.py --task down_sampling --T 33 --data dataset/test-ffhq --checkpoint checkpoints/ffhq256.pt --output results/sr4_T33 --evaluate
```

Example — nonlinear deblurring:

```bash
python main.py --task nonlinear_blur --T 16 --data dataset/test-ffhq --checkpoint checkpoints/ffhq256.pt --output results/nonlinear_T16 --evaluate
```

Use `--image_ids 3-12` to reconstruct only source IDs 3 through 12. This selects
the original IDs, not new indices assigned after subsetting. Each run must use
a new or empty output directory.

## Configuration and Evaluation

The default SGPS configuration uses `alpha=0.5`, 100 Langevin guidance steps per
outer step, PCA patch size 8, one Gaussian MC-SURE probe, batch size 1, and seed
42. Task-specific operator and guidance settings are defined in `main.py`.

Results are saved under the requested `--output` directory:

```text
recon/                    reconstruction PNGs, organized by trial
gt/                       ground-truth PNGs used for evaluation
per_image.csv             image-level sampling time, PSNR and LPIPS
summary.csv               mean sampling time; PSNR/LPIPS mean and sample SD
config.json               execution settings and selected source IDs
```

PSNR and LPIPS are computed from the same saved RGB PNGs. LPIPS uses the
official `lpips==0.1.4` package with VGG, version 0.1. Phase retrieval uses four
trials, selects the largest PNG PSNR per image, and reports LPIPS from that same
reconstruction. `per_trial.csv` additionally records the four phase-retrieval
trials; LPIPS is evaluated only for the selected reconstruction. VGG weights
may be downloaded on the first LPIPS evaluation. Without `--evaluate`, PSNR
and sampling time are still recorded, but LPIPS is left blank.

Timing covers sampling only, with CUDA synchronization before and after the
sampler. Model loading, image I/O and metric evaluation are excluded. For phase
retrieval, per-image time includes all four trials. No runtime SD is reported.

Existing measurements can be reused with `--manifest` and
`--measurement_source`; source IDs, hashes and measurement settings are
validated before sampling. Without these options, measurements are generated
using the configured seed and original source IDs.

## Acknowledgements

This implementation is built on [FAST-DIPS](https://github.com/ququlza/FAST-DIPS).

We thank the authors of the following projects for making their code available:

- [DAPS](https://github.com/zhangbingliang2019/DAPS)
- [BKSE](https://github.com/VinAIResearch/blur-kernel-space-exploring), for the nonlinear blur operator
- [motionblur](https://github.com/LeviBorodenko/motionblur), for the motion blur operator

## Citation

If you find this code useful, please cite our paper:

```bibtex
@misc{kim2025sureguidedposteriorsampling,
title={SURE Guided Posterior Sampling: Trajectory Correction for Diffusion-Based Inverse Problems},
author={Minwoo Kim and Hongki Lim},
year={2025},
eprint={2512.23232},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/2512.23232},
}
```
