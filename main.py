"""SURE Guided Posterior Sampling for pixel-space inverse problems.

SURE differentiates with respect to the guided image through the frozen
denoiser. See README.md for the data and evaluation protocol.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn


class Scheduler:
    """The legacy linear-sigma, poly-7 schedule (including its arithmetic)."""

    def __init__(self, num_steps, sigma_max=100, sigma_min=0.1, sigma_final=0):
        if num_steps < 1 or not 0 < sigma_min <= sigma_max or sigma_final < 0:
            raise ValueError("Invalid noise schedule")
        self.num_steps = int(num_steps)
        self.sigma_max = sigma_max
        steps = np.linspace(0, 1, self.num_steps)
        time_fn = lambda r: (sigma_max ** (1 / 7) + r *
                             (sigma_min ** (1 / 7) - sigma_max ** (1 / 7))) ** 7
        self.time_steps = np.append(np.array([time_fn(s) for s in steps]), sigma_final)
        self.sigma_steps = self.time_steps.copy()
        self.factor_steps = [max(2 * self.time_steps[i] *
                                 (self.time_steps[i] - self.time_steps[i + 1]), 0)
                             for i in range(self.num_steps)]


class StatelessRNG:
    """Component-keyed Gaussian draws, identical to the revision experiment."""

    def __init__(self, base_seed, task_name, trial_id=0, sample_offset=0):
        self.base_seed = int(base_seed)
        self.task_name = str(task_name)
        self.trial_id = int(trial_id)
        self.sample_offset = int(sample_offset)

    def _seed(self, stream_name, sample_id, outer_step=-1, inner_step=-1, trace_index=-1):
        key = '|'.join(map(str, (self.base_seed, self.task_name, self.trial_id,
                                stream_name, sample_id, outer_step, inner_step, trace_index)))
        digest = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
        return int.from_bytes(digest, byteorder='little', signed=False) % (2 ** 63 - 1)

    def randn_like(self, ref, stream_name, outer_step=-1, inner_step=-1, trace_index=-1):
        chunks = []
        for local_id in range(ref.shape[0]):
            generator = torch.Generator(device=ref.device)
            generator.manual_seed(self._seed(stream_name, self.sample_offset + local_id,
                                             outer_step, inner_step, trace_index))
            chunks.append(torch.randn(ref[local_id:local_id + 1].shape,
                                      device=ref.device, dtype=ref.dtype, generator=generator))
        return torch.cat(chunks, dim=0)


def pca_noise_level(batch, patch_size=8):
    """Patch-covariance scale, without fitting/calibration or synthetic noise."""
    values = []
    for image in batch:
        patches = torch.nn.functional.unfold(image.unsqueeze(0), kernel_size=patch_size)
        patches = patches.squeeze(0).transpose(0, 1)
        if patches.shape[0] <= 1:
            raise ValueError("PCA requires at least two patches")
        centered = patches - patches.mean(dim=0)
        covariance = centered.T.mm(centered) / (centered.shape[0] - 1)
        remaining = torch.linalg.eigvalsh(covariance).detach().cpu().tolist()
        while len(remaining) > 1:
            mean = sum(remaining) / len(remaining)
            median = statistics.median(remaining)
            if abs(mean - median) < 1e-6:
                break
            remaining.remove(max(remaining))
        values.append(max(sum(remaining) / len(remaining), 0.0) ** 0.5)
    return torch.tensor(values, device=batch.device, dtype=batch.dtype)


class LangevinDynamics:
    def __init__(self, num_steps=100, lr=1e-4, tau=0.01, lr_min_ratio=0.01):
        self.num_steps = int(num_steps)
        self.lr, self.tau, self.lr_min_ratio = float(lr), float(tau), float(lr_min_ratio)
        if self.num_steps < 1 or self.lr <= 0 or self.tau <= 0 or self.lr_min_ratio <= 0:
            raise ValueError("Invalid Langevin configuration")

    def sample(self, x0hat, operator, measurement, sigma, ratio, rng, outer_step):
        lr = (1 + ratio * (self.lr_min_ratio - 1)) * self.lr
        with torch.enable_grad():
            x = x0hat.clone().detach().requires_grad_(True)
            optimizer = torch.optim.SGD([x], lr)
            for inner_step in range(self.num_steps):
                optimizer.zero_grad()
                # loss() is the modern name for legacy Operator.error().
                loss = operator.loss(x, measurement).sum() / (2 * self.tau ** 2)
                loss = loss + ((x - x0hat.detach()) ** 2).sum() / (2 * sigma ** 2)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    epsilon = rng.randn_like(x, 'langevin', outer_step, inner_step)
                    x.data = x.data + np.sqrt(2 * lr) * epsilon
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"Non-finite Langevin state at step {outer_step}")
        return x.detach()


class SGPS(nn.Module):
    """Image-wise posterior sampling with Langevin guidance and MC-SURE correction."""

    def __init__(self, T=16, alpha=0.5, epsilon_scale=1e-3, patch_size=8,
                 langevin_config=None):
        super().__init__()
        self.annealing_scheduler = Scheduler(T, sigma_max=100, sigma_min=0.1)
        self.lgvd = LangevinDynamics(**(langevin_config or {}))
        self.alpha, self.epsilon_scale, self.patch_size = alpha, epsilon_scale, patch_size
        if not np.isfinite(alpha) or epsilon_scale <= 0 or patch_size < 1:
            raise ValueError("Invalid SURE configuration")

    def get_start(self, ref, base_seed, task_name, trial_id=0, sample_offset=0):
        rng = StatelessRNG(base_seed, task_name, trial_id, sample_offset)
        return rng.randn_like(ref, 'initial_noise') * self.annealing_scheduler.sigma_max

    def _reverse_diffusion(self, model, xt, sigma):
        # Direct tweedie(xt, sigma) would change floating-point operations.
        scheduler = Scheduler(1, sigma_max=sigma, sigma_min=0.01)
        score = model.score(xt, scheduler.sigma_steps[0])
        return xt + scheduler.factor_steps[0] * score * 0.5

    def _sure_gradient(self, model, x_pre, sigma_hat, rng, outer_step):
        epsilon = x_pre.detach().amax() * self.epsilon_scale
        if not torch.isfinite(epsilon) or epsilon.item() <= 0:
            raise RuntimeError(f"Invalid MC-SURE epsilon: {epsilon.item()}")
        with torch.enable_grad():
            x_in = x_pre.detach().requires_grad_(True)
            x_hat = model.tweedie(x_in, sigma_hat)
            sigma_squared = sigma_hat ** 2
            probe = rng.randn_like(x_pre, 'sure_probe', outer_step=outer_step, trace_index=0)
            perturbed_sigma = torch.maximum(sigma_hat, epsilon.detach().expand_as(sigma_hat))
            perturbed = model.tweedie(x_in + epsilon * probe, perturbed_sigma)
            trace_estimate = torch.dot(probe.flatten(), (perturbed - x_hat).flatten()) / epsilon
            sure = (torch.sum((x_hat - x_in) ** 2) - x_pre[0].numel() * sigma_squared
                    + 2 * sigma_squared * trace_estimate).sum()
            gradient = torch.autograd.grad(sure, x_in, retain_graph=False, create_graph=False)[0]
        return gradient

    @torch.no_grad()
    def sample(self, model, x_start, operator, measurement, base_seed=42,
               task_name=None, trial_id=0, sample_offset=0):
        if x_start.shape[0] != 1:
            raise ValueError("SGPS reproduction requires batch_size=1")
        rng = StatelessRNG(base_seed, task_name or operator.name, trial_id, sample_offset)
        xt = x_start
        for outer_step in range(self.annealing_scheduler.num_steps):
            sigma = self.annealing_scheduler.sigma_steps[outer_step]
            x0hat = self._reverse_diffusion(model, xt, sigma)
            x_pre = self.lgvd.sample(x0hat, operator, measurement, sigma,
                                    outer_step / self.annealing_scheduler.num_steps, rng, outer_step)
            sigma_hat = pca_noise_level(x_pre, self.patch_size)
            gradient = self._sure_gradient(model, x_pre, sigma_hat, rng, outer_step)
            x_post = x_pre - self.alpha * gradient
            xt = x_post + rng.randn_like(x_post, 'renoise', outer_step=outer_step) * \
                self.annealing_scheduler.sigma_steps[outer_step + 1]
            if not torch.isfinite(xt).all():
                raise RuntimeError(f"Non-finite SGPS state at step {outer_step}")
        return xt


TASK_CONFIGS = {
    'down_sampling': ({'name': 'down_sampling', 'resolution': 256, 'scale_factor': 4}, 1e-4),
    'inpainting': ({'name': 'inpainting', 'resolution': 256, 'mask_type': 'box',
                    'mask_len_range': [128, 129]}, 5e-5),
    'inpainting_rand': ({'name': 'inpainting', 'resolution': 256, 'mask_type': 'random',
                         'mask_prob_range': [0.7, 0.71]}, 1e-4),
    'gaussian_blur': ({'name': 'gaussian_blur', 'kernel_size': 61, 'intensity': 3.0}, 1e-4),
    'motion_blur': ({'name': 'motion_blur', 'kernel_size': 61, 'intensity': 0.5}, 5e-5),
    'phase_retrieval': ({'name': 'phase_retrieval', 'resolution': 256, 'oversample': 2.0}, 5e-5),
    'nonlinear_blur': ({'name': 'nonlinear_blur',
                        'opt_yml_path': 'forward_operator/bkse/options/generate_blur/default.yml'}, 5e-5),
    'hdr': ({'name': 'high_dynamic_range', 'scale': 2}, 2e-5),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(TASK_CONFIGS), required=True)
    parser.add_argument('--data', type=Path, required=True, help='Correct FFHQ RGB source directory')
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/ffhq256.pt'))
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrite a run')
    parser.add_argument('--manifest', type=Path, help='Reference manifest; required for frozen measurements')
    parser.add_argument('--measurement_source', type=Path, help='Directory with measurement_package.pt and metadata')
    parser.add_argument('--image_ids', help='Source IDs, e.g. 3-12 or 3,8,12; not subset-local indices')
    parser.add_argument('--T', type=int, choices=(16, 33), default=16)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bkse_config', type=Path, help='Optional nonlinear blur model options path')
    parser.add_argument('--evaluate', action='store_true', help='Official LPIPS 0.1.4 / VGG on saved RGB PNGs')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from data import ImageDataset
    from forward_operators import get_operator, use_deterministic_padding
    from model import get_model
    from misc import (MODEL_PRESETS, configure_sgps_determinism, prepare_sgps_data,
                      prepare_sgps_measurement, write_json,
                      evaluate_png_psnr, evaluate_saved_lpips, write_metric_tables)
    from torchvision.utils import save_image

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty; choose a new directory: {args.output}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    configure_sgps_determinism(args.seed, args.device)
    dataset = ImageDataset(root=args.data, resolution=256, device=args.device)
    source_ids, selected, manifest = prepare_sgps_data(dataset, args.manifest, args.image_ids)
    args.output.mkdir(parents=True, exist_ok=True)
    model_config = copy.deepcopy(MODEL_PRESETS['ffhq256ddpm'])
    model_config['model_config']['model_path'] = str(args.checkpoint.resolve())
    images = dataset.get_data(len(dataset), sigma=0)
    operator_config, lr = copy.deepcopy(TASK_CONFIGS[args.task])
    operator_config['sigma'] = 0.05
    if args.bkse_config is not None:
        if args.task != 'nonlinear_blur':
            raise ValueError('--bkse_config is only valid for nonlinear_blur')
        operator_config['opt_yml_path'] = str(args.bkse_config.resolve())
    device_config = {} if args.task == 'phase_retrieval' else {'device': args.device}
    operator = get_operator(**operator_config, **device_config)
    use_deterministic_padding(operator)
    measurement = prepare_sgps_measurement(args, operator, images, source_ids, manifest)
    model = get_model(**model_config, device=args.device).eval().requires_grad_(False)
    sampler = SGPS(T=args.T, alpha=args.alpha, langevin_config={'lr': lr})
    num_trials = 4 if args.task == 'phase_retrieval' else 1
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(source_ids=selected, num_trials=num_trials, operator=operator_config,
                  langevin={'num_steps': 100, 'lr': lr, 'tau': 0.01, 'lr_min_ratio': 0.01},
                  patch_size=8, epsilon_scale=1e-3)
    write_json(args.output / 'config.json', config)
    rows = []
    for source_id in selected:
        index = source_ids.index(source_id)
        image, y = images[index:index + 1], measurement[index:index + 1]
        gt_path = args.output / 'gt' / f'{source_id:05d}.png'
        gt_path.parent.mkdir(exist_ok=True)
        save_image((image + 1) / 2, gt_path)
        candidates = []
        for trial_id in range(num_trials):
            x_start = sampler.get_start(image, args.seed, operator.name, trial_id, source_id)
            if image.is_cuda:
                torch.cuda.synchronize(image.device)
            started = time.perf_counter()
            result = sampler.sample(model, x_start, operator, y, args.seed,
                                    operator.name, trial_id, source_id)
            if image.is_cuda:
                torch.cuda.synchronize(image.device)
            elapsed = time.perf_counter() - started
            recon_path = args.output / 'recon' / f'trial_{trial_id}' / f'{source_id:05d}.png'
            recon_path.parent.mkdir(parents=True, exist_ok=True)
            save_image((result + 1) / 2, recon_path)
            row = {'source_image_id': source_id, 'trial_id': trial_id,
                   'seconds': elapsed, 'PSNR': evaluate_png_psnr(recon_path, gt_path),
                   'LPIPS': None, 'reconstruction': str(recon_path.relative_to(args.output)),
                   'ground_truth': str(gt_path.relative_to(args.output))}
            candidates.append(row)
            print(f"ID {source_id:03d} trial {trial_id}: {elapsed:.3f}s, PSNR={row['PSNR']:.4f}", flush=True)
        best = max(candidates, key=lambda row: row['PSNR'])
        for row in candidates:
            row['selected_by_psnr'] = row is best
            rows.append(row)
        write_metric_tables(rows, args.output)
    del model
    if str(args.device).startswith('cuda'):
        torch.cuda.empty_cache()
    if args.evaluate:
        evaluate_saved_lpips(rows, args.output, args.device)
    summary = write_metric_tables(rows, args.output)
    message = (f"Mean: {summary['seconds_mean']:.3f}s/image, "
               f"PSNR={summary['PSNR_mean']:.4f}")
    if summary['LPIPS_mean'] is not None:
        message += f", LPIPS={summary['LPIPS_mean']:.4f}"
    print(message, flush=True)


if __name__ == '__main__':
    main()
