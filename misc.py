"""SGPS data loading, measurement reuse and saved-PNG evaluation."""


import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


MODEL_PRESETS = {'ffhq256ddpm': {'model_config': {'attention_resolutions': 16, 'channel_mult': '', 'class_cond': False, 'dropout': 0.0, 'image_size': 256, 'learn_sigma': True, 'model_path': 'checkpoints/ffhq256.pt', 'num_channels': 128, 'num_head_channels': 64, 'num_heads': 4, 'num_heads_upsample': -1, 'num_res_blocks': 1, 'resblock_updown': True, 'use_checkpoint': False, 'use_fp16': False, 'use_new_attention_order': False, 'use_scale_shift_norm': True}, 'name': 'ddpm'}}


def sha256_file(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    os.replace(temporary, path)


def configure_sgps_determinism(seed, device):
    import random
    import sys
    # The legacy Windows conda CUDA package puts NVRTC builtins in env/bin.
    # NVRTC loads that DLL lazily (e.g. for complex abs in phase retrieval).
    if os.name == 'nt':
        cuda_bin = Path(sys.executable).parent / 'bin'
        if cuda_bin.is_dir():
            os.environ['PATH'] = str(cuda_bin) + os.pathsep + os.environ.get('PATH', '')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def prepare_sgps_data(dataset, manifest_path, image_ids):
    import hashlib
    if not dataset.data:
        raise ValueError('No input images found')
    if any(not path.stem.isdigit() for path in dataset.data):
        raise ValueError('Use numeric source-ID filenames, e.g. 00003.png')
    by_id = {int(path.stem): path for path in dataset.data}
    if len(by_id) != len(dataset.data):
        raise ValueError('Duplicate source image IDs')
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        records = manifest['records']
        digest = hashlib.sha256(json.dumps(records, sort_keys=True,
                                           separators=(',', ':')).encode('utf-8')).hexdigest()
        if digest != manifest['manifest_sha256']:
            raise ValueError('Manifest digest does not match its records')
        if digest == '4c13b44282618be96470f0d226049195956c170f2a317c11acf01f4d7e3ab3a7':
            raise ValueError('Excluded historical non-FFHQ manifest')
        ids = [int(record['source_image_id']) for record in records]
        if set(ids) != set(by_id) or len(ids) != len(set(ids)):
            raise ValueError('Input directory does not exactly match manifest source IDs')
        for record in records:
            path = by_id[int(record['source_image_id'])]
            if sha256_file(path) != record['sha256']:
                raise ValueError(f'Source content hash mismatch: {path}')
        if len({record['sha256'] for record in records}) != len(records):
            raise ValueError('Duplicate image content in manifest')
    else:
        ids = sorted(by_id)
        manifest = None
    dataset.data = [by_id[source_id] for source_id in ids]
    selected = ids.copy()
    if image_ids is not None:
        selected = []
        for group in image_ids.split(','):
            bounds = group.strip().split('-')
            if len(bounds) == 1:
                selected.append(int(bounds[0]))
            elif len(bounds) == 2:
                first, last = map(int, bounds)
                selected.extend(range(first, last + 1))
            else:
                raise ValueError('Invalid --image_ids')
        if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(ids):
            raise ValueError('Image IDs must be unique and present in the input directory')
    return ids, selected, manifest


@torch.no_grad()
def prepare_sgps_measurement(args, operator, images, source_ids, manifest):
    # Importing main here is safe: it contains no executable work on import.
    from main import StatelessRNG
    clean = operator(images)
    if args.measurement_source:
        if args.manifest is None:
            raise ValueError('--manifest is required with --measurement_source')
        source = args.measurement_source.resolve()
        metadata = json.loads((source / 'measurement_metadata.json').read_text(encoding='utf-8'))
        if (metadata['manifest_sha256'] != manifest['manifest_sha256'] or
                metadata['base_seed'] != args.seed or metadata['task'] != operator.name or
                abs(metadata['sigma_y'] - operator.sigma) > 1e-12):
            raise ValueError('Frozen measurement provenance/config mismatch')
        # Only load trusted checkpoints/packages. weights_only prevents arbitrary
        # Python object unpickling for these plain tensor dictionaries.
        package = torch.load(source / 'measurement_package.pt', map_location='cpu', weights_only=True)
        source_order = metadata['source_image_ids']
        if len(source_order) != len(set(source_order)) or not set(source_ids).issubset(source_order):
            raise ValueError('Frozen measurement source IDs are missing/duplicated')
        for key, hash_key in (('clean_measurement', 'clean_measurement_hash'),
                              ('measurement', 'measurement_hash'), ('noise', 'noise_hash')):
            if tensor_sha256(package[key]) != metadata[hash_key]:
                raise ValueError(f'Frozen tensor hash mismatch: {key}')
        indices = [source_order.index(source_id) for source_id in source_ids]
        saved_clean = package['clean_measurement'][indices].to(images.device)
        if saved_clean.shape != clean.shape or not torch.isfinite(saved_clean).all():
            raise ValueError('Invalid frozen measurement shape/values')
        difference = float((clean - saved_clean).abs().max().item())
        tolerance = 3e-6 if operator.name == 'nonlinear_blur' else 2e-6
        if difference > tolerance:
            raise ValueError(f'Operator does not reproduce frozen clean measurement: max diff={difference}')
        measurement = package['measurement'][indices].to(images.device)
        if not torch.isfinite(measurement).all():
            raise ValueError('Non-finite frozen observations')
        return measurement
    noise = torch.cat([StatelessRNG(args.seed, operator.name, sample_offset=source_id).randn_like(
        clean[index:index + 1], 'measurement_noise') for index, source_id in enumerate(source_ids)]) * operator.sigma
    measurement = clean + noise
    return measurement


def _png_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0


def evaluate_png_psnr(reconstruction, ground_truth):
    # Same float32 difference / float64 mean as skimage PSNR(data_range=1),
    # which the DPS compute_metrics evaluation uses for saved RGB PNGs.
    difference = _png_array(ground_truth) - _png_array(reconstruction)
    mse = np.mean(difference ** 2, dtype=np.float64)
    return float(10 * np.log10(1.0 / mse))


def evaluate_saved_lpips(rows, root, device):
    from importlib.metadata import version
    import lpips
    if version('lpips') != '0.1.4':
        raise RuntimeError('Install the official lpips==0.1.4 package')
    metric = lpips.LPIPS(net='vgg', version='0.1').to(device).eval().requires_grad_(False)
    with torch.no_grad():
        for row in rows:
            # Phase retrieval selects by PSNR only. Never choose an independent
            # LPIPS-best reconstruction to populate the same row.
            if not row['selected_by_psnr']:
                continue
            pred = torch.from_numpy(_png_array(root / row['reconstruction'])).permute(2, 0, 1).contiguous().unsqueeze(0)
            gt = torch.from_numpy(_png_array(root / row['ground_truth'])).permute(2, 0, 1).contiguous().unsqueeze(0)
            value = float(metric((pred * 2 - 1).to(device), (gt * 2 - 1).to(device)).item())
            if not np.isfinite(value):
                raise ValueError('Non-finite LPIPS')
            row['LPIPS'] = value


def write_metric_tables(rows, root):
    selected = [row for row in rows if row['selected_by_psnr']]
    ids = [row['source_image_id'] for row in selected]
    if not selected or len(ids) != len(set(ids)):
        raise ValueError('Missing/duplicate selected reconstructions')
    per_image = []
    for row in selected:
        value = {k: v for k, v in row.items() if k != 'selected_by_psnr'}
        # Phase retrieval includes the time spent on all four trials.
        value['seconds'] = sum(r['seconds'] for r in rows
                               if r['source_image_id'] == row['source_image_id'])
        per_image.append(value)
    tables = [('per_image.csv', per_image)]
    if len(rows) != len(selected):
        tables.append(('per_trial.csv', rows))
    summary = {'N': len(selected), 'seconds_mean': float(np.mean([r['seconds'] for r in per_image]))}
    for metric in ('PSNR', 'LPIPS'):
        values = [row[metric] for row in selected]
        if all(value is None for value in values):
            summary[metric + '_mean'] = summary[metric + '_sample_SD'] = None
        else:
            if any(value is None for value in values) or not np.isfinite(values).all():
                raise ValueError(f'Incomplete/non-finite {metric} results')
            summary[metric + '_mean'] = float(np.mean(values))
            summary[metric + '_sample_SD'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    tables.append(('summary.csv', [summary]))
    for filename, values in tables:
        with (root / filename).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    return summary


def resolve_checkpoint_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return str(path.resolve())
    name = path.name
    project = Path(__file__).resolve().parent
    for candidate in (project / 'checkpoints' / name, project.parent / 'checkpoints' / name):
        if candidate.exists():
            return str(candidate)
    return str(project / 'checkpoints' / name)
