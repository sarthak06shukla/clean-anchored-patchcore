"""Reproducible, normal-only fitting followed by locked bottle test evaluation."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from torch.nn import functional as F

from core import (Extractor, transform_image, greedy_indices, anchored_bank,
                  nearest, choose_validated, calibrated_threshold, score_features, unanchored_bank)

METHODS = ['clean', 'pooled', 'uniform', 'adaptive', 'no_guard', 'no_anchor']

def save_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def stamp():
    return datetime.now(timezone.utc).isoformat()

def build_manifest(data, output, config):
    category = config['category']
    files = sorted((data / category / 'train' / 'good').glob('*.png'))
    expected = {'bottle': (209,83), 'hazelnut': (391,110), 'tile': (230,117)}[category]
    if len(files) != expected[0]:
        raise ValueError(f'Incorrect {category} training count: {len(files)}')
    order = np.random.default_rng(config['seed']).permutation(len(files))
    nf = int(len(files) * config['fit_fraction'])
    nv = int(len(files) * config['validation_fraction'])
    splits = {'fit': order[:nf], 'validation': order[nf:nf+nv], 'calibration': order[nf+nv:]}
    manifest = {key: [files[int(i)].relative_to(data).as_posix() for i in indices]
                for key, indices in splits.items()}
    manifest['test'] = [p.relative_to(data).as_posix() for p in sorted((data/category/'test').glob('*/*.png'))]
    if len(manifest['test']) != expected[1]:
        raise ValueError(f'Incorrect {category} test count')
    sets = [set(v) for v in manifest.values()]
    assert all(not (sets[i] & sets[j]) for i in range(4) for j in range(i))
    # Verify every source image and anomalous mask before fitting; labels never tune a bank.
    for key, paths in manifest.items():
        for name in paths:
            with Image.open(data/name) as im:
                im.verify()
            p = Path(name)
            if key == 'test' and p.parent.name != 'good':
                mask = data/category/'ground_truth'/p.parent.name/(p.stem+'_mask.png')
                with Image.open(mask) as im:
                    assert np.asarray(im).max() > 0
    save_json(output/'split_manifest.json', manifest)
    return manifest

def collect(extractor, data, paths, views, cap, seed, cache, size):
    if cache.exists():
        return np.load(cache)
    per_image = (size // 8) ** 2
    total = len(paths) * len(views) * per_image
    selected = np.sort(np.random.default_rng(seed).choice(total, min(cap, total), replace=False))
    blocks = []
    index = 0
    for view in views:
        for name in paths:
            lo, hi = index * per_image, (index + 1) * per_image
            subset = selected[(selected >= lo) & (selected < hi)] - lo
            if len(subset):
                features = extractor(transform_image(data/name, view, size))
                blocks.append(features[torch.from_numpy(subset).to(features.device)].cpu().numpy())
            index += 1
            if index % 40 == 0:
                print(f'Features {cache.stem}: {index}/{len(paths)*len(views)} images', flush=True)
    result = np.concatenate(blocks)
    np.save(cache, result)
    return result

def fit(extractor, data, output, cache, config, manifest, device):
    sync()
    started = time.perf_counter()
    groups = []
    group_views = [['clean'], ['brightness_0.85', 'brightness_1.15'], ['jpeg_85']]
    seen = set()
    retained = []
    for g, views in enumerate(group_views):
        array = collect(extractor, data, manifest['fit'], views,
                        config['candidate_cap_per_group'], config['seed']+g,
                        cache/f'fit_{g}.npy', config['image_size'])
        keep = []
        for i, row in enumerate(array):
            digest = hashlib.sha256(row.tobytes()).digest()
            if digest not in seen:
                seen.add(digest)
                keep.append(i)
        retained.append(len(keep))
        groups.append(torch.from_numpy(array[keep]).to(device))
    print(f'Unique candidates per group: {retained}', flush=True)
    validation = [torch.from_numpy(collect(extractor, data, manifest['validation'], views,
                  10**9, config['seed'], cache/f'validation_{g}.npy', config['image_size'])).to(device)
                  for g, views in enumerate(group_views)]
    feature_seconds = time.perf_counter() - started
    budget = config['budget']
    times, banks, choices = {}, {}, {}
    sync(); t = time.perf_counter()
    clean_order = greedy_indices(groups[0], budget)
    banks['clean'] = groups[0][clean_order]
    sync(); times['clean'] = time.perf_counter() - t
    reference = float(nearest(validation[0], banks['clean']).mean())
    sync(); t = time.perf_counter()
    pooled = torch.cat(groups)
    banks['pooled'] = pooled[greedy_indices(pooled, budget)]
    sync(); times['pooled'] = time.perf_counter() - t
    del pooled
    print('Clean and pooled banks built', flush=True)
    for method in ['uniform', 'adaptive']:
        options = []
        sync(); t = time.perf_counter()
        for rho in config['anchor_fractions']:
            bank, allocation = anchored_bank(groups, clean_order, budget, rho, method == 'adaptive')
            coverage = [float(nearest(v, bank).mean()) for v in validation]
            options.append({'bank': bank, 'rho': rho, 'allocation': allocation, 'coverage': coverage})
            print(f'{method} rho={rho}: allocation={allocation}; coverage={coverage}', flush=True)
        banks[method], choices[method] = choose_validated(options, banks['clean'], reference, config['clean_tolerance'])
        choices[method]['options'] = [{k:v for k,v in o.items() if k != 'bank'} for o in options]
        sync(); times[method] = time.perf_counter() - t
        if method == 'adaptive':
            selected = min(options, key=lambda o: (max(o['coverage'][1:]), -o['rho']))
            banks['no_guard'] = selected['bank']
            choices['no_guard'] = {k:v for k,v in selected.items() if k != 'bank'} | {'fallback':False}
            times['no_guard'] = times['adaptive']
    sync(); t = time.perf_counter()
    bank, allocation = unanchored_bank(groups, budget)
    coverage = [float(nearest(v, bank).mean()) for v in validation]
    fallback = coverage[0] > config['clean_tolerance'] * max(reference, 1e-12)
    banks['no_anchor'] = banks['clean'] if fallback else bank
    choices['no_anchor'] = {'allocation': [budget,0,0] if fallback else allocation,
                            'candidate_allocation': allocation, 'coverage':coverage, 'fallback':fallback}
    sync(); times['no_anchor'] = time.perf_counter()-t
    # Calibration occurs only after normal-validation decisions are locked.
    cal_scores = {m: [] for m in METHODS}
    for name in manifest['calibration']:
        features = extractor(transform_image(data/name, 'clean', config['image_size']))
        for method, bank in banks.items():
            cal_scores[method].append(score_features(features, bank)[1])
    thresholds = {}
    for method in METHODS:
        threshold, rank = calibrated_threshold(cal_scores[method])
        thresholds[method] = {'value': threshold if np.isfinite(threshold) else None, 'rank': rank,
                              'n': len(cal_scores[method])}
        np.save(output/f'bank_{method}.npy', banks[method].cpu().numpy())
    save_json(output/'calibration_scores.json', cal_scores)
    info = {'fitting_completed_utc': stamp(), 'reference_clean_coverage': reference,
            'unique_candidates': retained, 'choices': choices, 'thresholds': thresholds,
            'bank_construction_seconds': times, 'feature_preparation_seconds': feature_seconds,
            'construction_timing_note': 'Uniform/adaptive include two rho builds and validation; clean anchor prefix is reused. Feature preparation excluded.',
            'descriptor_bank_bytes': budget*config['descriptor_dim']*4,
            'peak_cuda_allocated_bytes_through_fitting': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None}
    save_json(output/'fit_summary.json', info)
    return banks, info

def evaluate(extractor, data, output, config, manifest, banks, fit_info, feature_cache):
    size = config['image_size']
    labels = np.array([Path(p).parent.name != 'good' for p in manifest['test']], dtype=np.uint8)
    masks = []
    for name, label in zip(manifest['test'], labels):
        p = Path(name)
        if label:
            path = data/config['category']/'ground_truth'/p.parent.name/(p.stem+'_mask.png')
            with Image.open(path) as im:
                masks.append(np.array(im.convert('L').resize((size,size), Image.Resampling.NEAREST)) > 0)
        else:
            masks.append(np.zeros((size,size), dtype=bool))
    truth = np.stack(masks).reshape(-1)
    all_metrics, all_scores = [], []
    # Keep predetermined first image per defect type for visual inspection.
    defects = sorted({Path(p).parent.name for p in manifest['test']})
    examples = {next(p for p in manifest['test'] if Path(p).parent.name == defect) for defect in defects}
    for view in config['test_views']:
        checkpoint = output/f'evaluation_{view}.json'
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            all_metrics.extend(saved['metrics']); all_scores.extend(saved['scores'])
            continue
        maps = {m: [] for m in METHODS}
        grids = {m: [] for m in METHODS}
        scores = {m: [] for m in METHODS}
        image_records = []
        t = time.perf_counter()
        for i, name in enumerate(manifest['test']):
            image = transform_image(data/name, view, size)
            cache_path = feature_cache / view / (name.replace('/', '__')+'.npy')
            if cache_path.exists():
                features = torch.from_numpy(np.load(cache_path)).to(next(extractor.parameters()).device)
            else:
                features = extractor(image)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, features.cpu().numpy())
            for method, bank in banks.items():
                distances, score = score_features(features, bank)
                scores[method].append(score)
                grid = distances.reshape(1,1,size//8,size//8)
                grids[method].append(grid[0,0].cpu().numpy())
                anomaly_map = F.interpolate(grid, size=(size,size), mode='bilinear', align_corners=False)[0,0].cpu().numpy()
                anomaly_map = gaussian_filter(anomaly_map, sigma=config['map_smoothing_sigma'])
                maps[method].append(anomaly_map)
                threshold = fit_info['thresholds'][method]['value']
                image_records.append({'method': method, 'condition': view, 'file': name,
                    'label': int(labels[i]), 'score': score,
                    'prediction': int(score > (threshold if threshold is not None else float('inf')))})
                if name in examples and view in ['clean', 'brightness_0.45']:
                    dest = output/'examples'
                    dest.mkdir(exist_ok=True)
                    key = f'{Path(name).parent.name}_{view}'
                    np.save(dest/f'{key}_{method}.npy', anomaly_map)
                    if method == 'clean':
                        image.save(dest/f'{key}.png')
                        Image.fromarray(masks[i].astype(np.uint8)*255).save(dest/f'{key}_mask.png')
            if (i+1)%20 == 0:
                print(f'Evaluate {view}: {i+1}/{len(labels)}', flush=True)
        condition_metrics = []
        for method in METHODS:
            score_array = np.asarray(scores[method])
            threshold = fit_info['thresholds'][method]['value']
            predictions = score_array > (threshold if threshold is not None else float('inf'))
            pixels = np.stack(maps[method]).reshape(-1)
            metric = {'method': method, 'condition': view,
                      'image_auroc': float(roc_auc_score(labels, score_array)),
                      **pixel_metrics(truth, pixels),
                      'fpr': float(predictions[labels==0].mean()),
                      'recall': float(predictions[labels==1].mean())}
            condition_metrics.append(metric)
        np.savez_compressed(output/f'patch_scores_{view}.npz', **{m:np.stack(grids[m]) for m in METHODS})
        save_json(checkpoint, {'metrics': condition_metrics, 'scores': image_records,
                              'elapsed_seconds_including_metrics': time.perf_counter()-t})
        all_metrics.extend(condition_metrics); all_scores.extend(image_records)
        print(f'Completed {view}: '+str([(m['method'], round(m['image_auroc'],4)) for m in condition_metrics]), flush=True)
    for name, records in [('metrics.csv',all_metrics),('per_image_scores.csv',all_scores)]:
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    # Separate warm batch-one end-to-end latency, including image read and maps.
    latencies = {}
    example = data/manifest['calibration'][0]
    for method, bank in banks.items():
        samples = []
        for i in range(15):
            sync(); begin = time.perf_counter()
            features = extractor(transform_image(example, 'clean', size))
            d, image_score = score_features(features, bank)
            result = F.interpolate(d.reshape(1,1,size//8,size//8), size=(size,size), mode='bilinear',align_corners=False)
            gaussian_filter(result[0,0].cpu().numpy(), sigma=config['map_smoothing_sigma'])
            sync()
            if i >= 5:
                samples.append((time.perf_counter()-begin)*1000)
        latencies[method] = {'median_ms': float(np.median(samples)), 'samples_ms': samples}
    save_json(output/'latency.json', latencies)
    save_json(output/'completion.json', {'completed_utc': stamp(), 'test_normal': int((labels==0).sum()),
        'test_anomalous': int(labels.sum()), 'conditions': len(config['test_views']),
        'metrics_rows': len(all_metrics), 'score_rows': len(all_scores)})


def pixel_metrics(truth, scores):
    """Exact tied-score AUROC and AP, sharing one sort of all pixels."""
    order = np.argsort(scores, kind='stable')[::-1]
    y = np.asarray(truth, dtype=np.uint8)[order]
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores)-1]
    tp = np.cumsum(y, dtype=np.int64)[ends]
    fp = (ends+1)-tp
    positives = int(tp[-1]); negatives = len(scores)-positives
    recall = tp/positives
    precision = tp/(tp+fp)
    ap = np.sum(np.diff(np.r_[0.,recall])*precision)
    auc = np.trapezoid(np.r_[0.,recall], np.r_[0.,fp/negatives])
    return {'pixel_auroc':float(auc), 'pixel_ap':float(ap)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--cache', type=Path, default=Path('cache'))
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--feature-cache', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    cache_key = {k:config[k] for k in ['category','seed','image_size','candidate_cap_per_group','fit_fraction','validation_fraction','backbone','weights','fit_views','dataset_revision']}
    core_hash = hashlib.sha256(Path(__file__).with_name('core.py').read_bytes()).hexdigest()
    cache_key['core_sha256'] = core_hash
    cache_lock = args.cache/'cache_config.json'
    if cache_lock.exists() and json.loads(cache_lock.read_text()) != cache_key:
        raise ValueError('Feature cache configuration mismatch')
    save_json(cache_lock, cache_key)
    args.feature_cache.mkdir(parents=True, exist_ok=True)
    test_lock = args.feature_cache/'identity.json'
    test_identity = {k:cache_key[k] for k in ['category','image_size','backbone','weights','dataset_revision','core_sha256']}
    if test_lock.exists() and json.loads(test_lock.read_text()) != test_identity:
        raise ValueError('Test feature cache identity mismatch')
    save_json(test_lock,test_identity)
    lock = args.output/'config_locked.json'
    if lock.exists() and json.loads(lock.read_text()) != config:
        raise ValueError('Changed configuration: use a new output AND cache directory')
    save_json(lock, config)
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    versions = {p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','Pillow','scipy','scikit-learn']}
    save_json(args.output/'environment.json', {'started_utc': stamp(), 'platform': platform.platform(),
        'python': platform.python_version(), 'device': str(device),
        'gpu': torch.cuda.get_device_name() if device.type=='cuda' else None, 'versions': versions})
    manifest = build_manifest(args.data, args.output, config)
    print('Splits: '+str({k:len(v) for k,v in manifest.items()}), flush=True)
    extractor = Extractor(device)
    print('Feature extractor ready', flush=True)
    if (args.output/'fit_summary.json').exists():
        fit_info = json.loads((args.output/'fit_summary.json').read_text())
        banks = {m:torch.from_numpy(np.load(args.output/f'bank_{m}.npy')).to(device) for m in METHODS}
    else:
        banks, fit_info = fit(extractor,args.data,args.output,args.cache,config,manifest,device)
    print('Fitting and calibration locked. Starting test evaluation.', flush=True)
    evaluate(extractor,args.data,args.output,config,manifest,banks,fit_info,args.feature_cache)
    print('EXPERIMENT COMPLETE', flush=True)

if __name__ == '__main__':
    main()
