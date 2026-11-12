"""Independent PatchCore-style implementation for the Group 23 pilot.

Feature construction follows the patch alignment/pooling design documented in
amazon-science/patchcore-inspection; no upstream source is vendored here.
The pilot uses maximum L2 patch distance, without image-score reweighting.
"""
import io
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


def transform_image(path, view, size=256):
    with Image.open(path) as source:
        image = source.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    if view.startswith('brightness_'):
        image = ImageEnhance.Brightness(image).enhance(float(view.split('_')[1]))
    elif view.startswith('jpeg_'):
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=int(view.split('_')[1]), subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = compressed.convert('RGB').copy()
    elif view.startswith('blur_'):
        # Explicit PIL Gaussian approximation; radius is its standard deviation.
        image = image.filter(ImageFilter.GaussianBlur(float(view.split('_')[1])))
    elif view != 'clean':
        raise ValueError(view)
    return image


class Extractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                  backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, image):
        x = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)[None]
        x = (x.to(self.mean.device) - self.mean) / self.std
        low = self.layer2(self.stem(x))
        high = self.layer3(low)
        target = low.shape[-2:]
        descriptors = []
        for features in (low, high):
            b, channels, h, w = features.shape
            patches = F.unfold(features, kernel_size=3, padding=1)
            # Interpolate each channel/patch-offset plane to the layer-2 grid.
            patches = patches.reshape(b, channels * 9, h, w)
            if (h, w) != target:
                patches = F.interpolate(patches, size=target, mode='bilinear', align_corners=False)
            patches = patches.permute(0, 2, 3, 1).reshape(-1, channels * 9)
            descriptors.append(F.adaptive_avg_pool1d(patches[:, None], 1024)[:, 0])
        return F.adaptive_avg_pool1d(torch.stack(descriptors, dim=1).reshape(-1, 1, 2048), 1024)[:, 0].contiguous()


def squared_distances(x, y):
    return (x.square().sum(1)[:, None] + y.square().sum(1)[None] - 2 * (x @ y.T)).clamp_min_(0)


def nearest(x, bank, chunk=2048):
    return torch.cat([squared_distances(part, bank).min(1).values for part in x.split(chunk)])


def greedy_indices(pool, count):
    if count < 1 or count > len(pool):
        raise ValueError('Invalid bank budget')
    mean = pool.mean(0)
    index = int(((pool - mean) ** 2).sum(1).argmin())
    chosen = []
    used = torch.zeros(len(pool), dtype=torch.bool, device=pool.device)
    distances = torch.full((len(pool),), float('inf'), device=pool.device)
    for _ in range(count):
        chosen.append(index)
        used[index] = True
        torch.minimum(distances, ((pool - pool[index]) ** 2).sum(1), out=distances)
        index = int(distances.masked_fill(used, -1).argmax())
    return torch.tensor(chosen, device=pool.device)


def anchored_bank(groups, clean_order, budget, rho, adaptive):
    n_anchor = math.floor(rho * budget)
    anchor = groups[0][clean_order[:n_anchor]]
    candidates = torch.cat(groups[1:])
    split = len(groups[1])
    # Exact descriptor duplicates are removed before grouping by the caller.
    distances = nearest(candidates, anchor)
    used = torch.zeros(len(candidates), dtype=torch.bool, device=candidates.device)
    chosen = []
    allocation = [0, 0]
    for step in range(budget - n_anchor):
        eligible = [bool((~used[:split]).any()), bool((~used[split:]).any())]
        if not any(eligible):
            raise ValueError('Insufficient distinct augmented candidates')
        if adaptive:
            errors = [float(distances[:split].mean()), float(distances[split:].mean())]
            group = max((g for g in range(2) if eligible[g]), key=lambda g: (errors[g], -g))
        else:
            group = min((g for g in range(2) if eligible[g]), key=lambda g: (allocation[g], g))
        start, stop = (0, split) if group == 0 else (split, len(candidates))
        index = start + int(distances[start:stop].masked_fill(used[start:stop], -1).argmax())
        chosen.append(index)
        used[index] = True
        allocation[group] += 1
        torch.minimum(distances, ((candidates - candidates[index]) ** 2).sum(1), out=distances)
    return torch.cat([anchor, candidates[chosen]]), [n_anchor, *allocation]


def choose_validated(options, baseline, reference_error, tolerance=1.1):
    eligible = [o for o in options if o['coverage'][0] <= tolerance * max(reference_error, 1e-12)]
    if not eligible:
        return baseline, {'fallback': True, 'rho': None, 'allocation': [len(baseline), 0, 0]}
    selected = min(eligible, key=lambda o: (max(o['coverage'][1:]), -o['rho']))
    return selected['bank'], {k: v for k, v in selected.items() if k != 'bank'} | {'fallback': False}


def calibrated_threshold(scores):
    rank = math.ceil(.95 * (len(scores) + 1))
    return (float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float('inf')), rank


def score_features(features, bank, neighbours=9):
    """Patch distances and Eq. (7) image score; support includes the nearest item."""
    matrix = squared_distances(features, bank)
    values, indices = matrix.min(1)
    patch = values.sqrt()
    index = int(patch.argmax())
    nearest_id = int(indices[index])
    support_dist = squared_distances(bank[nearest_id:nearest_id+1], bank)[0]
    # Explicit inclusion avoids numerical self-distance/tie ambiguity.
    support_dist[nearest_id] = -1
    support = support_dist.argsort(stable=True)[:min(neighbours, len(bank))]
    distances = torch.linalg.vector_norm(features[index] - bank[support], dim=1)
    weight = 1 - torch.softmax(distances, dim=0)[0]
    return patch, float(weight * patch[index])


def unanchored_bank(groups, budget):
    """Mean-nearest initialization; worst coverage allocation over all 3 groups."""
    pool = torch.cat(groups)
    first = int(((pool - pool.mean(0)) ** 2).sum(1).argmin())
    selected = [first]
    used = torch.zeros(len(pool), dtype=torch.bool, device=pool.device)
    used[first] = True
    distances = ((pool - pool[first]) ** 2).sum(1)
    bounds = np.cumsum([0] + [len(g) for g in groups]).tolist()
    allocation = [int(bounds[g] <= first < bounds[g+1]) for g in range(len(groups))]
    for _ in range(budget-1):
        eligible = [g for g in range(len(groups)) if bool((~used[bounds[g]:bounds[g+1]]).any())]
        group = max(eligible, key=lambda g: (float(distances[bounds[g]:bounds[g+1]].mean()), -g))
        lo, hi = bounds[group:group+2]
        index = lo + int(distances[lo:hi].masked_fill(used[lo:hi], -1).argmax())
        selected.append(index); used[index] = True; allocation[group] += 1
        torch.minimum(distances, ((pool-pool[index])**2).sum(1), out=distances)
    return pool[selected], allocation
