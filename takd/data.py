"""COCO single as GPU-resident uint8 tensors.

The benchmark transform is deterministic: every image is resized to 224x224 with
PIL bicubic (what ``torchvision.transforms.functional.resize`` calls for PIL
input), and a horizontal flip is the only training augmentation. Each image is
therefore resized once, cached as uint8, and kept on the GPU (about 0.6 GB for
all splits). A training step only gathers, flips and normalizes on the device.

This matters for the experiment: the reference runner decoded JPEGs with
``num_workers=0`` and was input-bound (~76 samples/s on the A5000), which would
hide the teacher-side savings of masking in the measured training time.
"""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
PATCH = 16
GRID = IMAGE_SIZE // PATCH
NUM_PATCHES = GRID * GRID
CACHE_VERSION = 1
SPLITS = ("train", "val", "test")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(root, relative):
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe dataset path: {relative}")
    return Path(root) / path


def _load_row(root, row):
    from PIL import Image

    with Image.open(_safe(root, row["image"])) as source:
        image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    with Image.open(_safe(root, row["mask"])) as source:
        mask = source.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    pixels = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
    # Same rule as the reference diagnostics: a patch is foreground when at least
    # half of its pixels lie inside the object mask.
    fraction = (np.asarray(mask, dtype=np.float32) / 255.0).reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))
    return pixels, (fraction.reshape(-1) >= 0.5)


def build_cache(root, cache_path, workers=8):
    """Resize every manifest image once; reuse the cache while the manifest hash matches."""
    root, cache_path = Path(root), Path(cache_path)
    manifest_path = root / "manifest.json"
    manifest_sha = sha256(manifest_path)
    if cache_path.is_file():
        header = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
        if header.get("version") == CACHE_VERSION and header.get("manifest_sha256") == manifest_sha:
            print(f"REUSE data cache {cache_path}", flush=True)
            return cache_path
        print(f"Rebuilding stale data cache {cache_path}", flush=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = {}
    for split in SPLITS:
        rows = [row for row in manifest["images"] if row["split"] == split]
        if not rows:
            raise ValueError(f"COCO manifest has no {split} images")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            loaded = list(pool.map(lambda row: _load_row(root, row), rows))
        splits[split] = {
            "images": torch.from_numpy(np.stack([pixels for pixels, _ in loaded])),
            "foreground": torch.from_numpy(np.stack([fg for _, fg in loaded])),
            "labels": torch.tensor([int(row["label"]) for row in rows], dtype=torch.long),
            "probe": torch.tensor([bool(row.get("probe")) for row in rows], dtype=torch.bool),
            "ids": [row["image"] for row in rows],
        }
        print(f"  cached {split}: {len(rows)} images", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    torch.save({"version": CACHE_VERSION, "manifest_sha256": manifest_sha,
                "classes": list(manifest["classes"]), "splits": splits}, temporary)
    os.replace(temporary, cache_path)
    return cache_path


@dataclass
class Split:
    name: str
    images: torch.Tensor      # uint8 (N, 3, 224, 224) on device
    labels: torch.Tensor      # int64 (N,)
    foreground: torch.Tensor  # bool (N, 196)
    ids: list

    def __len__(self):
        return self.labels.numel()


@dataclass
class Dataset:
    classes: list
    manifest_sha256: str
    train: Split
    val: Split
    test: Split


def load_cache(cache_path, device):
    blob = torch.load(cache_path, map_location="cpu", weights_only=True)
    if blob.get("version") != CACHE_VERSION:
        raise ValueError(f"Unsupported data cache version in {cache_path}; rerun prepare")
    splits = {name: Split(name, value["images"].to(device), value["labels"].to(device),
                          value["foreground"].to(device), list(value["ids"]))
              for name, value in blob["splits"].items()}
    return Dataset(list(blob["classes"]), blob["manifest_sha256"], splits["train"], splits["val"], splits["test"])


def synthetic_dataset(device, classes=10, sizes=(160, 40, 40), seed=0):
    """Small learnable stand-in for smoke tests.

    Each class tints a band of four patch rows with its own RGB offset (the
    "object", also used as the foreground mask); the band position varies per
    image, so the class is carried by content, which tiny ViTs learn quickly.
    """
    generator = torch.Generator().manual_seed(seed)
    tints = torch.randint(20, 120, (classes, 3), generator=generator, dtype=torch.int32)
    splits = {}
    for name, n in zip(SPLITS, sizes):
        labels = torch.arange(n) % classes
        images = torch.randint(0, 96, (n, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator, dtype=torch.int32)
        rows = torch.randint(0, GRID - 3, (n,), generator=generator)
        foreground = torch.zeros(n, NUM_PATCHES, dtype=torch.bool)
        for i, (label, row) in enumerate(zip(labels.tolist(), rows.tolist())):
            images[i, :, row * PATCH:(row + 4) * PATCH, :] += tints[label].view(3, 1, 1)
            foreground[i, row * GRID:(row + 4) * GRID] = True
        splits[name] = Split(name, images.clamp(0, 255).to(torch.uint8).to(device), labels.to(device),
                             foreground.to(device), [f"synthetic/{name}/{i}" for i in range(n)])
    return Dataset([f"class_{c}" for c in range(classes)], "synthetic", splits["train"], splits["val"], splits["test"])


class Normalizer:
    def __init__(self, device):
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255
        self.inv_std = 1.0 / (torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255)

    def __call__(self, images, flip=None):
        x = images.float()
        if flip is not None:
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
        return (x - self.mean) * self.inv_std


def train_batches(split, batch_size, normalizer, seed, epoch):
    """Shuffled batches with per-sample flips; the order depends only on (seed, epoch)."""
    device = split.labels.device
    generator = torch.Generator(device=device).manual_seed(seed * 1_000_003 + epoch)
    order = torch.randperm(len(split), device=device, generator=generator)
    flips = torch.rand(len(split), device=device, generator=generator) < 0.5
    for start in range(0, len(split), batch_size):
        index = order[start:start + batch_size]
        yield normalizer(split.images[index], flips[start:start + batch_size]), split.labels[index]


def eval_batches(split, batch_size, normalizer):
    for start in range(0, len(split), batch_size):
        stop = start + batch_size
        yield (normalizer(split.images[start:stop]), split.labels[start:stop],
               split.foreground[start:stop], slice(start, stop))
