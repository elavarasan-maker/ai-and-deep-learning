from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from .config import IMAGE_SIZE, SELECTED_BAND_INDICES, PATCH_FIRE_FRACTION_THRESHOLD, SPLITS, SEED


def list_files(data_dir, split):
    files = []
    for scene in SPLITS[split]:
        folder = Path(data_dir) / scene
        if not folder.exists():
            raise FileNotFoundError(f"Missing {folder}. Download/extract Sen2Fire first.")
        files.extend(sorted(folder.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"No NPZ files found for split={split} in {data_dir}")
    return files


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        required = {"image", "label"}
        if not required.issubset(keys):
            raise ValueError(f"{path}: expected NPZ keys {required}, found {keys}")
        image = np.asarray(z["image"])
        label = np.asarray(z["label"])
    if image.ndim != 3:
        raise ValueError(f"{path}: image must be 3D, got {image.shape}")
    # Published Sen2Fire data stores image as (12,H,W); tolerate H,W,C too.
    if image.shape[0] in (12, 13):
        image = np.moveaxis(image[:12], 0, -1)
    elif image.shape[-1] in (12, 13):
        image = image[..., :12]
    else:
        raise ValueError(f"{path}: cannot identify 12-band axis from {image.shape}")
    if label.ndim == 3 and label.shape[0] == 1:
        label = label[0]
    if label.ndim == 3 and label.shape[-1] == 1:
        label = label[..., 0]
    if label.ndim != 2:
        raise ValueError(f"{path}: label must be 2D, got {label.shape}")
    if label.shape != image.shape[:2]:
        raise ValueError(f"{path}: image/label spatial mismatch {image.shape[:2]} vs {label.shape}")
    return image.astype(np.float32), label


def label_fraction(label):
    valid = np.isfinite(label) & (label != 255)
    if not valid.any():
        return 0.0
    return float(np.count_nonzero(label[valid] == 1) / np.count_nonzero(valid))


def patch_label(label):
    return int(label_fraction(label) >= PATCH_FIRE_FRACTION_THRESHOLD)


def make_features(image):
    x = image[..., SELECTED_BAND_INDICES]
    # Sentinel-2 reflectance may be stored as scaled integers; retain scale consistently.
    if np.nanpercentile(x, 99) > 2.0:
        x = x / 10000.0
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return x.astype(np.float32)


def compute_nbr(image):
    # image is H,W,12 in B1..B12 order; NIR=B8 index 7, SWIR2=B12 index 11.
    nir = image[..., 7].astype(np.float32)
    swir2 = image[..., 11].astype(np.float32)
    if np.nanpercentile(np.abs(nir), 99) > 2.0:
        nir, swir2 = nir / 10000.0, swir2 / 10000.0
    return (nir - swir2) / (nir + swir2 + 1e-6)


def get_train_stats(data_dir, max_files=None):
    files = list_files(data_dir, "train")
    if max_files:
        files = files[:max_files]
    total = np.zeros(5, np.float64)
    total2 = np.zeros(5, np.float64)
    count = 0
    for p in files:
        image, _ = load_npz(p)
        x = image[..., SELECTED_BAND_INDICES].astype(np.float32)
        if np.nanpercentile(x, 99) > 2.0:
            x /= 10000.0
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # sample every 8th pixel to make stats computation tractable.
        x = x[::8, ::8, :].reshape(-1, 5)
        total += x.sum(axis=0)
        total2 += np.square(x).sum(axis=0)
        count += len(x)
    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(total2 / max(count, 1) - mean ** 2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


class PatchSequence(tf.keras.utils.Sequence):
    def __init__(self, files, mean, std, batch_size=8, training=False, shuffle=False):
        self.files = list(files)
        self.mean = np.asarray(mean, np.float32).reshape(1, 1, 5)
        self.std = np.asarray(std, np.float32).reshape(1, 1, 5)
        self.batch_size = batch_size
        self.training = training
        self.shuffle = shuffle
        self.indices = np.arange(len(self.files))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.default_rng(SEED).shuffle(self.indices)

    def __getitem__(self, index):
        ids = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        xs, ys = [], []
        for i in ids:
            image, label = load_npz(self.files[i])
            x = (make_features(image) - self.mean) / self.std
            y = patch_label(label)
            if self.training:
                k = np.random.randint(0, 4)
                x = np.rot90(x, k)
                if np.random.rand() < 0.5: x = np.flip(x, axis=0)
                if np.random.rand() < 0.5: x = np.flip(x, axis=1)
                if np.random.rand() < 0.5:
                    x = x * np.random.uniform(0.85, 1.15)
            xs.append(np.ascontiguousarray(x, dtype=np.float32))
            ys.append(y)
        return np.stack(xs), np.asarray(ys, dtype=np.float32)
