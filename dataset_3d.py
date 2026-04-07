from pathlib import Path
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


def load_nifti(path):
    return nib.load(str(path)).get_fdata().astype(np.float32)


def normalize_ct(volume, clip_min=-100, clip_max=400):
    volume = np.clip(volume, clip_min, clip_max)
    volume = (volume - clip_min) / (clip_max - clip_min)
    return volume.astype(np.float32)


def random_patch_3d(volume, mask, patch_size=(64, 128, 128), tumor_oversample_prob=0.5):
    d, h, w = volume.shape
    pd, ph, pw = patch_size

    if d < pd or h < ph or w < pw:
        pad_d = max(0, pd - d)
        pad_h = max(0, ph - h)
        pad_w = max(0, pw - w)
        volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        mask = np.pad(mask, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        d, h, w = volume.shape

    if random.random() < tumor_oversample_prob and np.any(mask == 2):
        coords = np.argwhere(mask == 2)
        cz, cy, cx = coords[random.randint(0, len(coords) - 1)]
        z1 = np.clip(cz - pd // 2, 0, d - pd)
        y1 = np.clip(cy - ph // 2, 0, h - ph)
        x1 = np.clip(cx - pw // 2, 0, w - pw)
    else:
        z1 = random.randint(0, d - pd)
        y1 = random.randint(0, h - ph)
        x1 = random.randint(0, w - pw)

    vol_patch = volume[z1:z1+pd, y1:y1+ph, x1:x1+pw]
    mask_patch = mask[z1:z1+pd, y1:y1+ph, x1:x1+pw]
    return vol_patch, mask_patch


def resolve_task_root(data_root):
    root = Path(data_root)
    direct = root / 'Task03_Liver'
    if direct.exists():
        return direct
    if root.name == 'Task03_Liver':
        return root
    raise FileNotFoundError(
        f"Could not resolve Task03_Liver from path: {root}. Pass either the parent folder containing Task03_Liver or the Task03_Liver folder itself."
    )


class Task03Liver3DDataset(Dataset):
    def __init__(self, data_root, file_names, patch_size=(64, 128, 128), samples_per_volume=2):
        root = resolve_task_root(data_root)
        self.images_dir = root / 'imagesTr'
        self.labels_dir = root / 'labelsTr'
        self.file_names = [f for f in file_names if not Path(f).name.startswith('._')]
        self.patch_size = tuple(patch_size)
        self.samples_per_volume = samples_per_volume

    def __len__(self):
        return len(self.file_names) * self.samples_per_volume

    def __getitem__(self, idx):
        case_name = self.file_names[idx % len(self.file_names)]
        img_path = self.images_dir / case_name
        lbl_path = self.labels_dir / case_name

        image = load_nifti(img_path)
        mask = load_nifti(lbl_path).astype(np.int64)
        image = normalize_ct(image)
        image, mask = random_patch_3d(image, mask, self.patch_size)
        image = np.expand_dims(image, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)
