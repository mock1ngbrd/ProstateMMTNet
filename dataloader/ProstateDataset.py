"""Load paired ultrasound volumes and optional segmentation masks from NPZ files."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PD1C_B_E(Dataset):
    """Read B-mode (Z,Y,X) and SE (Z,Y,X,C) arrays for binary classification."""

    def __init__(self, case_list, root_path, transform=None):
        self.case_list = list(case_list)
        self.root_path = Path(root_path)
        self.transform = transform

    def _load_sample(self, index):
        case_id = self.case_list[index]
        with np.load(self.root_path / f"{case_id}.npz", allow_pickle=True) as data:
            label = data["label"]
            sample = {
                "name": case_id,
                "volume1": data["volume1"].transpose(2, 1, 0),
                "volume2": data["volume2"].transpose(3, 2, 1, 0),
                "cspca": data["CsPCa"] if label != 0 else 0,
            }
        return sample

    def __getitem__(self, index):
        sample = self._load_sample(index)
        return self.transform(sample) if self.transform else sample

    def __len__(self):
        return len(self.case_list)


class PD1C_B_E_mask(PD1C_B_E):
    """Add a matching binary mask to an annotated positive sample."""

    def __init__(self, case_list, root_path, mask_root_path, transform=None):
        super().__init__(case_list, root_path, transform)
        self.mask_root_path = Path(mask_root_path)

    def __getitem__(self, index):
        sample = self._load_sample(index)
        with np.load(self.mask_root_path / f"{sample['name']}.npz") as data:
            sample["mask"] = data["mask"].transpose(2, 1, 0).astype(np.int8)
        return self.transform(sample) if self.transform else sample


class ToTensor:
    """Add B-mode/mask channel dimensions and convert arrays to tensors."""

    def __init__(self, mask_prefix=False):
        self.mask_prefix = mask_prefix

    def __call__(self, sample):
        sample["volume1"] = torch.from_numpy(np.expand_dims(sample["volume1"], 0).copy())
        sample["volume2"] = torch.from_numpy(sample["volume2"].copy())
        if self.mask_prefix:
            sample["mask"] = torch.from_numpy(np.expand_dims(sample["mask"], 0).copy())
        return sample
