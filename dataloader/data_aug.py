"""Paired spatial augmentation for B-mode, SE, and optional masks."""
import numpy as np
from scipy.ndimage import rotate


class RandomRotateTransform(object):

    def __init__(self, mask_prefix=False, angle_range=(-10, 10), p_per_sample=1.0):
        self.p_per_sample = p_per_sample
        self.angle_range = angle_range
        self.mask_prefix = mask_prefix

    def __call__(self, sample):
        volume1 = sample['volume1']
        volume2 = sample['volume2']
        if np.random.uniform() < self.p_per_sample:
            rand_angle = np.random.randint(self.angle_range[0], self.angle_range[1])
            volume1 = rotate(volume1, angle=rand_angle, axes=(-2, -3), reshape=False, order=1)
            volume2 = rotate(volume2, angle=rand_angle, axes=(-2, -3), reshape=False, order=1)
            if self.mask_prefix:
                mask = sample['mask']
                mask = rotate(mask, angle=rand_angle, axes=(-2, -3), reshape=False, order=1)
                sample['mask'] = mask
        sample['volume1'] = volume1
        sample['volume2'] = volume2
        return sample


class MirrorTransform(object):

    def __init__(self, mask_prefix=False, axes=(0, 1, 2), p=0.5):
        self.axes = axes
        self.mask_prefix = mask_prefix
        self.p = p

    def __call__(self, sample):
        volume1 = sample['volume1']
        volume2 = sample['volume2']
        if self.mask_prefix:
            mask = sample['mask']
        if isinstance(self.axes, int):
            if np.random.uniform() < self.p:
                volume1 = np.flip(volume1, self.axes)
                volume2 = np.flip(volume2, self.axes)
                if self.mask_prefix:
                    mask = np.flip(mask, self.axes)
        else:
            for axis in self.axes:
                if np.random.uniform() < self.p:
                    volume1 = np.flip(volume1, axis=axis)
                    volume2 = np.flip(volume2, axis=axis)
                    if self.mask_prefix:
                        mask = np.flip(mask, axis=axis)
        sample['volume1'] = volume1
        sample['volume2'] = volume2
        if self.mask_prefix:
            sample['mask'] = mask
        return sample
