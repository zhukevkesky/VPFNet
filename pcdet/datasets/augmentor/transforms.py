from __future__ import division
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample, sample2):
        for t in self.transforms:
            sample, sample2 = t(sample, sample2)

        return sample, sample2


class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __call__(self, sample, sample2):
        left = np.transpose(sample, (2, 0, 1))  # [3, H, W]
        sample = torch.from_numpy(left) / 255.
        right = np.transpose(sample2, (2, 0, 1))
        sample2 = torch.from_numpy(right) / 255.

        return sample, sample2


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample, sample2):

        # Images have converted to tensor, with shape [C, H, W]
        for t, m, s in zip(sample, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(sample2, self.mean, self.std):
            t.sub_(m).div_(s)
        return sample, sample2


class ToPILImage(object):

    def __call__(self, sample, sample2):
        sample = Image.fromarray(sample.astype('uint8'))
        sample2 = Image.fromarray(sample2.astype('uint8'))

        return sample, sample2


class ToNumpyArray(object):

    def __call__(self, sample, sample2):
        sample = np.array(sample).astype(np.float32)
        sample2 = np.array(sample2).astype(np.float32)

        return sample, sample2


# Random coloring
class RandomContrast(object):
    """Random contrast"""

    def __call__(self, sample, sample2):
        if np.random.random() < 0.5:
            contrast_factor = np.random.uniform(0.8, 1.2)

            sample = F.adjust_contrast(sample, contrast_factor)
            sample2 = F.adjust_contrast(sample2, contrast_factor)

        return sample, sample2


class RandomGamma(object):

    def __call__(self, sample, sample2):
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet

            sample = F.adjust_gamma(sample, gamma)
            sample2 = F.adjust_gamma(sample2, gamma)

        return sample, sample2


class RandomBrightness(object):

    def __call__(self, sample, sample2):
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.5, 2.0)

            sample = F.adjust_brightness(sample, brightness)
            sample2 = F.adjust_brightness(sample2, brightness)

        return sample, sample2


class RandomHue(object):

    def __call__(self, sample, sample2):
        if np.random.random() < 0.5:
            hue = np.random.uniform(-0.1, 0.1)

            sample = F.adjust_hue(sample, hue)
            sample2 = F.adjust_hue(sample2, hue)

        return sample, sample2


class RandomSaturation(object):

    def __call__(self, sample, sample2):
        if np.random.random() < 0.5:
            saturation = np.random.uniform(0.8, 1.2)

            sample = F.adjust_saturation(sample, saturation)
            sample2 = F.adjust_saturation(sample2, saturation)

        return sample, sample2


class RandomColor(object):

    def __call__(self, sample, sample2):
        transforms = [RandomContrast(),
                      RandomGamma(),
                      RandomBrightness(),
                      RandomHue(),
                      RandomSaturation()]

        # sample,sample2 = ToPILImage()(sample,sample2)

        if np.random.random() < 0.5:
            # A single transform
            t = random.choice(transforms)
            sample, sample2 = t(sample, sample2)
        else:
            # Combination of transforms
            # Random order
            random.shuffle(transforms)
            for t in transforms:
                sample, sample2 = t(sample, sample2)

        sample, sample2 = ToNumpyArray()(sample, sample2)

        return sample, sample2
