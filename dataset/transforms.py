import torch
import os
import numpy as np
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import random
from scipy.ndimage import rotate

class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self):
        self.axes = (0, 1, 2)

    def __call__(self, m, random_flip):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for index in range(len(self.axes)):
            axis = self.axes[index]
            flip_prob = random_flip[index]
            if flip_prob > 0.5:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    for c in range(m.shape[0]):
                        m[c] = np.flip(m[c], axis)

        return m

class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """
    def __init__(self):
        self.axes = [(1, 0), (2, 1), (2, 0)]

    def __call__(self, m, k, axis):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        axis = self.axes[axis]
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, axis)
        else:
            for c in range(m.shape[0]):
                m[c] = np.rot90(m[c], k, axis)

        return m

class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, angle_spectrum=20, axes=None):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0
        self.angle_spectrum = angle_spectrum
        self.axes = axes

    def __call__(self, m, axis, angle):
        # axis = self.axes[self.random_state.randint(len(self.axes))]
        # angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)
        axis = self.axes[axis]
        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=0, mode='constant', cval=0)
        else:
            for c in range(m.shape[0]):
                m[c] = rotate(m[c], angle, axes=axis, reshape=False, order=0, mode='constant', cval=0)

        return m

