#AUTOGENERATED! DO NOT EDIT! File to edit: dev/fiducial_detect.ipynb (unless otherwise specified).

__all__ = ['DotVisionCheckerDLDetector']

#Cell
import math
import warnings

import torch
from skimage.measure import label, regionprops
from torchvision import transforms

from .utils import *

#Cell
class DotVisionCheckerDLDetector():
    def __init__(self, file_model, cuda=False):
        model = torch.jit.load(file_model.as_posix())
        model = model.cuda() if cuda else model.cpu()
        model = model.eval()

        self.cuda  = cuda
        self.model = model

    def format_arr(self, arr):
        assert_allclose(len(arr.shape), 2)      # Grayscale check
        if arr.min() < 0: warnings.warn('Value less than zero detected')
        if arr.max() > 1: warnings.warn('Value greater than 1 detected')

        arr = arr.float()                       # Must be float
        arr = imresize(arr, 384)                # Network trained on grayscale 384 sized images
        arr = rescale(arr, (0, 1), (-1, 1))     # Network trained on images between [-1,1]
        arr = arr[None, None]                   # Add batch and channel dimension
        if self.cuda: arr = arr.cuda()          # Possibly run on gpu
        return arr

    def get_mask(self, arr):
        model = self.model
        with torch.no_grad():
            mask = model(self.format_arr(arr))  # Inference
            mask = mask.argmax(dim=1)           # Convert from scores to labels
            mask = mask.squeeze(0)              # Remove batch dimension
        return mask

    def __call__(self, arr):
        mask = self.get_mask(arr)

        # Extract fiducial points from mask
        ps_f = arr.new_full((4,2), math.nan)
        for idx, p_f in enumerate(ps_f):
            regions = regionprops(label(mask == (idx+1)))
            if len(regions) > 0:
                region = regions[arr.new_tensor([r.area for r in regions]).argmax()]
                ps_f[idx] = arr.new_tensor(reverse(region.centroid))
        ps_f *= (shape(arr)/shape(mask)).mean()

        return ps_f