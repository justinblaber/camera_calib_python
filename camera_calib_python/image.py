#AUTOGENERATED! DO NOT EDIT! File to edit: dev/image.ipynb (unless otherwise specified).

__all__ = ['rgb2gray', 'Img', 'FileImg', 'File16bitImg', 'ArrayImg']

#Cell
import warnings

import numpy as np
from PIL import Image

from .utils import *

#Cell
def rgb2gray(arr): # From Pillow documentation
    return arr[:,:,0]*(299/1000) + arr[:,:,1]*(587/1000) + arr[:,:,2]*(114/1000)

#Cell
class Img:
    def exists(self): raise NotImplementedError('Please implement exists()')

    @property
    def name(self):   raise NotImplementedError('Please implement name')
    @property
    def size(self):   raise NotImplementedError('Please implement size')
    @property
    def array(self):  raise NotImplementedError('Please implement array')
    @property
    def array_gs(self): # gs == "gray scale"
        arr = self.array
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            arr = rgb2gray(arr)
        elif len(arr.shape) == 2:
            pass
        else:
            raise RuntimeError(f'Dont know how to handle array of shape: {arr.shape}')
        return arr

#Cell
class FileImg(Img):
    def __init__(self, file_img):
        self.file_img = file_img

    def exists(self): return self.file_img.exists()

    @property
    def name(self):   return self.file_img.stem
    @property
    def size(self):   return reverse(Image.open(self.file_img).size) # fast

#Cell
class File16bitImg(FileImg):
    def __init__(self, file_img):
        super().__init__(file_img)

    @property
    def array(self):
        arr = np.array(Image.open(self.file_img), dtype=np.float)
        arr /= 2**16 # Scale between 0 and 1 for 16 bit image
        return arr

#Cell
class ArrayImg(Img):
    def __init__(self, arr, name=None):
        if len(arr.shape) < 2: raise RuntimeError('Input array has less than 2 dimensions')
        self.sz = np.array(arr.shape[:2])
        self.n = name

        assert_allclose(arr.dtype, np.float)
        if arr.min() < 0: warnings.warn('Value less than 0 found')
        if arr.max() > 1: warnings.warn('Value greater than 1 found')
        self.arr = arr

    @property
    def name(self):   return self.n
    @property
    def size(self):   return self.sz
    @property
    def array(self):  return self.arr

    def exists(self): return True