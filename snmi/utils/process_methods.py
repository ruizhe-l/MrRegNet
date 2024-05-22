import cv2
import torch
import numpy as np
import scipy.ndimage as nd
from skimage.exposure import match_histograms

""" The library of process methods.  
    non-parametered methods can be defined like a funtion (e.g. min_max)
    parametered methods should be defined as a funtional class (e.g. ExpandDim)
"""

# Normalization---------------------------------------
def min_max(x):
    _x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return _x

def zero_mean(x):
    _x = (x - np.mean(x)) / np.std(x)
    return _x

def median_mean(x):
    _x = (x - np.median(x)) / np.std(x)
    return _x


# Image-----------------------------------------------
def rgb2gray(x):
    assert x.shape[-1] == 3, 'Not RGB image!'
    _x = np.dot(x[...,:3], [0.299, 0.587, 0.114])
    return _x

def channel_last2first(x):
    dims = len(x.shape)
    return np.transpose(x, [dims-1] + list(range(dims-1)))

def channel_first2last(x):
    dims = len(x.shape)
    return np.transpose(x, list(range(1, dims)) + [0])

class ToTensor:
    def __init__(self, axis=None):
        self.axis = axis
    
    def __call__(self, x):
        if self.axis is not None:
            x = np.expand_dims(x, self.axis)
        return torch.Tensor(x)

class ExpandDim:
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, x):
        return np.expand_dims(x, self.axis)

class Transpose:
    def __init__(self, axis):
        self.axis = axis
    
    def __call__(self, x):
        return np.transpose(x, self.axis)

class Resize:
    def __init__(self, tar_size, nearest=False):
        self.tar_size = tar_size
        self.nearest = nearest
    
    def __call__(self, x):
        if len(self.tar_size) == 2:
            return self.resize2d(x)
        elif len(self.tar_size) == 3:
            return self.resize3d(x)
        else:
            raise ValueError(f'Wrong target size {self.tar_size}, should be [x,y] or [x,y,z]')
            
    def resize2d(self, x):
        tar_size = tuple(self.tar_size)
        if self.nearest:
            _x = cv2.resize(x, tar_size[::-1], interpolation=cv2.INTER_NEAREST)
        else:
            _x = cv2.resize(x, tar_size[::-1])
        return _x

    def resize3d(self, x):
        if self.tar_size is None:
            return x
        sx = x.shape[0]
        sy = x.shape[1]
        sz = x.shape[2]
        zoom = [self.tar_size[0] / sx, self.tar_size[1] / sy, self.tar_size[2] / sz]
        if self.nearest:
            _x = nd.zoom(x, zoom=zoom, order=0)
        else:
            _x = nd.zoom(x, zoom=zoom)
        assert np.all(_x.shape[0:3] == np.array(self.tar_size)), \
            'Fail to resize 3d image: expect {}, got {}.'.format(self.tar_size, _x.shape[0:3])
        return _x

class Crop:
    def __init__(self, crop_list):
        self.crop_list = crop_list

    def __call__(self, x):
        clist = self.crop_list
        if len(clist) == 4:
            return x[clist[0]:clist[1], clist[2]:clist[3]]
        elif len(clist) == 6:
            return x[clist[0]:clist[1], clist[2]:clist[3], clist[4]:clist[5]]
        else:
            raise ValueError(f'Wrong crop list {clist}, should be [x1,x2,y1,y2] or [x1,x2,y1,y2,z1,z2]')

class HistogramMatching:
    def __init__(self, template):
        self.template = template

    def __call__(self, source):
        matched = match_histograms(source, self.template)

        return matched
            
# Label-----------------------------------------------
class OneHot:
    def __init__(self, n_classes, dtype=np.float32):
        self.n_classes = n_classes
        self.dtype = dtype

    def __call__(self, x):
        nc = list(range(self.n_classes)) if type(self.n_classes) is int else self.n_classes
        if type(nc) in [list, tuple, np.ndarray]:
            x_ = np.zeros([len(nc)] + list(x.shape), self.dtype)
            for i in range(len(nc)):
                x_[i, ...][x == nc[i]] = 1
        else:
            raise Exception('Wrong type of n_class.')
        return x_    
    