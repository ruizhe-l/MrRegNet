from itertools import combinations
import numpy as np
from scipy import ndimage

from .process_methods import Resize

class RandomAugmentation:

    def __init__(self, img_suffix, lab_suffix, aug_rate=0.5):
        self.img_suffix = img_suffix
        self.lab_suffix = lab_suffix
        self.aug_rate = aug_rate

    def __call__(self, data_dict):
        r = np.random.rand()
        if r > self.aug_rate:
            return data_dict
        
        img = data_dict[self.img_suffix]
        lab = data_dict[self.lab_suffix]

        methods = [Rotate(np.random.randint(360), [0, 1]), flipup, fliplr]
        comb = []
        for r in range(1,len(methods)+1):
            comb += list(combinations(methods, r))
        selected = np.random.choice(comb, 1)[0]

        for m in selected:
            img = m(img, is_mask=False)
            lab = m(lab, is_mask=True)
        
        data_dict.update({self.img_suffix: img, 
                          self.lab_suffix: lab})
        
        return data_dict


class Rotate:
    def __init__(self, angle, ax):
        self.angle = angle
        self.ax = ax
    def __call__(self, img, is_mask):
        order = 0 if is_mask else 3
        img = ndimage.rotate(img, self.angle, self.ax, reshape=False, prefilter=True, order=order)
        return img

def flipup(img, is_mask):
    return np.flip(img, 0).copy()

def fliplr(img, is_mask):
    return np.flip(img, 1).copy()

class RandomCrop():
    def __init__(self, img_suffix, lab_suffix, tar_size, seg_suffix=None, resize=None):
        self.img_suffix = img_suffix
        self.lab_suffix = lab_suffix
        self.tar_size = tar_size
        self.seg_suffix = seg_suffix
        self.resize = resize

    def __call__(self, data_dict):
        imgo = data_dict[self.img_suffix].copy()
        mean_ind = np.mean(imgo)

        xs = np.random.randint(imgo.shape[0] - self.tar_size[0] + 1) 
        ys = np.random.randint(imgo.shape[1] - self.tar_size[1] + 1)
        img = imgo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]] 
        while np.mean(img) < mean_ind / 10:
            xs = np.random.randint(imgo.shape[0] - self.tar_size[0] + 1) 
            ys = np.random.randint(imgo.shape[1] - self.tar_size[1] + 1)
            img = imgo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]] 

        if self.resize is not None and self.tar_size != self.resize:
            img = Resize(self.resize)(img)
        data_dict.update({self.img_suffix: img})

        if self.seg_suffix is not None:
            seg = data_dict[self.seg_suffix]
            seg = seg[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]]   
            data_dict.update({self.seg_suffix: seg}) 
        
        return data_dict  

class RandomForegroundCrop():
    def __init__(self, img_suffix, seg_suffix, tar_size, foreground_ratio):
        self.img_suffix = img_suffix
        self.seg_suffix = seg_suffix
        self.tar_size = tar_size
        self.foreground_ratio = foreground_ratio

    def __call__(self, data_dict):
        imgo = data_dict[self.img_suffix].copy()
        labo = data_dict[self.seg_suffix].copy()
        count_fore = np.sum(labo==255)

        xs = np.random.randint(imgo.shape[0] - self.tar_size[0] + 1) 
        ys = np.random.randint(imgo.shape[1] - self.tar_size[1] + 1)
        img = imgo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]] 
        lab = labo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]]
        while np.sum(lab==255) < count_fore * self.foreground_ratio:
            xs = np.random.randint(imgo.shape[0] - self.tar_size[0] + 1) 
            ys = np.random.randint(imgo.shape[1] - self.tar_size[1] + 1)
            img = imgo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]] 
            lab = labo[xs:xs+self.tar_size[0], ys:ys+self.tar_size[1]]

        data_dict.update({self.img_suffix: img})

        return data_dict  
