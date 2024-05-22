import os
import ssl
import cv2
import inspect
import numbers

import numpy as np
import nibabel as nib

from PIL import Image
import SimpleITK as sitk
    
    
def load_file(path, dtype=np.float32):
    # Since there are some errors for catching exception, now just use if
    path = path.strip()
    assert os.path.isfile(path), 'File {} not found!'.format(path)
    suffix = path.split('.')[-1]
    # Nifty load
    if suffix in ['nii.gz', 'nii', 'gz']:
        data = nib.load(path)
        return np.array(data.get_fdata()).astype(dtype)

    # text to ndarray
    if suffix == 'txt':
        data = np.genfromtxt(path, dtype)
        return data.astype(dtype)
    
    if suffix == 'npy':
        data = np.load(path)
        return data.astype(dtype)

    if suffix in ['mhd', 'raw']:
        itkimage = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(itkimage)
        return data.astype(dtype)

    # PIL load
    try:
        data = Image.open(path)
    except:
        pass
    else:
        return np.array(data, dtype)

    
    

    raise IOError('Invalid data type: {}'%path)


# dict ------------------------------------------------------

def dict_append(old_dict, new_dict):
    if new_dict is None:
        return old_dict
    if old_dict is None or not old_dict:
        for key in new_dict:
            old_dict[key] = []

    for key in old_dict:
        assert key in new_dict, 'No key "{}" in old dict!'.format(key)
        old_dict[key].append(new_dict[key])

    return old_dict

def dict_concat(old_dict, new_dict, axis=0):
    if new_dict is None:
        return old_dict
    if old_dict is None or not old_dict:
        old_dict = new_dict
    else:
        for key in old_dict:
            assert key in new_dict, 'No key "{}" in old dict!'.format(key)
            old_v = old_dict[key]
            new_v = new_dict[key]
            old_v = [old_v] if np.ndim(old_v) == 0 else old_v
            new_v = [new_v] if np.ndim(new_v) == 0 else new_v
            old_dict[key] = np.concatenate((old_v, new_v), axis)

    return old_dict


def dict_add(old_dict, new_dict):
    if new_dict is None:
        return old_dict
    if old_dict is None:
        old_dict = new_dict
    else:
        for key in new_dict:
            old_dict[key] += new_dict[key]
    return old_dict

def dict_mean(eval_dict, axis=0):
    if not eval_dict:
        return None
    o_d = {}
    for key in eval_dict:
        if key in ['__globals__','__header__', '__version__', 'name', 'img', 'image']:  # ignor header and image of dict
            continue
        value = eval_dict.get(key)
        if type(value) is dict or np.ndim(value) > 2:
            continue

        value = np.array(value)
        if value.size == 2: # calculate mean along batch
            mean = np.mean(value, axis) #[1:]
        else:
            mean = np.mean(value)
        
        o_d.update({key: mean})
    return o_d

def dict_to_str(eval_dict, axis=0):
    eval_dict_mean = dict_mean(eval_dict, axis)
    if not eval_dict_mean:
        return ''
    o_s = ''
    for key in eval_dict_mean:
        value = eval_dict_mean[key]
        if isinstance(value, numbers.Number):  # keep mean to one dimension array for prosess
            value = [value]
        value = ['%.4f'%v for v in value] # value to string
        o_s += '%s: '%key
        for s in value:
            o_s += '%s '%s
        o_s += '  '
    return o_s.strip()


def class_to_str(mclass, deep):
    prefix = '  ' * (deep+1)
    if inspect.isfunction(mclass):
        return f'{prefix}' + mclass.__name__
    cname = mclass.__class__.__name__
    kvs = [f'%s = %s' % (k,v) for k,v in mclass.__dict__.items()]
    return f'{prefix}{cname} -- \n{prefix}  ' + f'\n{prefix}  '.join(kvs) + f'\n{prefix}'


def kv_to_str(kv, deep):
    if kv is None:
        return f'\n None'
    if isinstance(kv, (numbers.Number, str)):
        return f'{kv}'
    if isinstance(kv, dict):
        return '\n' + config_to_str(kv,deep+1)
    if isinstance(kv, (tuple, list)):
        s = ''
        for v in kv:
            s += f'{kv_to_str(v,deep)} '
        return s
    return f'\n{class_to_str(kv,deep)}'
    

def config_to_str(config_dict, deep=0):
    s = ''
    prefix = '  '*deep
    for k, v in config_dict.items():
        if isinstance(k, str) and k.startswith('__'):
            continue
        s += f'{prefix}{kv_to_str(k,deep)}: {kv_to_str(v,deep)}\n'
    return s


def config_to_txt(config_dict, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(config_to_str(config_dict))


# image ------------------------------------------------------

def recale_array(array, nmin=None, nmax=None, tmin=0, tmax=255, dtype=np.uint8):
    array = np.array(array)
    if nmin is None:
        nmin = np.min(array)
    array = array - nmin
    if nmax is None:
        nmax = np.max(array) + 1e-9
    array = array / nmax
    array = (array * (tmax - tmin)) - tmin
    return array.astype(dtype)

def gray2rgb(img):
    return np.stack((img,)*3, axis=-1)

def reg2gray(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

def gray_img_blend(img, lab, colormap=cv2.COLORMAP_RAINBOW, weight=[0.7,0.3], channel_first=False):
    if channel_first:
        img = img.transpose(0, 2, 3, 1)
        lab = lab.transpose(0, 2, 3, 1)
    if np.max(lab) > 0:
        lab = lab / np.max(lab) 
    lab = np.array(lab * 255, np.uint8)
    comb = np.zeros(lab.shape[:-1] + (3, ))
    for b in range(img.shape[0]):
        tmp_img = gray2rgb(img[b, ..., 0]).astype(np.float32)
        tmp_lab = (cv2.applyColorMap(lab[b, ..., 0], colormap) / 255).astype(np.float32)
        tmp_lab[lab[b, ..., 0] == 0] = 0
        tmp_lab = np.array(tmp_lab, np.float32)
        tmp_comb = cv2.addWeighted(tmp_img, weight[0], tmp_lab, weight[1], 0)
        comb[b] = tmp_comb
    if channel_first:
        comb = comb.transpose(0, 3, 1, 2)
    return comb

def combine_2d_imgs_from_tensor(img_list, nmin=None, nmax=None, channel_first=False):
    imgs = []
    combined = None
    for im in img_list:
        assert len(im.shape) == 3 or len(im.shape) == 4 and im.shape[-3] in [1, 3], \
        'Only accept gray or rgb 2d images with shape [n, x, y] or  [n, x, y, c], where c = 1 (gray) or 3 (rgb).'
        if len(im.shape) == 4:
            if channel_first:
                im = im.transpose(0, 2, 3, 1)
            if im.shape[-1] != 3:
                im = im[..., 0]
                im = gray2rgb(im)
        else:
            im = gray2rgb(im)
            
        im = recale_array(im, nmin=nmin, nmax=nmax)
        im = im.reshape(-1, im.shape[-2], im.shape[-1])
        imgs.append(im)
    combined = np.concatenate(imgs, 1)
    return combined

def combine_3d_imgs_from_tensor(img_list, vis_slices, nmin=None, nmax=None, channel_first=False):
    tmp = img_list[0]
    if len(tmp.shape) == 5:
        tmp = tmp[:,0,...] if channel_first else tmp[..., 0]
            
    vis_slices = min(vis_slices, tmp.shape[-1])
    idx = [int(tmp.shape[-1] // vis_slices * (s + 0.5)) for s in range(vis_slices)]
    imgs = None
    for n in range(tmp.shape[0]):
        tmp_list = []
        for xs in img_list:
            assert len(xs.shape) in [4, 5]
            tmp_xs = np.array(xs[n])
            if len(tmp_xs.shape) == 4:
                tmp_xs = tmp_xs[0] if channel_first else tmp_xs[..., 0]
            tmp_xs = tmp_xs[..., idx]
            tmp_list.append(tmp_xs.transpose(2, 0, 1))
        img = combine_2d_imgs_from_tensor(tmp_list, nmin=nmin, nmax=nmax)
        imgs = img if imgs is None else np.concatenate((imgs, img), 0)
    return imgs