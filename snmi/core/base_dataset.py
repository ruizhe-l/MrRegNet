import glob
import numpy as np

from torch.utils.data.dataset import Dataset
from ..utils.utils import load_file

class BaseDataset(Dataset):

    """Basic PyTorch dataset for segmentation data. 

    Parameter
    ----------
    data: str, list or dict
        str: Path to search the file.
        list: A list of all filenames. 
        dict: A dict include all data.
    data_suffix: list or tuple
        A list of suffix, the first value should be the suffix of input images.
    preprocesses: dict, optional
        Maps from suffix to process methods for image pre-processing.
        e.g. {'org': [min_max, Resize([256,256])],
              'lab': [OneHot(2), Resize([256,256], nearest=True)]}

    Attributes
    ----------
    _data_suffix: str
        Suffix for input images and their corresponding key in the output.
    _other_suffixes: list of tuple, optional
        Suffix for other images and their corresponding keys in the output.
    _preprocesses: dict
        Maps from suffix to process methods for image pre-processing.
    _data_list: list:str
        A list of all filenames/data. 
    
    """
    def __init__(self,
                data,
                data_suffix,
                preprocesses=None,
                augmentation=None):
        assert len(data) > 0, 'Empty dataset!'
        assert len(data_suffix) > 0, 'Empty suffix!'
        self._org_suffix = data_suffix[0]
        self._other_suffix = data_suffix[1:]
        
        self._preprocesses = preprocesses
        self._augmentation = augmentation
        
        self._file_list = None
        self._all_data = None
        if isinstance(data, (str)):
            self._file_list = glob.glob(data)
        elif isinstance(data, (list, np.ndarray)):
            self._file_list = data
        elif isinstance(data, dict):
            self._all_data = data
        else:
            raise ValueError('Only accept one of (search_path, file_list, data_dict).')
    
    def __len__(self):
        return len(self._file_list) if self._file_list is not None else len(self._all_data)

    def __getitem__(self, idx):
        data_dict = {}
        if self._all_data is not None: # if all data loaded in __init__()
            for key in self._all_data:
                data_dict.update({key: self._all_data[key][idx]})
        else:   # if filename list loaded in __init__()
            x_name = self._file_list[idx]
            data_dict.update({self._org_suffix: load_file(x_name)})
            for o_suffix in self._other_suffix:
                o_name = x_name.replace(self._org_suffix, o_suffix)
                data_dict.update({o_suffix: load_file(o_name)})

        data_dict = self.augmentation(data_dict)
        data_dict = self.pre_process(data_dict)
        return data_dict

    def pre_process(self, data_dict):
        if self._preprocesses is None:
            return data_dict

        for key in self._preprocesses:
            for method in self._preprocesses[key]:
                assert key in data_dict, f'Data process error: no key \'{key}\' in data_dict'
                data_dict.update({key: method(data_dict[key])})
        return data_dict

    def augmentation(self, data_dict):
        if self._augmentation is None:
            return data_dict
        
        return self._augmentation(data_dict)

class PairedDataset(BaseDataset):
    def __init__(self, data_more, data_less, more_key, less_key, data_suffix, preprocesses=None, augmentation=None):

        assert isinstance(data_more, (list, np.ndarray)), 'Only accpet filename list.'
        assert isinstance(data_more, (list, np.ndarray)), 'Only accpet filename list.'
        while len(data_more) < len(data_less):
            data_more = list(data_more) * 2
        if preprocesses:
            new_preprocesses = {}
            [new_preprocesses.update({more_key+k: v}) for k, v in preprocesses.items()]
            [new_preprocesses.update({less_key+k: v}) for k, v in preprocesses.items()]
        super().__init__(data_more, data_suffix, new_preprocesses, augmentation)
        
        self._data_more = self._file_list
        self._data_less = data_less
        self._more_key = more_key
        self._less_key = less_key

    def __getitem__(self, idx):
        data_dict = {}

        x_name = self._data_more[idx] # source
        data_dict.update({self._more_key + self._org_suffix: load_file(x_name)})
        y_name = self._data_less[idx % len(self._data_less)] #target
        data_dict.update({self._less_key + self._org_suffix: load_file(y_name)})
        for o_suffix in self._other_suffix:
            o_name = x_name.replace(self._org_suffix, o_suffix) # source labels
            data_dict.update({self._more_key + o_suffix: load_file(o_name)})
            yo_name = y_name.replace(self._org_suffix, o_suffix) # target labels
            data_dict.update({self._less_key + o_suffix: load_file(yo_name)})

        data_dict = self.augmentation(data_dict)
        data_dict = self.pre_process(data_dict)
        return data_dict




