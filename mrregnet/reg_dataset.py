import glob
import numpy as np
from ..snmi.core.base_dataset import BaseDataset
from ..snmi.utils.utils import load_file

class RegDataset(BaseDataset):

    def __init__(self, data_source, data_target, source_key, target_key, data_suffix, preprocesses=None, augmentation=None):

        assert isinstance(data_source, (list, np.ndarray)), 'Only accpet filename list.'
        assert isinstance(data_target, (list, np.ndarray)), 'Only accpet filename list.'

        if preprocesses:
            new_preprocesses = {}
            [new_preprocesses.update({source_key+k: v}) for k, v in preprocesses.items()]
            [new_preprocesses.update({target_key+k: v}) for k, v in preprocesses.items()]
        super().__init__(data_target, data_suffix, new_preprocesses, augmentation)
        
        self._data_source = data_source
        self._data_target = data_target
        self._source_key = source_key
        self._target_key = target_key

    def __getitem__(self, idx):
        data_dict = {}

        # target
        t_name = self._data_target[idx] # target image
        data_dict.update({self._target_key + self._org_suffix: load_file(t_name)})
        for o_suffix in self._other_suffix:
            if o_suffix is None:
                continue
            o_name = t_name.replace(self._org_suffix, o_suffix) # target label
            data_dict.update({self._target_key + o_suffix: load_file(o_name)})
            

        # source
        if len(self._data_source) > 0:
            s_name = self._data_source[idx % len(self._data_source)] # source image
            data_dict.update({self._source_key + self._org_suffix: load_file(s_name)})
            for o_suffix in self._other_suffix:
                if o_suffix is None:
                    continue
                so_name = s_name.replace(self._org_suffix, o_suffix) # source label
                data_dict.update({self._source_key + o_suffix: load_file(so_name)})

        data_dict = self.augmentation(data_dict)
        data_dict = self.pre_process(data_dict)
        return data_dict


    def pre_process(self, data_dict):
        if self._preprocesses is None:
            return data_dict

        for key in self._preprocesses:
            for method in self._preprocesses[key]:
                if key in data_dict:
                    data_dict.update({key: method(data_dict[key])})
        return data_dict

        





