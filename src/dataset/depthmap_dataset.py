import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class DepthmapDatasetParams:
    def __init__(self, path_to_dataset : str, scale : float, max_len : int = None):
        '''
        scale: value to scale loaded dataset (to adjust for metersmillimeters)
        max_len: use this to trim dataset to first max_len elements
        '''
        self.path_to_dataset = path_to_dataset        
        self.scale = scale
        self.max_len = max_len

class DepthmapDataset(Dataset):

    def __init__(self, params : DepthmapDatasetParams):
        super().__init__()
        self._params = params        
        self._get_filelist()
    
    def __len__(self):
        return len(self._filepaths) if (self._params.max_len == None) or (len(self._filepaths) < self._params.max_len) else self._params.max_len

    def _get_filelist(self):
        self._filepaths = list(map(lambda x: os.path.join(self._params.path_to_dataset,x),os.listdir(self._params.path_to_dataset)))
        self._filepaths = [x for x in self._filepaths if x.endswith(".pgm")]
        self._filepaths.sort(key = lambda x: int(x.split("\\")[-1].split(".")[0].split("_")[2]))

    def __getitem__(self,idx):

        fname = self._filepaths[idx]
        img = cv2.imread(fname,cv2.IMREAD_ANYDEPTH).astype(np.float32) * self._params.scale
        
        return torch.from_numpy(img).unsqueeze(0)    

