import cv2
import numpy as np
import os
from ..base_sampler import BaseSampler

class ImageBackgroundSamplerParams:
    def __init__(self, path_to_dataset : str, scale : float, rnd_seed : int = 6677):
        '''
        scale: value to scale loaded background (to adjust for metersmillimeters)
        '''
        self.path_to_dataset = path_to_dataset
        self.rnd_seed = rnd_seed        
        self.scale = scale

class ImageBackgroundSampler(BaseSampler):

    def __init__(self, params : ImageBackgroundSamplerParams):
        super().__init__()
        self._params = params
        self._rnd_gen = np.random.RandomState(self._params.rnd_seed)
        self._get_filelist()
    

    def _get_filelist(self):
        self._filepaths = list(map(lambda x: os.path.join(self._params.path_to_dataset,x),os.listdir(self._params.path_to_dataset)))

    def sample(self):

        bgcount = len(self._filepaths)
        index = self._rnd_gen.randint(0,bgcount)

        fname = self._filepaths[index]
        img = cv2.imread(fname,cv2.IMREAD_ANYDEPTH).astype(np.float32) * self._params.scale
        return img
