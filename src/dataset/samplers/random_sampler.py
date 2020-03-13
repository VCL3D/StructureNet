import numpy as np
from .base_sampler import BaseSampler

class RandomSampler(BaseSampler):
    def __init__(self, data : list, weights : list, rnd_seed = 1234):
        super().__init__()
        assert(len(data) == len(weights))

        self._data = data
        self._probabilities = np.cumsum(weights) / np.sum(weights)
        self._rng = np.random.RandomState(rnd_seed)

    def sample(self):
        p = self._rng.uniform()
        index = np.min([np.searchsorted(self._probabilities,p,side='right'), len(self._probabilities)-1])
        return self._data[index]
