import numpy as np
from ..base_sampler import BaseSampler

class UniformNoisyBackgroundGeneratorParams:
    def __init__(self, width : int, height : int, depth_min : float = 0.5, depth_max : float = 8.0, rnd_seed : int = 1234):
        self.width = width
        self.height = height
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.rnd_seed = rnd_seed

class UniformNoisyBackgroundGenerator(BaseSampler):
    def __init__(self,params : UniformNoisyBackgroundGeneratorParams):
        super().__init__()
        self._params = params
        self._rng = np.random.RandomState(self._params.rnd_seed)

    def sample(self):        
        noisy_bg = self._rng.uniform(self._params.depth_min, self._params.depth_max,(self._params.height, self._params.width))
        return noisy_bg


class GaussianNoisyBackgroundGeneratorParams:
    def __init__(self, width : int, height : int, depth_mean : float = 3.5, depth_std : float = 1.5, rnd_seed = 1234):
        self.width = width
        self.height = height
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.rnd_seed = rnd_seed

class GaussianNoisyBackgroundGenerator(BaseSampler):
    def __init__(self, params : GaussianNoisyBackgroundGeneratorParams):
        super().__init__()
        self._params = params
        self._rng = np.random.RandomState(self._params.rnd_seed)

    def sample(self):        
        noisy_bg = self._params.depth_std * self._rng.randn(self._params.height,self._params.width) + self._params.depth_mean
        np.clip(noisy_bg, 0.0, np.Inf, out = noisy_bg)
        return noisy_bg
