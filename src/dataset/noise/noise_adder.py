import torch
from . import noise
import numpy as np
from abc import ABC, abstractclassmethod
import cv2

class BaseNoiseAdder(ABC):
    def __init__(self):
        super().__init__()

    @abstractclassmethod
    def add_noise(self, depth):
        pass

class DisparityNoiseParams:
    
    def __init__(self, depth_pre_scale_factor : float = 1.0, sigma_depth : float = (1.0/6.0), sigma_space : float = (1.0/2.0), mean_space : float = 0.5):
        '''
        depth_pre_scale_factor: scale to multiply depth with, in order to make it in meters
        sigma_depth, sigma_space, mean_space: set these values for meter units
        '''
        self.depth_pre_scale_factor = depth_pre_scale_factor
        self.sigma_depth = sigma_depth
        self.sigma_space = sigma_space
        self.mean_space = mean_space

class TofNoiseParams:
    def __init__(self, sigma_fraction : float = 0.1):
        self.sigma_fraction = sigma_fraction

class DisparityNoiseAdder(BaseNoiseAdder):
    def __init__(self, params : DisparityNoiseParams):
        self._params = params
    def add_noise(self, depth : np.array) -> np.array:
        torch_depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        #scale = 1.0 if self._params.disparity_noise_params.distance_unit == DistanceUnit.Meters else 0.001 # disparity noise model requires input in meters
        scale = self._params.depth_pre_scale_factor
        noisy_depth_torch , _ = noise.disparity_noise(scale * torch_depth,self._params.sigma_depth, \
                        self._params.sigma_space, self._params.mean_space)            
        noisy_depth = torch.squeeze(1/scale * noisy_depth_torch).data.numpy()
        return noisy_depth

class TofNoiseAdder(BaseNoiseAdder):
    def __init__(self, params : TofNoiseParams):
        self._params = params
    def add_noise(self,depth : np.array) -> np.array:
        torch_depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        noisy_depth = torch.squeeze(noise.tof_noise(torch_depth,self._params.sigma_fraction)).data.numpy()
        return noisy_depth


class HoleNoiseParams:
    def __init__(self, min_radius : int, max_radius: int, min_hole_count : int , max_hole_count : int, rnd_seed : int = 4567):
        '''
        min_radius: minimum radius in pixels
        max_radius: max radius in pixels
        min_hole_count : min number of holes to create
        max_hole_count : max number of holes to create
        '''
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_hole_count = min_hole_count
        self.max_hole_count = max_hole_count
        self.rnd_seed = rnd_seed


class HoleNoiseAdder(BaseNoiseAdder):
    def __init__(self, params : HoleNoiseParams) :
        self._params = params
        self._rnd_gen = np.random.RandomState(self._params.rnd_seed)

    def add_noise(self, depth : np.array) -> np.array:

        height = depth.shape[0]
        width = depth.shape[1]

        mask = depth != 0
        #loc = [(y,x) for x in range(width) for y in range(height) if mask[y,x] != 0]
        loc = np.concatenate((np.expand_dims(np.nonzero(mask)[0], axis = 1),np.expand_dims(np.nonzero(mask)[1], axis = 1)),axis = 1).tolist()

        noisy_depth = depth.copy()

        if(len(loc) == 0):
            return noisy_depth

        hcount = self._rnd_gen.randint(low = self._params.min_hole_count, high = self._params.max_hole_count)

        hole_center_ind = self._rnd_gen.randint(low = 0, high = len(loc), size = hcount)

        for idx in range(len(hole_center_ind)):
            y, x = loc[hole_center_ind[idx]]            
            radius = self._rnd_gen.randint(low = self._params.min_radius, high = self._params.max_radius)
            
            noisy_depth = cv2.circle(noisy_depth , tuple(loc[hole_center_ind[idx]]), radius, color = 0.0 , thickness=cv2.FILLED,lineType=8)

        return noisy_depth

class BorderNoiseParams:

    def __init__(self, border_width : int = 3, iterations: int = 1):
        self.border_width = border_width
        self.iterations = iterations

class BorderErodeNoiseAdder(BaseNoiseAdder):

    def __init__(self, params : BorderNoiseParams):
        self._params = params

    def add_noise(self, depth : np.array) -> np.array:

        # Taking a matrix of a kernel
        kernel = np.ones((self._params.border_width,self._params.border_width), np.float32)

        mask = np.float32(depth != 0.0)
        
        eroded_mask = cv2.erode(mask,kernel,iterations = self._params.iterations)

        noisy_depth = depth.copy()        
        noisy_depth[mask != eroded_mask] = 0.0

        return noisy_depth

class BorderDilateNoiseAdder(BaseNoiseAdder):

    def __init__(self, params : BorderNoiseParams):
        self._params = params

    def add_noise(self, depth : np.array) -> np.array:
        # Taking a matrix of a kernel
        kernel = np.ones((self._params.border_width,self._params.border_width), np.float32)

        mask = np.float32(depth != 0.0)
        
        dilated_mask = cv2.dilate(mask,kernel,iterations = self._params.iterations)

        noisy_depth = depth.copy()        
        noisy_depth[mask != dilated_mask] = 0.0

        return noisy_depth
