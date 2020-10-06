import numpy
import json
from .base_sampler import BaseSampler

class IntrinsicsGeneratorParams(object):

    def __init__(self, width : int, height : int, rnd_seed : int):      
        self.width = width
        self.height = height
        self.rnd_seed = rnd_seed

class IntrinsicsGenerator(BaseSampler):

    def __init__(self, params : IntrinsicsGeneratorParams):
        super().__init__()
        self._params = params
        self._rnd_gen = numpy.random.RandomState(self._params.rnd_seed)

        # tuple((width,height)): [[fx fy cx cy], [fx fy cx cy], ...]
        self._device_intrinsics =  {            
            (1280,720) :[
                            [943.5726318359375, 943.5726318359375, 636.60302734375, 352.9541015625],            # RS2 Intrinsics
                            [939.235107421875, 939.235107421875, 639.2382202148438, 350.4108581542969]          # RS2 Intrinsics
                        ],
             (640,480) :[
                            [629.0484008789063,629.0484008789063,317.7353515625,235.302734375],                 # RS2 Intrinsics
                            [626.15673828125,626.15673828125, 319.4921569824219, 233.60723876953126],           # RS2 Intrinsics
                            [582.023188, 585.883685, 314.722819, 224.157081],                                   # K1 Intrinsics
                            [581.810914, 580.285359, 314.055143, 231.700159],                                   # K1 Intrinsics
                            [592.417057, 576.458251, 326.514575, 243.213944],                                   # K1 Intrinsics
                            [578.750057, 584.497763, 325.442541, 237.415025],                                   # K1 Intrinsics
                            [580.203486, 585.823696, 330.492915, 221.879735],                                   # K1 Intrinsics
                            [596.025924, 592.786521, 333.317519, 248.456780],                                   # K1 Intrinsics
                            [587.712380, 581.384658, 328.100841, 231.595926]                                    # K1 Intrinsics
                        ],
            (512,424) : [ 
                            [366.7136, 366.7136, 256.5948, 207.1343],                                           #K2 Intrinsics
                            [367.4368, 367.4368, 260.8115, 205.1943],                                           #K2 Intrinsics
                            [364.3239, 364.3239, 258.5376, 203.6222],                                           #K2 Intrinsics
                            [365.2731, 365.2731, 255.1621, 208.3562]                                            #K2 Intrinsics
                        ],
            (320,288) : [
                            [252.3858, 252.4081, 163.4171, 165.4822],
                            [252.3462, 252.3647, 157.0585, 166.1083],
                            [252.2103, 252.2250, 167.5221, 170.2071],
                            [251.9636, 251.9164, 163.3273, 166.7225],
                            [251.8373, 251.7830, 166.1493, 171.8638],
                            [252.5108, 252.5648, 163.9615, 170.0882]
            ]
        }

        width = self._params.width
        height = self._params.height

        if not (width, height) in self._device_intrinsics:
            new_intrinsics_list = []
            for reso in self._device_intrinsics.keys():
                
                if (reso[0] % width) == 0 and (reso[1] % height) == 0:
                    downscale_factor_x = reso[0] / width
                    downscale_factor_y = reso[1] / height
                    
                    for intr in self._device_intrinsics[reso]:
                        new_intrinsics = [intr[0] / downscale_factor_x, intr[1] / downscale_factor_y, intr[2] / downscale_factor_x, intr[3]/downscale_factor_y]
                        new_intrinsics_list.append(new_intrinsics)

            if len(new_intrinsics_list) > 0:
                self._device_intrinsics[(width,height)] = new_intrinsics_list
            else:
                raise Exception('invalid intrinsics request. no suitable device intrinsics are available for this resolution')


    def sample(self) -> list:

        reso = (self._params.width, self._params.height)
        intr_list = self._device_intrinsics[reso]

        index = self._rnd_gen.randint(0,len(intr_list))        

        intrinsics = intr_list [index]
        return intrinsics

    @property
    def width(self) -> int:
        return self._params.width

    @property
    def height(self) -> int:
        return self._params.height 
