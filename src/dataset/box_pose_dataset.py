import torch.nn as nn
from torch.utils.data.dataset import Dataset
from .rendering.box_renderer import *
from .samplers.random_sampler import RandomSampler as rs
from .samplers.pose.pose_sampler import *
from .samplers.intrinsics_generator import *
from ..utils.image_utils import *
from .noise.noise_adder import HoleNoiseAdder
class BoxPoseDatasetParams:

    def __init__(self, box_renderer : BoxRenderer, pose_sampler : PoseSampler,
                 intrinsics_generator : IntrinsicsGenerator,
                 renderer_params: BoxRendererParams,
                 background_samplers : list,
                 background_samplers_probabilities : list,      # list of floats
                 noise_adders : list,                           
                 noise_adders_probabilities : list,             # list of floats
                 border_noise_adders : list,
                 border_noise_adders_probabilities : list,      # list of floats
                 hole_adder : HoleNoiseAdder,                 
                 output_resolution : tuple = None,              # default output resolution is determined by intrinsics generator. if specified, the output will be interpolated to new resolution (width x height)
                 background_sampler_seed = 1111,
                 noise_sampler_seed = 3333,
                 border_noise_adder_seed = 3456):

        self.box_renderer = box_renderer        
        self.pose_sampler = pose_sampler
        self.intrinsics_generator = intrinsics_generator
        self.renderer_params = renderer_params
        self.background_samplers = background_samplers
        self.background_samplers_probabilities = background_samplers_probabilities
        self.noise_adders = noise_adders
        self.noise_adders_probabilities = noise_adders_probabilities
        self.hole_adder = hole_adder
        self.border_noise_adders = border_noise_adders
        self.border_noise_adders_probabilities = border_noise_adders_probabilities
        self.border_noise_adder_seed = border_noise_adder_seed
        self.output_resolution = output_resolution
        self.noise_sampler_seed = noise_sampler_seed
        self.background_sampler_seed = background_sampler_seed

class BoxPoseDataset(Dataset):

    def __init__(self, params : BoxPoseDatasetParams):
        super().__init__()
        self._params = params        
        self.background_sampler = rs(params.background_samplers,params.background_samplers_probabilities,rnd_seed = params.background_sampler_seed)
        self.noise_adder_sampler = rs(params.noise_adders,params.noise_adders_probabilities, rnd_seed = params.noise_sampler_seed)
        self.border_noise_adder_sampler = rs(params.border_noise_adders, params.border_noise_adders_probabilities, rnd_seed = params.border_noise_adder_seed)
        
    def _create_intrinsics_matrix(
        self,
        camera_intrinsics   :   list,
        aspect_ratios       :   tuple,
        )                       -> torch.tensor:
        fx = camera_intrinsics[0] / aspect_ratios[0] #fx
        cx = camera_intrinsics[2] / aspect_ratios[0] #cx

        fy = camera_intrinsics[1] / aspect_ratios[1] #fy
        cy = camera_intrinsics[3] / aspect_ratios[1] #cy

        r = torch.eye(3)
        r[0,0] = fx
        r[1,1] = fy
        r[0,2] = cx
        r[1,2] = cy

        return r

    def __len__(self):
        return self._params.pose_sampler.transformations.shape[0]


    def __getitem__(self,idx):
        camera_pose = self._params.pose_sampler.transformations[idx]
        self._params.box_renderer.canvas_width = self._params.intrinsics_generator.width
        self._params.box_renderer.canvas_height = self._params.intrinsics_generator.height
        camera_intrinsics = self._params.intrinsics_generator.sample()

        color, depth, normals , labels = self._params.box_renderer.render(camera_pose = camera_pose, camera_intrinsics = camera_intrinsics, 
                                    znear = self._params.renderer_params.depth_zmin, zfar = self._params.renderer_params.depth_zmax,
                                    render_flags = self._params.renderer_params.render_flags)        

        bg = np.float32(self.background_sampler.sample().sample())        
        bg = random_crop_and_scale_to_fit_target(bg,self._params.intrinsics_generator.width,self._params.intrinsics_generator.height)        

        hole_depth = self._params.hole_adder.add_noise(depth)
        nd = self.noise_adder_sampler.sample().add_noise(hole_depth)
        mask1 = nd == 0
        
        bnd = self.border_noise_adder_sampler.sample().add_noise(nd)
        mask2 = bnd == 0

        fmask = mask2 * (~mask1)

        final_depth = bg
        #final_depth[nd > 0.0] = nd[nd > 0.0]
        final_depth[depth > 0.0] = nd[depth > 0.0]
        #final_depth = self.border_noise_adder_sampler.sample().add_noise(final_depth)        
        final_depth[fmask] = bnd[fmask]

        normals [final_depth == 0] = 0
        labels = np.ascontiguousarray(labels)
        labels [final_depth == 0] = 0
        color = np.ascontiguousarray(color)
        color [final_depth == 0] = 0

        fdepth = torch.from_numpy(final_depth).unsqueeze(0)
        fnormals = torch.from_numpy(normals).permute(2,0,1)
        flabels = torch.from_numpy(labels).unsqueeze(0)
        fcolor = torch.from_numpy(color).permute(2,0,1)

        aspect_ratio = (1.0,1.0)    
        if self._params.output_resolution != None:
            out_width, out_height = self._params.output_resolution            
            fdepth   = nn.functional.interpolate(fdepth.unsqueeze(0), size=[out_height, out_width], mode='nearest').squeeze(0)
            fnormals = nn.functional.interpolate(fnormals.unsqueeze(0), size=[out_height, out_width], mode='nearest').squeeze(0)
            flabels  = nn.functional.interpolate(flabels.unsqueeze(0).float(), size=[out_height, out_width], mode='nearest').squeeze(0).to(torch.uint8)
            fcolor   = nn.functional.interpolate(fcolor.unsqueeze(0).float(), size=[out_height, out_width], mode='nearest').squeeze(0).to(torch.uint8)
            aspect_ratio = (self._params.box_renderer.canvas_width/out_width,
                            self._params.box_renderer.canvas_height/out_height)

        intrinsics = self._create_intrinsics_matrix(camera_intrinsics,aspect_ratio)

        r = {
            "depth" : fdepth.float(),
            "normals" : fnormals.float(),
            "labels" : flabels,
            "color" : fcolor,
            "intrinsics_original" : torch.from_numpy(np.ascontiguousarray(camera_intrinsics)).float(),                       # maybe used to deprojet depth
            "intrinsics" : intrinsics.float(),
            "camera_resolution" : (self._params.intrinsics_generator.width, self._params.intrinsics_generator.height),       # original resolution before transform to use to deproject with intrinsics after resizing image
            "camera_pose" : torch.from_numpy(camera_pose).float(),
            "type" : "synthetic"
        }
        return r

    

