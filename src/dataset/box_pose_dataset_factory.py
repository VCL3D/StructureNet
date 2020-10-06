from .box_pose_dataset import *
from .samplers.pose import *
from .samplers.intrinsics_generator import *
from .distance_unit import *
from .samplers.background.noisy_background_generator import *
from .samplers.background.image_background_sampler import *
from .noise.noise_adder import *
from .rendering.box_renderer import BoxRenderFlags
from enum import Flag, auto
import random

class BoxPoseDatasetType(Flag):
  HORIZONTAL = auto()
  VERTICAL_1 = auto()
  VERTICAL_2 = auto()
  HORIZONTAL_AND_VERTICAL_1 = HORIZONTAL | VERTICAL_1
  HORIZONTAL_AND_VERTICAL_2 = HORIZONTAL | VERTICAL_2


class BoxPoseDatasetFactoryParams:

    def __init__(self, background_sampler_probabilities : list,             # list of 5 floats
                       noise_add_probabilities : list,                      # list of 2 floats  
                       border_noise_add_probabilities : list,               # list of 2 floats                                              
                       out_resolution : tuple = None,                       # (width , height)                       
                       uniform_noisy_background_rnd_seed : int = 1111,
                       gaussian_noisy_background_rnd_seed : int = 3333,
                       corbs_image_background_rnd_seed : int = 4444,
                       vcl_image_background_rnd_seed : int = 5555,
                       intnet_image_background_rnd_seed : int = 8080,
                       composite_background_sampler_rnd_seed : int = 6161,
                       composite_noise_adder_rnd_seed : int = 7777,
                       composite_border_noise_adder_rnd_seed : int = 8888
                     ):
        '''
        background_sampler_probabilities: [uniform_background_noise, gaussian_background_noise, corbs, vcl] probabilities
        noise_add_probabilities: [disparity_noise, tof_noise]
        border_noise_add_probabilities [dilation_noise errosion_noise]
        out_resolution (width x height)
        '''                
        self.out_resolution = out_resolution
        self.uniform_noisy_background_rnd_seed = uniform_noisy_background_rnd_seed
        self.gaussian_noisy_background_rnd_seed = gaussian_noisy_background_rnd_seed
        self.corbs_image_background_rnd_seed = corbs_image_background_rnd_seed
        self.vcl_image_background_rnd_seed = vcl_image_background_rnd_seed
        self.intnet_image_background_rnd_seed = intnet_image_background_rnd_seed
        self.composite_background_sampler_rnd_seed = composite_background_sampler_rnd_seed
        self.composite_noise_adder_rnd_seed = composite_noise_adder_rnd_seed
        self.composite_border_noise_adder_rnd_seed = composite_border_noise_adder_rnd_seed

        assert(len(background_sampler_probabilities) == 5)

        sprob = sum(background_sampler_probabilities)
        self.background_sampler_probabilities = list(map(lambda x : x / sprob, background_sampler_probabilities))

        assert(len(noise_add_probabilities) == 2)

        sprob = sum(noise_add_probabilities)
        self.noise_add_probabilities = list(map(lambda x : x / sprob, noise_add_probabilities))

        sprob = sum(border_noise_add_probabilities)
        self.border_noise_add_probabilities = list(map(lambda x : x / sprob, border_noise_add_probabilities))


def create_box_pose_dataset(renderer_params : BoxRendererParams, \
                            intrinsics_params : IntrinsicsGeneratorParams, \
                            pose_sampler_params  : PoseSamplerParams, \
                            path_to_corbs_background_dataset : str, \
                            path_to_vcl_background_dataset :str, \
                            path_to_intnet_background_dataset : str, \
                            factory_params : BoxPoseDatasetFactoryParams, \
                            units : DistanceUnit = DistanceUnit.Meters) \
                            -> BoxPoseDataset :

    ps = PoseSampler(pose_sampler_params)

    boxscale = 1.0 if (units == DistanceUnit.Millimeters) else 0.001

    br = BoxRenderer(box_scale = boxscale)
    ig = IntrinsicsGenerator(intrinsics_params)
    
    canvas_width = intrinsics_params.width
    canvas_height = intrinsics_params.height

    uniform_bgnoise_p = UniformNoisyBackgroundGeneratorParams(canvas_width,canvas_height, rnd_seed = factory_params.uniform_noisy_background_rnd_seed)
    gaussian_bgnoise_p = GaussianNoisyBackgroundGeneratorParams(canvas_width,canvas_height, rnd_seed = factory_params.gaussian_noisy_background_rnd_seed)
    
    corbs_bg_p = ImageBackgroundSamplerParams(path_to_corbs_background_dataset,0.0002, rnd_seed = factory_params.corbs_image_background_rnd_seed)
    vcl_bg_p = ImageBackgroundSamplerParams(path_to_vcl_background_dataset,0.001, rnd_seed = factory_params.vcl_image_background_rnd_seed)
    intnet_bg_p = ImageBackgroundSamplerParams(path_to_intnet_background_dataset, 0.001, rnd_seed = factory_params.intnet_image_background_rnd_seed)

    background_samplers = [UniformNoisyBackgroundGenerator(uniform_bgnoise_p),
                           GaussianNoisyBackgroundGenerator(gaussian_bgnoise_p),
                           ImageBackgroundSampler(corbs_bg_p),
                           ImageBackgroundSampler(vcl_bg_p),
                           ImageBackgroundSampler(intnet_bg_p)]
    
    background_sampler_probabilities = factory_params.background_sampler_probabilities

    disp_noise_p = DisparityNoiseParams(sigma_depth = 1/4.0, sigma_space = 1.0, mean_space = 0.8)
    tof_noise_p = TofNoiseParams(0.35)

    noise_adders = [DisparityNoiseAdder(disp_noise_p), TofNoiseAdder(tof_noise_p)]
    noise_adder_probabilities = factory_params.noise_add_probabilities

    dilation_noise_p = BorderNoiseParams(border_width = 4, iterations = 2)
    erosion_noise_p = BorderNoiseParams(border_width = 4, iterations = 2)

    border_noise_adders = [BorderDilateNoiseAdder(dilation_noise_p), BorderErodeNoiseAdder(erosion_noise_p)]
    border_noise_adder_probabilities = factory_params.border_noise_add_probabilities

    hole_noise_p = HoleNoiseParams(min_radius = 3, max_radius = 10, min_hole_count = 0, max_hole_count = 20)
    hole_adder = HoleNoiseAdder(hole_noise_p)

    box_pose_dataset_params = BoxPoseDatasetParams(br, ps, ig,  renderer_params,
                                                    background_samplers, background_sampler_probabilities,
                                                    noise_adders,
                                                    noise_adder_probabilities,
                                                    border_noise_adders,
                                                    border_noise_adder_probabilities,
                                                    hole_adder,
                                                    factory_params.out_resolution,
                                                    factory_params.composite_background_sampler_rnd_seed,
                                                    factory_params.composite_noise_adder_rnd_seed,
                                                    factory_params.composite_border_noise_adder_rnd_seed)

    return BoxPoseDataset(box_pose_dataset_params)


def _create_boxpose_dataset_factory_params(out_resolution : tuple):     # resolution (width,height)

    return BoxPoseDatasetFactoryParams([0.125, 0.125, 0.2, 0.2, 0.35],[0.5, 0.5], [0.5, 0.5],                                               
                                                out_resolution = out_resolution,
                                                uniform_noisy_background_rnd_seed=random.randint(0,10000),
                                                gaussian_noisy_background_rnd_seed=random.randint(0,10000),
                                                corbs_image_background_rnd_seed=random.randint(0,10000),
                                                vcl_image_background_rnd_seed=random.randint(0,10000),
                                                intnet_image_background_rnd_seed=random.randint(0,10000),
                                                composite_background_sampler_rnd_seed=random.randint(0,10000),
                                                composite_noise_adder_rnd_seed=random.randint(0,10000),
                                                composite_border_noise_adder_rnd_seed=random.randint(0,10000))

def create_random_box_pose_datasets(sample_count_per_dataset : int,                                     
                                    path_to_corbs_background_dataset : str, path_to_vcl_background_dataset : str,
                                    path_to_intnet_background_dataset : str,
                                    box_render_flags : BoxRenderFlags = None,
                                    dataset_type : BoxPoseDatasetType = BoxPoseDatasetType.HORIZONTAL,
                                    out_resolution_width : int = 320, out_resolution_height : int = 240 ) -> list:
  '''
  returns list of BoxPoseDataset for resolutions: [640x360, 640x480, 512x424] and pose types = [horizontal , vertical #1, vertical #2]
  all datasets are configured to resize output to the resolution parameters
  '''
  raise Exception("Pose parameters should be as in 16_9 dataset") 
  out_resolution = (out_resolution_width,out_resolution_height)
  datasets = []
  up_vector_variance = 10.0
  rmin = 1.5
  rmax = 2.5
  if dataset_type & BoxPoseDatasetType.HORIZONTAL:
    # horizontal dataset, 640x360
    render_params_horiz_640x360 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_horiz_640x360 = IntrinsicsGeneratorParams(640,360, random.randint(0,10000))
    pose_params_horiz_640x360 = PoseSamplerParams(num_positions = sample_count_per_dataset,
                                            rmin = rmin,
                                            rmax = rmax,
                                            zmin = -0.35,
                                            zmax = 1.0,
                                            look_at_radius = 0.5,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.HORIZONAL,
                                            random_seed=random.randint(0,10000))
    
    fp_horiz_640x360 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_horiz_640x360, intrinsics_params_horiz_640x360, pose_params_horiz_640x360, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_horiz_640x360))

    # horizontal dataset, 640x480
    render_params_horiz_640x480 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_horiz_640x480 = IntrinsicsGeneratorParams(640,480, random.randint(0,10000))
    pose_params_horiz_640x480 = PoseSamplerParams(num_positions = sample_count_per_dataset,
                                            rmin = rmin,
                                            rmax = rmax,
                                            zmin = -0.35,
                                            zmax = 1.0,
                                            look_at_radius = 0.5,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.HORIZONAL,
                                            random_seed=random.randint(0,10000))
    
    fp_horiz_640x480 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_horiz_640x480, intrinsics_params_horiz_640x480, pose_params_horiz_640x480, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_horiz_640x480))
    # horizontal dataset, 512x424
    render_params_horiz_512x424 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_horiz_512x424 = IntrinsicsGeneratorParams(512,424, random.randint(0,10000))
    pose_params_horiz_512x424 = PoseSamplerParams(num_positions = sample_count_per_dataset,
                                            rmin = rmin,
                                            rmax = rmax,
                                            zmin = -0.35,
                                            zmax = 1.0,
                                            look_at_radius = 0.5,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.HORIZONAL,
                                            random_seed=random.randint(0,10000))
    
    fp_horiz_512x424 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_horiz_512x424, intrinsics_params_horiz_512x424, pose_params_horiz_512x424, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_horiz_512x424))  

  if dataset_type & BoxPoseDatasetType.VERTICAL_1:
    # vertical dataset #1, 640x360
    render_params_vert1_640x360 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert1_640x360 = IntrinsicsGeneratorParams(640,360, random.randint(0,10000))
    pose_params_vert1_640x360 = PoseSamplerParams(num_positions = sample_count_per_dataset,
                                            rmin = rmin,
                                            rmax = rmax,
                                            zmin = -0.35,
                                            zmax = 1.0,
                                            look_at_radius = 0.5,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.VERTICAL_1,
                                            random_seed=random.randint(0,10000))
    
    fp_vert1_640x360 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert1_640x360, intrinsics_params_vert1_640x360, pose_params_vert1_640x360, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert1_640x360))  
    # vertical dataset #1, 640x480
    render_params_vert1_640x480 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert1_640x480 = IntrinsicsGeneratorParams(640,480, random.randint(0,10000))
    pose_params_vert1_640x480 = PoseSamplerParams(num_positions = sample_count_per_dataset,
                                            rmin = rmin,
                                            rmax = rmax,
                                            zmin = -0.35,
                                            zmax = 1.0,
                                            look_at_radius = 0.5,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.VERTICAL_1,
                                            random_seed=random.randint(0,10000))
    
    fp_vert1_640x480 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert1_640x480, intrinsics_params_vert1_640x480, pose_params_vert1_640x480, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert1_640x480))  
    # vertical dataset #1, 512x424
    render_params_vert1_512x424 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert1_512x424 = IntrinsicsGeneratorParams(512,424, random.randint(0,10000))
    pose_params_vert1_512x424 = PoseSamplerParams(num_positions = sample_count_per_dataset,
                                            rmin = rmin,
                                            rmax = rmax,
                                            zmin = -0.35,
                                            zmax = 1.0,
                                            look_at_radius = 0.5,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.VERTICAL_1,
                                            random_seed=random.randint(0,10000))
    
    fp_vert1_512x424 = _create_boxpose_dataset_factory_params(out_resolution)
    
    datasets.append(create_box_pose_dataset(render_params_vert1_512x424, intrinsics_params_vert1_512x424, pose_params_vert1_512x424, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert1_512x424)) 

  if dataset_type & BoxPoseDatasetType.VERTICAL_2:
    # vertical dataset #2, 640x360
    render_params_vert2_640x360 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert2_640x360 = IntrinsicsGeneratorParams(640,360, random.randint(0,10000))
    pose_params_vert2_640x360 = PoseSamplerParams(num_positions = sample_count_per_dataset,
                                            rmin = rmin,
                                            rmax = rmax,
                                            zmin = -0.35,
                                            zmax = 1.0,
                                            look_at_radius = 0.5,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.VERTICAL_2,
                                            random_seed=random.randint(0,10000))
    
    fp_vert2_640x360 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert2_640x360, intrinsics_params_vert2_640x360, pose_params_vert2_640x360, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert2_640x360)) 
    # vertical dataset #2, 640x480
    render_params_vert2_640x480 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert2_640x480 = IntrinsicsGeneratorParams(640,480, random.randint(0,10000))
    pose_params_vert2_640x480 = PoseSamplerParams(num_positions = sample_count_per_dataset,
                                            rmin = rmin,
                                            rmax = rmax,
                                            zmin = -0.35,
                                            zmax = 1.0,
                                            look_at_radius = 0.5,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.VERTICAL_2,
                                            random_seed=random.randint(0,10000))
    
    fp_vert2_640x480 = _create_boxpose_dataset_factory_params(out_resolution)
    datasets.append(create_box_pose_dataset(render_params_vert2_640x480, intrinsics_params_vert2_640x480, pose_params_vert2_640x480, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert2_640x480))
    # vertical dataset #2, 512x424
    render_params_vert2_512x424 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert2_512x424 = IntrinsicsGeneratorParams(512,424, random.randint(0,10000))
    pose_params_vert2_512x424 = PoseSamplerParams(num_positions = sample_count_per_dataset,
                                            rmin = rmin,
                                            rmax = rmax,
                                            zmin = -0.35,
                                            zmax = 1.0,
                                            look_at_radius = 0.5,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.VERTICAL_2,
                                            random_seed=random.randint(0,10000))
    
    fp_vert2_512x424 = _create_boxpose_dataset_factory_params(out_resolution)
    datasets.append(create_box_pose_dataset(render_params_vert2_512x424, intrinsics_params_vert2_512x424, pose_params_vert2_512x424, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert2_512x424))

  return datasets

def create_rs2_16_9_box_pose_dataset(                                    
                                    path_to_corbs_background_dataset    : str,
                                    path_to_vcl_background_dataset      : str,
                                    path_to_intnet_background_dataset   : str,
                                    pose_params                         : PoseSamplerParams,
                                    box_render_flags                    : BoxRenderFlags = None,
                                    dataset_type                        : BoxPoseDatasetType = BoxPoseDatasetType.HORIZONTAL,
                                    out_resolution_width                : int = 320,
                                    out_resolution_height               : int = 180 ) -> list:
    
  out_resolution = (out_resolution_width,out_resolution_height)
  datasets = []
  up_vector_variance = 10.0
  rmin = 1.5
  rmax = 2.5
  if dataset_type & BoxPoseDatasetType.HORIZONTAL:
    # horizontal dataset, 320x180
    render_params_horiz_320x180 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_horiz_320x180 = IntrinsicsGeneratorParams(320,180, random.randint(0,10000))
    pose_params_horiz_320x180 = PoseSamplerParams(num_positions = pose_params.num_positions,
                                            rmin = pose_params.rmin,
                                            rmax = pose_params.rmax,
                                            zmin = pose_params.heightmin,
                                            zmax = pose_params.heightmax,
                                            look_at_radius = pose_params.look_at_radius,
                                            up_vector_variance = pose_params.up_vector_variance,
                                            pose_type = PoseType.HORIZONAL,
                                            random_seed=random.randint(0,10000))
    
    fp_horiz_320x180 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_horiz_320x180, intrinsics_params_horiz_320x180, pose_params_horiz_320x180, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_horiz_320x180))

    # horizontal dataset, 1280x720
    render_params_horiz_1280x720 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_horiz_1280x720 = IntrinsicsGeneratorParams(1280,720, random.randint(0,10000))
    pose_params_horiz_1280x720 = PoseSamplerParams(num_positions = pose_params.num_positions,
                                            rmin = pose_params.rmin,
                                            rmax = pose_params.rmax,
                                            zmin = pose_params.heightmin,
                                            zmax = pose_params.heightmax,
                                            look_at_radius = pose_params.look_at_radius,
                                            up_vector_variance = pose_params.up_vector_variance,
                                            pose_type = PoseType.HORIZONAL,
                                            random_seed=random.randint(0,10000))
    
    fp_horiz_1280x720 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_horiz_1280x720, intrinsics_params_horiz_1280x720, pose_params_horiz_1280x720, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_horiz_1280x720))

  if dataset_type & BoxPoseDatasetType.VERTICAL_1:
    # vertical dataset #1, 320x180
    render_params_vert1_320x180 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert1_320x180 = IntrinsicsGeneratorParams(320,180, random.randint(0,10000))
    pose_params_vert1_320x180 = PoseSamplerParams(num_positions = pose_params.num_positions,
                                            rmin = pose_params.rmin,
                                            rmax = pose_params.rmax,
                                            zmin = pose_params.heightmin,
                                            zmax = pose_params.heightmax,
                                            look_at_radius = pose_params.look_at_radius,
                                            up_vector_variance = pose_params.up_vector_variance,
                                            pose_type = PoseType.VERTICAL_1,
                                            random_seed=random.randint(0,10000))
    
    fp_vert1_320x180 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert1_320x180, intrinsics_params_vert1_320x180, pose_params_vert1_320x180, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert1_320x180))

    # vertical dataset #1, 1280x720
    render_params_vert1_1280x720 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert1_1280x720 = IntrinsicsGeneratorParams(1280,720, random.randint(0,10000))
    pose_params_vert1_1280x720 = PoseSamplerParams(num_positions = pose_params.num_positions,
                                            rmin = pose_params.rmin,
                                            rmax = pose_params.rmax,
                                            zmin = pose_params.heightmin,
                                            zmax = pose_params.heightmax,
                                            look_at_radius = pose_params.look_at_radius,
                                            up_vector_variance = pose_params.up_vector_variance,
                                            pose_type = PoseType.VERTICAL_1,
                                            random_seed=random.randint(0,10000))
    
    fp_vert1_1280x720 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert1_1280x720, intrinsics_params_vert1_1280x720, pose_params_vert1_1280x720, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert1_1280x720))

  if dataset_type & BoxPoseDatasetType.VERTICAL_2:
    # vertical dataset #2, 320x180
    render_params_vert2_320x180 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert2_320x180 = IntrinsicsGeneratorParams(320,180, random.randint(0,10000))
    pose_params_vert2_320x180 = PoseSamplerParams(num_positions = pose_params.num_positions,
                                            rmin = pose_params.rmin,
                                            rmax = pose_params.rmax,
                                            zmin = pose_params.heightmin,
                                            zmax = pose_params.heightmax,
                                            look_at_radius = pose_params.look_at_radius,
                                            up_vector_variance = pose_params.up_vector_variance,
                                            pose_type = PoseType.VERTICAL_2,
                                            random_seed=random.randint(0,10000))
    
    fp_vert2_320x180 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert2_320x180, intrinsics_params_vert2_320x180, pose_params_vert2_320x180, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert2_320x180))

    # vertical dataset #2, 1280x720
    render_params_vert2_1280x720 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert2_1280x720 = IntrinsicsGeneratorParams(1280,720, random.randint(0,10000))
    pose_params_vert2_1280x720 = PoseSamplerParams(num_positions = pose_params.num_positions,
                                            rmin = pose_params.rmin,
                                            rmax = pose_params.rmax,
                                            zmin = pose_params.heightmin,
                                            zmax = pose_params.heightmax,
                                            look_at_radius = pose_params.look_at_radius,
                                            up_vector_variance = pose_params.up_vector_variance,
                                            pose_type = PoseType.VERTICAL_2,
                                            random_seed=random.randint(0,10000))
    
    fp_vert2_1280x720 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert2_1280x720, intrinsics_params_vert2_1280x720, pose_params_vert2_1280x720, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert2_1280x720))
                    
  return datasets


def create_kinect_box_pose_dataset(                                    
                                    path_to_corbs_background_dataset    : str,
                                    path_to_vcl_background_dataset      : str,
                                    path_to_intnet_background_dataset   : str,
                                    pose_params                         : PoseSamplerParams,
                                    box_render_flags                    : BoxRenderFlags = None,
                                    dataset_type                        : BoxPoseDatasetType = BoxPoseDatasetType.HORIZONTAL,
                                    out_resolution_width                : int = 320,
                                    out_resolution_height               : int = 288 ) -> list:
    
  out_resolution = (out_resolution_width,out_resolution_height)
  datasets = []
  up_vector_variance = 10.0
  rmin = 1.5
  rmax = 2.5
  if dataset_type & BoxPoseDatasetType.HORIZONTAL:
    # horizontal dataset, 320x180
    render_params_horiz_320x180 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_horiz_320x180 = IntrinsicsGeneratorParams(out_resolution_width,out_resolution_height, random.randint(0,10000))
    pose_params_horiz_320x180 = PoseSamplerParams(num_positions = pose_params.num_positions,
                                            rmin = pose_params.rmin,
                                            rmax = pose_params.rmax,
                                            zmin = pose_params.heightmin,
                                            zmax = pose_params.heightmax,
                                            look_at_radius = pose_params.look_at_radius,
                                            up_vector_variance = pose_params.up_vector_variance,
                                            pose_type = PoseType.HORIZONAL,
                                            random_seed=random.randint(0,10000))
    
    fp_horiz_320x180 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_horiz_320x180, intrinsics_params_horiz_320x180, pose_params_horiz_320x180, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_horiz_320x180))

    # horizontal dataset, 1280x720
    # render_params_horiz_1280x720 = BoxRendererParams(render_flags = box_render_flags)
    # intrinsics_params_horiz_1280x720 = IntrinsicsGeneratorParams(1280,720, random.randint(0,10000))
    # pose_params_horiz_1280x720 = PoseSamplerParams(num_positions = pose_params.num_positions,
    #                                         rmin = pose_params.rmin,
    #                                         rmax = pose_params.rmax,
    #                                         zmin = pose_params.heightmin,
    #                                         zmax = pose_params.heightmax,
    #                                         look_at_radius = pose_params.look_at_radius,
    #                                         up_vector_variance = pose_params.up_vector_variance,
    #                                         pose_type = PoseType.HORIZONAL,
    #                                         random_seed=random.randint(0,10000))
    
    # fp_horiz_1280x720 = _create_boxpose_dataset_factory_params(out_resolution)

    # datasets.append(create_box_pose_dataset(render_params_horiz_1280x720, intrinsics_params_horiz_1280x720, pose_params_horiz_1280x720, path_to_corbs_background_dataset,
    #                 path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_horiz_1280x720))

  if dataset_type & BoxPoseDatasetType.VERTICAL_1:
    # vertical dataset #1, 320x180
    render_params_vert1_320x180 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert1_320x180 = IntrinsicsGeneratorParams(320,288, random.randint(0,10000))
    pose_params_vert1_320x180 = PoseSamplerParams(num_positions = pose_params.num_positions,
                                            rmin = pose_params.rmin,
                                            rmax = pose_params.rmax,
                                            zmin = pose_params.heightmin,
                                            zmax = pose_params.heightmax,
                                            look_at_radius = pose_params.look_at_radius,
                                            up_vector_variance = pose_params.up_vector_variance,
                                            pose_type = PoseType.VERTICAL_1,
                                            random_seed=random.randint(0,10000))
    
    fp_vert1_320x180 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert1_320x180, intrinsics_params_vert1_320x180, pose_params_vert1_320x180, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert1_320x180))

    # vertical dataset #1, 1280x720
    render_params_vert1_1280x720 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert1_1280x720 = IntrinsicsGeneratorParams(1280,720, random.randint(0,10000))
    pose_params_vert1_1280x720 = PoseSamplerParams(num_positions = pose_params.num_positions,
                                            rmin = pose_params.rmin,
                                            rmax = pose_params.rmax,
                                            zmin = pose_params.heightmin,
                                            zmax = pose_params.heightmax,
                                            look_at_radius = pose_params.look_at_radius,
                                            up_vector_variance = pose_params.up_vector_variance,
                                            pose_type = PoseType.VERTICAL_1,
                                            random_seed=random.randint(0,10000))
    
    fp_vert1_1280x720 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert1_1280x720, intrinsics_params_vert1_1280x720, pose_params_vert1_1280x720, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert1_1280x720))

  if dataset_type & BoxPoseDatasetType.VERTICAL_2:
    # vertical dataset #2, 320x180
    render_params_vert2_320x180 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert2_320x180 = IntrinsicsGeneratorParams(320,288, random.randint(0,10000))
    pose_params_vert2_320x180 = PoseSamplerParams(num_positions = pose_params.num_positions,
                                            rmin = pose_params.rmin,
                                            rmax = pose_params.rmax,
                                            zmin = pose_params.heightmin,
                                            zmax = pose_params.heightmax,
                                            look_at_radius = pose_params.look_at_radius,
                                            up_vector_variance = pose_params.up_vector_variance,
                                            pose_type = PoseType.VERTICAL_2,
                                            random_seed=random.randint(0,10000))
    
    fp_vert2_320x180 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert2_320x180, intrinsics_params_vert2_320x180, pose_params_vert2_320x180, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert2_320x180))

    # vertical dataset #2, 1280x720
    # render_params_vert2_1280x720 = BoxRendererParams(render_flags = box_render_flags)
    # intrinsics_params_vert2_1280x720 = IntrinsicsGeneratorParams(1280,720, random.randint(0,10000))
    # pose_params_vert2_1280x720 = PoseSamplerParams(num_positions = pose_params.num_positions,
    #                                         rmin = pose_params.rmin,
    #                                         rmax = pose_params.rmax,
    #                                         zmin = pose_params.heightmin,
    #                                         zmax = pose_params.heightmax,
    #                                         look_at_radius = pose_params.look_at_radius,
    #                                         up_vector_variance = pose_params.up_vector_variance,
    #                                         pose_type = PoseType.VERTICAL_2,
    #                                         random_seed=random.randint(0,10000))
    
    # fp_vert2_1280x720 = _create_boxpose_dataset_factory_params(out_resolution)

    # datasets.append(create_box_pose_dataset(render_params_vert2_1280x720, intrinsics_params_vert2_1280x720, pose_params_vert2_1280x720, path_to_corbs_background_dataset,
    #                 path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert2_1280x720))
                    
  return datasets


def create_rs2_16_9_grid_box_pose_dataset(
                                    path_to_corbs_background_dataset              : str,
                                    path_to_vcl_background_dataset                : str,
                                    path_to_intnet_background_dataset             : str,
                                    pose_params                                   : PoseSamplerParamsGrid,
                                    box_render_flags                              : BoxRenderFlags = None,
                                    dataset_type                                  : BoxPoseDatasetType = BoxPoseDatasetType.HORIZONTAL,
                                    out_resolution_width                          : int = 320,
                                    out_resolution_height : int = 180 ) -> list:
    
  #raise Exception("Pose parameters should be as in 16_9 dataset") 
  out_resolution = (out_resolution_width, out_resolution_height)
  datasets = []

  rmin = pose_params.rmin
  rmax = pose_params.rmax
  zmin = pose_params.heightmin
  zmax = pose_params.heightmax
  look_at_radius = pose_params.look_at_radius
  up_vector_variance = pose_params.up_vector_variance
  dr = pose_params.dr
  dz = pose_params.dz
  dphi = pose_params.dphi

  if dataset_type & BoxPoseDatasetType.HORIZONTAL:
    # horizontal dataset, 320x180
    render_params_horiz_320x180 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_horiz_320x180 = IntrinsicsGeneratorParams(320,180, random.randint(0,10000))
    pose_params_horiz_320x180 = PoseSamplerParamsGrid(
                                            rmin = rmin,
                                            rmax = rmax,
                                            dr = dr,
                                            zmin = zmin,
                                            zmax = zmax,
                                            dz = dz,
                                            look_at_radius = look_at_radius,
                                            dphi = dphi,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.HORIZONAL,
                                            random_seed=random.randint(0,10000))
    
    fp_horiz_320x180 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_horiz_320x180, intrinsics_params_horiz_320x180, pose_params_horiz_320x180, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_horiz_320x180))

    # horizontal dataset, 1280x720
    render_params_horiz_1280x720 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_horiz_1280x720 = IntrinsicsGeneratorParams(1280,720, random.randint(0,10000))
    pose_params_horiz_1280x720 = PoseSamplerParamsGrid(
                                            rmin = rmin,
                                            rmax = rmax,
                                            dr = dr,
                                            zmin = zmin,
                                            zmax = zmax,
                                            dz = dz,
                                            look_at_radius = look_at_radius,
                                            dphi = dphi,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.HORIZONAL,
                                            random_seed=random.randint(0,10000))
    
    fp_horiz_1280x720 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_horiz_1280x720, intrinsics_params_horiz_1280x720, pose_params_horiz_1280x720, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_horiz_1280x720))

  if dataset_type & BoxPoseDatasetType.VERTICAL_1:
    # vertical dataset #1, 320x180
    render_params_vert1_320x180 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert1_320x180 = IntrinsicsGeneratorParams(320,180, random.randint(0,10000))
    pose_params_vert1_320x180 = PoseSamplerParamsGrid(
                                            rmin = rmin,
                                            rmax = rmax,
                                            dr = dr,
                                            zmin = zmin,
                                            zmax = zmax,
                                            dz = dz,
                                            look_at_radius = look_at_radius,
                                            dphi = dphi,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.VERTICAL_1,
                                            random_seed=random.randint(0,10000))
    
    fp_vert1_320x180 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert1_320x180, intrinsics_params_vert1_320x180, pose_params_vert1_320x180, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert1_320x180))

    # vertical dataset #1, 1280x720
    render_params_vert1_1280x720 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert1_1280x720 = IntrinsicsGeneratorParams(1280,720, random.randint(0,10000))
    pose_params_vert1_1280x720 = PoseSamplerParamsGrid(
                                            rmin = rmin,
                                            rmax = rmax,
                                            dr = dr,
                                            zmin = zmin,
                                            zmax = zmax,
                                            dz = dz,
                                            look_at_radius = look_at_radius,
                                            dphi = dphi,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.VERTICAL_1,
                                            random_seed=random.randint(0,10000))
    
    fp_vert1_1280x720 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert1_1280x720, intrinsics_params_vert1_1280x720, pose_params_vert1_1280x720, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert1_1280x720))

  if dataset_type & BoxPoseDatasetType.VERTICAL_2:
    # vertical dataset #2, 320x180
    render_params_vert2_320x180 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert2_320x180 = IntrinsicsGeneratorParams(320,180, random.randint(0,10000))
    pose_params_vert2_320x180 = PoseSamplerParamsGrid(
                                            rmin = rmin,
                                            rmax = rmax,
                                            dr = dr,
                                            zmin = zmin,
                                            zmax = zmax,
                                            dz = dz,
                                            look_at_radius = look_at_radius,
                                            dphi = dphi,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.VERTICAL_2,
                                            random_seed=random.randint(0,10000))    
    
    fp_vert2_320x180 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert2_320x180, intrinsics_params_vert2_320x180, pose_params_vert2_320x180, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert2_320x180))

    # vertical dataset #2, 1280x720
    render_params_vert2_1280x720 = BoxRendererParams(render_flags = box_render_flags)
    intrinsics_params_vert2_1280x720 = IntrinsicsGeneratorParams(1280,720, random.randint(0,10000))
    pose_params_vert2_1280x720 = PoseSamplerParamsGrid(
                                            rmin = rmin,
                                            rmax = rmax,
                                            dr = dr,
                                            zmin = zmin,
                                            zmax = zmax,
                                            dz = dz,
                                            look_at_radius = look_at_radius,
                                            dphi = dphi,
                                            up_vector_variance = up_vector_variance,
                                            pose_type = PoseType.VERTICAL_2,
                                            random_seed=random.randint(0,10000))
        
    fp_vert2_1280x720 = _create_boxpose_dataset_factory_params(out_resolution)

    datasets.append(create_box_pose_dataset(render_params_vert2_1280x720, intrinsics_params_vert2_1280x720, pose_params_vert2_1280x720, path_to_corbs_background_dataset,
                    path_to_vcl_background_dataset, path_to_intnet_background_dataset, fp_vert2_1280x720))
                    
  return datasets