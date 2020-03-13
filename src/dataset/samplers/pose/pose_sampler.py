import numpy
from .pose_sampler_utils import *
from enum import Enum

class PoseType(Enum):

    HORIZONAL = 1,
    VERTICAL_1 = 2,
    VERTICAL_2 = 3

#TODO: rename to PoseSamplerParamsRandom or something like this
class PoseSamplerParams(object):
    '''
        Parameters for @PoseSampler class. 
        Parameters define a camera position and a camera "look-at".
        Camera position is represented in cylidrical coordinates, with
        @param r in [rmin,rmax] : distance from the center
        @param z in [zmin,zmax] : height from z = 0 plane
        @param ef in [efmin, efmax] : polar angle (usually is in [0,2*pi)) 
        @param look_at_radius : if p (x,y,z) is a camera look at,then |x| < r ##TODO: this must be handled more carefully, not in a sphere
    '''
    def __init__(   self,
                    num_positions           : int,
                    rmin                    : float,
                    rmax                    : float,
                    zmin                    : float,
                    zmax                    : float,
                    look_at_radius          : float,
                    up_vector_variance      = 5.0, #degrees
                    phimin                  = 0.0,
                    phimax                  = 2 * numpy.pi,
                    pose_type               : PoseType = PoseType.HORIZONAL,
                    random_seed             = None
                ):
        self.num_positions = num_positions
        self.rmin = rmin
        self.rmax = rmax
        self.heightmin = zmin
        self.heightmax = zmax
        self.phimin = phimin
        self.phimax = phimax
        self.look_at_radius = look_at_radius
        self.up_vector_variance = up_vector_variance
        self.pose_type = pose_type
        self.random_seed = random_seed

class PoseSamplerParamsGrid(object):
    '''
        Parameters for @PoseSampler class, for grid sampling. 
        Parameters define a camera position and a camera "look-at".
        Camera position is represented in cylidrical coordinates, with
        @param r in [rmin,rmax] : distance from the center
        @param z in [zmin,zmax] : height from z = 0 plane
        @param ef in [efmin, efmax] : polar angle (usually is in [0,2*pi)) 
        @param look_at_radius : if p (x,y,z) is a camera look at,then |x| < r ##TODO: this must be handled more carefully, not in a sphere
    '''
    def __init__(   self,
                    rmin                    :float,
                    rmax                    :float,
                    dr                      :float,
                    zmin                    :float,
                    zmax                    :float,
                    dz                      :float,
                    look_at_radius          :float,
                    dphi                    :float,
                    up_vector_variance      = 5.0, #degrees
                    phimin                  = 0.0,
                    phimax                  = 2 * numpy.pi,
                    pose_type               : PoseType = PoseType.HORIZONAL,
                    random_seed             = None,
                ):
        self.rmin = rmin
        self.rmax = rmax
        self.dr = dr
        self.heightmin = zmin
        self.heightmax = zmax
        self.dz = dz
        self.phimin = phimin
        self.phimax = phimax
        self.dphi = dphi
        self.look_at_radius = look_at_radius
        self.up_vector_variance = up_vector_variance
        self.pose_type = pose_type
        self.random_seed = random_seed

class PoseSampler(object):
    '''
        @PoseSampler class that containts the samples
    '''
    def __init__(   self, 
                    params                      : PoseSamplerParams,
                ):
        self.rng = numpy.random.RandomState(seed = params.random_seed) # fix random seed in order to get data generation consistency
        
        self.params = params
        number_look_at_aug = 1

        up_direction = numpy.array([0.0, 1.0, 0.0])


        ### First generate camera positions (or camera centers) 
        if isinstance(self.params, PoseSamplerParams):
            number_of_samples = params.num_positions
            _R = self.rng.uniform(params.rmin, params.rmax, number_of_samples) #radius
            _phi = self.rng.uniform(params.phimin, params.phimax, number_of_samples) #phi 
            _y = self.rng.uniform(params.heightmin, params.heightmax, number_of_samples) #z
        elif isinstance(self.params, PoseSamplerParamsGrid): #GRID
            _R = numpy.arange(params.rmin, params.rmax, params.dr)
            _phi = numpy.arange(params.phimin, params.phimax, params.dphi)
            _y = numpy.arange(params.heightmin, params.heightmax, params.dz)

            Nr = len(_R)
            Nphi = len(_phi)
            Ny = len(_y)


            _phi = numpy.repeat(_phi, Nr * Ny)
            _R = numpy.tile(numpy.repeat(_R, Ny), Nphi)
            _y = numpy.tile(_y, Nr * Nphi)
            number_of_samples = len(_y)


        _x = numpy.multiply(_R, numpy.cos(_phi))
        _z = numpy.multiply(_R, numpy.sin(_phi))

        ########## Clean up
        del _R
        del _phi

        # Create camera positions in global space
        # N x 3
        _positions = numpy.concatenate(    (numpy.expand_dims(_x,axis = 1),
                                                numpy.expand_dims(_y,axis = 1),
                                                numpy.expand_dims(_z,axis = 1)),
                                                axis = 1) ### camera positions
        
        ######### Clean up
        del _x
        del _y
        del _z

        # Create "look at" targets for every camera position
        # number_look_at_aug * N x 3
        _look_at_augmentations = \
            generatePointsInCircle( number_look_at_aug,
                                    params.look_at_radius,
                                    _positions,
                                    rng_generator = self.rng)

        _positions = numpy.repeat(_positions,number_look_at_aug,axis=0)
        #_augmented_look_ats = _look_at_augmentations - _positions
        _augmented_look_ats = _positions - _look_at_augmentations
            

        
        # Create right handed camera coordinate system
        _right_vectors, _up_vectors, _augmented_look_ats = \
            createRightHandCartesian(_augmented_look_ats, up_direction)

        # Random angles for augmentation
        thetas = self.rng.uniform(
            -numpy.radians(params.up_vector_variance) / 2,
            numpy.radians(params.up_vector_variance) / 2,
            _up_vectors.shape[0])
        
        # Rotate every camera coordinate system along "look at" vector
        _up_vectors, _ = rotateVectorAboutAxis(_augmented_look_ats,thetas,_up_vectors)
        _right_vectors, _ = rotateVectorAboutAxis(_augmented_look_ats,thetas,_right_vectors)

        ########### Clean up
        del thetas

        #rotate about x axis
        _up_vectors, _ = rotateVectorAboutAxis(_right_vectors, numpy.asarray([numpy.pi]), _up_vectors)
        _augmented_look_ats, _ = rotateVectorAboutAxis(_right_vectors, numpy.asarray([numpy.pi]), _augmented_look_ats)
        
        if self.params.pose_type == PoseType.VERTICAL_1:
            _up_vectors, _ = rotateVectorAboutAxis(_augmented_look_ats, numpy.asarray([-numpy.pi/2]), _up_vectors)
            _right_vectors, _ = rotateVectorAboutAxis(_augmented_look_ats, numpy.asarray([-numpy.pi/2]), _right_vectors)
        elif self.params.pose_type == PoseType.VERTICAL_2:
            _up_vectors, _ = rotateVectorAboutAxis(_augmented_look_ats, numpy.asarray([numpy.pi/2]), _up_vectors)
            _right_vectors, _ = rotateVectorAboutAxis(_augmented_look_ats, numpy.asarray([numpy.pi/2]), _right_vectors)
        
        
        _rotation_matrices = numpy.concatenate((numpy.expand_dims(_right_vectors,2),
                                                    numpy.expand_dims(_up_vectors,2),
                                                    numpy.expand_dims(_augmented_look_ats,2)),axis=2)

        self.transformations = numpy.zeros((number_of_samples,4,4))
        self.transformations[:,:3,:3] = _rotation_matrices
        self.transformations[:,:3,3] = _positions
        self.transformations[:,3,:] = numpy.repeat(numpy.expand_dims(numpy.array([0.0,0.0,0.0,1.0]),0),number_of_samples ,axis = 0)
