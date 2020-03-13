import numpy
import torch ### in order to make batched operations


'''
    Computes camera transformation for a point
    @param camera_look_at : look at positions of a camera   Nx3
    @param camera_up_vector : camera up vector              Nx3
    @param camera_position : position of the camera         Nx3
    returns: rotation matrices                              Nx3x3
    http://ksimek.github.io/2012/08/22/extrinsic/ "Look at camera"
'''
def computeRotation(
    camera_look_at_position : numpy.array,      ### N x 3
    camera_up_vector        : numpy.array,      ### N x 3
    camera_position         : numpy.array       ### N x 3
):
    return_matrices = numpy.zeros((camera_position.shape[0],3,3))
    L = torch.from_numpy(camera_look_at_position) - torch.from_numpy(camera_position)
    L = torch.nn.functional.normalize(L)

    s = torch.cross(L, torch.from_numpy(camera_up_vector), dim = 1)
    s = torch.nn.functional.normalize(s)

    udot = torch.cross(s,L, dim = 1)

    return_matrices[:,0,:] = s
    return_matrices[:,1,:] = udot
    return_matrices[:,2,:] = -L

    return return_matrices

'''
    Computed the perpendicular plane for a vector as v = b - a,
    which passes through point a.
    @param a : point one            Nx3
    @param b : point two            Nx3
    returns: perpendicular planes   Nx4
    planes are defined as ax + by + cz + d = 0
'''
def computePerpendicularPlane(
    pointA          : numpy.array,          ### Nx3
    pointB          : numpy.array           ### Nx3
):
    torchA = torch.from_numpy(pointA)
    vectors = torch.from_numpy(pointB) - torchA #vectors abc's
    d = torch.bmm(vectors.unsqueeze(-1).view(-1,1,3),
     - torchA.unsqueeze(-1).view(-1,3,1)).squeeze(-1)
    return torch.cat([vectors,d], dim = 1).numpy()

'''
    Given a vector, find 2 random perpendicular vectors.
    @param vector : input vector
    returns x,y : perpendicular vectors
'''
def computePerpendicularVectors(
    vector          : numpy.array           ### Nx3
):
    vectorTorch = torch.nn.functional.normalize(torch.from_numpy(vector)).float()
    x = torch.randn((vectorTorch.shape))    # Same shape as input
    x -= torch.bmm(x.unsqueeze(-1).view(-1,1,3),
        vectorTorch.unsqueeze(-1).view(-1,3,1)).squeeze(-1) * vectorTorch

    y = torch.cross(vectorTorch, x , dim = 1)

    return torch.nn.functional.normalize(x).numpy(),torch.nn.functional.normalize(y).numpy()

'''
    Given a random vector v, produce N points that lay in
    a cirle of radius R perpendicular to v
    @param N : number of points to produce
    @param R : radius of circle
    @param V : perpendicular vector as discribed
    returns : a set of N*M x 3 points
'''
def generatePointsInCircle(
    N               : int,
    R               : float,
    vector          : numpy.array,           ### Mx3
    rng_generator   = numpy.random
):
    R = float(R)
    M = vector.shape[0]
    pX, pY = computePerpendicularVectors(vector)
    points = numpy.zeros((M*N,3), dtype = numpy.float)
    random_factor = rng_generator.uniform(0,1,(M*N,2))
    random_angle = rng_generator.uniform(0,2*numpy.pi,(M*N))
    for i in range(3):
        points[:,i] = \
        R*random_factor[:,0]*numpy.cos(random_angle)*numpy.repeat(pX[:,i],N) + \
        R*random_factor[:,1]*numpy.sin(random_angle)*numpy.repeat(pY[:,i],N)
    return points
    


'''
    Rotate vectors along axis by angle.
'''
def rotateVectorAboutAxisXYZ(
    angle           : float,
    axis            : int, #x,y,z
    vectors         : numpy.array # N x 3
):
    from numpy import sin, cos
    vectorsTorch = torch.from_numpy(vectors).unsqueeze(1)
    b, _ , _ = vectorsTorch.shape
    if axis == 0:
        R = torch.tensor([
            [1,0,0],
            [0, cos(angle), -sin(angle)],
            [0, sin(angle), cos(angle)]
            ]).double()
    elif axis == 1:
        R = torch.tensor([
            [cos(angle),0,sin(angle)],
            [0, 1, 0],
            [-sin(angle), 0, cos(angle)]
            ]).double()
    elif axis == 2:
        R = torch.tensor([
            [cos(angle),-sin(angle),0],
            [sin(angle), cos(angle), 0],
            [0, 0, 1]
            ]).double()

    if vectors is not None:
        rotated = torch.bmm(vectorsTorch, R.unsqueeze(0).expand(b,3,3))
        return rotated.numpy(), R
    else:
        return None , R

def rotateVectorAboutAxis(
    axii            : numpy.array,
    thetas          : numpy.array,
    vectors         : numpy.array # N x 3
):
    #import math
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    #axis = np.asarray(axis)
    #axis = axis / math.sqrt(np.dot(axis, axis))
    torchaxii = torch.from_numpy(axii)
    if len(torchaxii.shape) == 1:
        torchaxii = torchaxii.unsqueeze(0)
    axii = torch.nn.functional.normalize(torchaxii).numpy() #Nx3
    a = numpy.cos(thetas / 2.0)
    T = -axii * numpy.repeat(numpy.expand_dims(numpy.sin(thetas / 2.0),1),3,axis=1)
    b = T[:,0]
    c = T[:,1]
    d = T[:,2]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    R = numpy.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    if vectors is not None:
        if len(vectors.shape) == 1:
            vectors_torch = torch.from_numpy(vectors).unsqueeze(0)
        else:
            vectors_torch = torch.from_numpy(vectors)
        return torch.bmm(
            torch.from_numpy(R).permute(2,0,1),
            vectors_torch.unsqueeze(-1).type_as(torch.from_numpy(R))).squeeze(-1).numpy(), R
    else:
        return R

def createRightHandCartesian(
    look_at             : numpy.array, # N x 3
    up_direction        : numpy.array  # 3
):
    look_at_n = torch.nn.functional.normalize(torch.from_numpy(look_at))
    n, _ = look_at.shape
    right_vector = torch.cross(torch.from_numpy(up_direction).unsqueeze(0).expand(n,3),
    look_at_n, dim = 1)
    right_vector = torch.nn.functional.normalize(right_vector)
    return \
        right_vector.numpy(),\
        torch.nn.functional.normalize(torch.cross(look_at_n, right_vector, dim = 1)).numpy(),\
        look_at_n.numpy()


    

