import torch

def transform_points(points, rotation, translation):
    b, _, h, w = points.size()  # [B, 3, H, W]
    points3d = points.reshape(b, 3, -1)  # [B, 3, H*W]
    return (
        (rotation @ points3d) # [B, 3, 3] * [B, 3, H*W]
        + translation # [B, 3, 1]
    ).reshape(b, 3, h, w)  # [B, 3, H, W]

def transform_points_homogeneous(points, transformation_matrix):
    b, c, h, w = points.size()  # [B, 4, H, W]
    if c == 3:
        points_homogeneous = torch.cat([points, torch.ones(b,1,h,w).type_as(points).to(points.device)], dim = 1)
    points_homogeneous = points_homogeneous.reshape(b, 4, -1)  # [B, 4, H*W]
    return (
        (transformation_matrix @ points_homogeneous) # [B, 4, 4] * [B, 4, H*W]
    ).reshape(b,4,h,w)[:,:3,:,:]  # [B, 3, H, W]

def extract_rotation_translation(pose):
    b, _, _ = pose.shape
    return pose[:, :3, :3].clone(), pose[:,:3, 3].reshape(b, 3, 1).clone() # rotation, translation

def transform_normals(oriented_points , rotation):
    b, _ , _ , _ = oriented_points.size()
    return transform_points(oriented_points , rotation , torch.zeros((b , 3 , 1)).type_as(oriented_points))

def rotatePointsAboutAxisXYZ(
    angle           : float,
    axis            : int, #x,y,z
    points          : torch.tensor # N x 3
):
    from numpy import sin, cos
    b, _ , h, w = points.shape
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

    rotated = torch.bmm(R.unsqueeze(0).expand(b,3,3) , points.view(b,3,-1)).view(b,3,h,w)

    return rotated