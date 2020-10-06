import numpy as np

def deproject_depthmap(
    depthmap            :   np.ndarray, #hxw
    intrinsics          :   np.ndarray, #3x3
    distortion_coeffs   :   dict
):
    '''Caution, from now on we try to keep the axii as x,y,z'''
    h,w = depthmap.shape
    grid2D = np.stack(
        [
            np.tile(np.arange(w).reshape(w,1),(1,h)).astype(depthmap.dtype),
            np.tile(np.arange(h).reshape(1,h),(w,1)).astype(depthmap.dtype),
        ],
        0
    )

    if distortion_coeffs is not None:
        grid2D[0,:,:] = (grid2D[0,:,:] - intrinsics[0,2]) / intrinsics[0,0]
        grid2D[1,:,:] = (grid2D[1,:,:] - intrinsics[1,2]) / intrinsics[1,1] 
        homogeneous_sq = grid2D * grid2D
        radius_sq = homogeneous_sq[0,:,:] + homogeneous_sq[1,:,:]
        radius_vec = np.stack([
            radius_sq,
            radius_sq * radius_sq,
            radius_sq * radius_sq * radius_sq
        ], 0)

        a = 1.0 + np.einsum("i, ijk -> jk", distortion_coeffs["radial"][:3], radius_vec)
        b = 1.0 + np.einsum("i, ijk -> jk", distortion_coeffs["radial"][3:], radius_vec)
        a = np.where(a!=0.0, 1.0/a, 1.0)
        d = a * b
        homogeneous_dist = grid2D * d
        homogeneous_2xy = 2.0 * homogeneous_dist[0,:,:] * homogeneous_dist[1,:,:]
        homogeneous_dist_sq = homogeneous_dist * homogeneous_dist

        temporary1 = np.transpose(np.tile(np.flip(distortion_coeffs["tangential"]), (w,h,1)), (2,0,1)) *\
            (3.0 * homogeneous_dist_sq + np.stack([homogeneous_dist_sq[1,:,:],homogeneous_dist_sq[0,:,:]]))

        temporary2 = np.expand_dims(homogeneous_2xy,0) *\
                np.transpose(np.tile(distortion_coeffs["tangential"],(w,h,1)), (2,0,1))


        homogeneous_dist -= temporary1 + temporary2

        grid2D = homogeneous_dist
        ones = np.expand_dims(np.ones_like(np.transpose(depthmap)), 0)
        grid3D = np.concatenate([grid2D,ones], 0) #### ATTENTION
        #grid3D = np.concatenate([-1.0 * grid2D,ones], 0) #### ATTENTION
        pointcloud = grid3D * np.transpose(depthmap)
    else:

        ones = np.expand_dims(np.ones_like(np.transpose(depthmap)), 0)
        #grid3D = np.concatenate([-1.0 * grid2D,ones], 0) #### ATTENTION
        grid3D = np.concatenate([grid2D,ones], 0) #### ATTENTION
        pointcloud = np.linalg.inv(intrinsics).dot(grid3D.reshape(3,-1)).reshape(3,w,h) * np.transpose(depthmap)

    return np.transpose(pointcloud, (1,2,0))

def compute_normals(
    pointcloud      :           np.ndarray, #wxhx3
):
    pointcloud_c = pointcloud.copy()
    pointcloud_c = np.transpose(pointcloud_c,(2,0,1))
    #pointcloud = np.linalg.inv(intrinsics).dot(grid.reshape(3,-1)).reshape(3,w,h)
    points_temp = np.pad(pointcloud_c, ((0,0),(0,1),(0,0)), mode = 'edge')
    dx = points_temp[:, :-1, :] - points_temp[:, 1:, :]  # NCHW
    points_temp = np.pad(pointcloud_c, ((0,0),(0,0),(0,1)), mode = 'edge')
    dy = points_temp[:, :, :-1] - points_temp[:, :, 1:]  # NCHW
    normals = np.transpose(np.cross(np.transpose(dy.reshape(3,-1)),np.transpose(dx.reshape(3,-1)))).reshape(pointcloud_c.shape)
    normals = np.where(pointcloud_c[2,:,:] == 0.0, np.zeros_like(normals), normals / (np.linalg.norm(normals, axis = 0) + 1.0e-5))
    return np.transpose(normals, (1,2,0))


if __name__ == "__main__":
    import cv2
    intrinsics = np.array([[200.0, 0, 90.0], [0.0, 200.0, 160.0], [0,0,1]])
    filename = r"D:\Projects\vsc\multisensorsetup\multisensor_calibration\crf_refinement\M72e_Depth.pgm"
    depthmap = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    x = np.arange(16).reshape(4,4)
    compute_normals(depthmap, intrinsics)
    bp = True