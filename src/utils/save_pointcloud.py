import numpy as np

def save_ply(filename, tensor, scale, color=[0,0,0] , normals = None):    
    #w,h,c = tensor.shape
    if len(tensor.shape) == 2:
        tensor = np.expand_dims(tensor, 0)
    
    h,w,c = tensor.shape
    x_coords = tensor[:, :, 0] * scale
    y_coords = tensor[:, :, 1] * scale
    z_coords = tensor[:, :, 2] * scale
    if normals is not None:
        if len(normals.shape) == 2:
            normals = np.expand_dims(normals, 0)        
        nx_coords = normals[:, :, 0]
        ny_coords = normals[:, :, 1]
        nz_coords = normals[:, :, 2]
    with open(filename, "w") as ply_file:        
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex {}\n".format(w * h))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        if normals is not None:
            ply_file.write('property float nx\n')
            ply_file.write('property float ny\n')
            ply_file.write('property float nz\n')
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")
        
        if normals is None:
            for x in np.arange(h):
                for y in np.arange(w):
                    ply_file.write("{} {} {} {} {} {}\n".format(\
                        x_coords[x, y], y_coords[x, y], z_coords[x, y],\
                        color[0],color[1],color[2]
                        ))
        else:
            for x in np.arange(h):
                for y in np.arange(w):
                    ply_file.write("{} {} {} {} {} {} {} {} {}\n".format(\
                        x_coords[x, y], y_coords[x, y], z_coords[x, y],\
                        nx_coords[x, y], ny_coords[x, y], nz_coords[x, y],\
                        color[0],color[1],color[2]))