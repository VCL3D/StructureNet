import sys
import tinyobjloader
from enum import Flag, auto

class BoxLoadFlags(Flag) :
    
    LOAD_SIDES = auto()
    LOAD_UP = auto()
    LOAD_DOWN = auto()
    LOAD_ALL = LOAD_SIDES | LOAD_UP | LOAD_DOWN

'''
obj loader to load box model
returns dictionary
{
    shape_names : [string list]    
    vertices : 3 coordinates per vertex , 4 vertices per side, 24 sides total,  vertices of side i at i*4*3 .. (i*4+3)*3
    normals: 4 normals per side, one for each side vertex, normals of side i at i*4*3 .. (i*4+3)*3
    indices: 6 indices perside (2 triangles), indices of side i at: 6*i ...  6*i+5
}
the units of the model are in millimeters (mm)
'''
def load_box_model(path_to_model_obj : str, flags : BoxLoadFlags = BoxLoadFlags.LOAD_ALL):
    # Create reader.
    reader = tinyobjloader.ObjReader()    

    # Load .obj(and .mtl) using default configuration
    ret = reader.ParseFromFile(path_to_model_obj)

    if ret == False:
        print("Warn:", reader.Warning())
        print("Err:", reader.Error())
        print("Failed to load : ", path_to_model_obj)

        sys.exit(-1)

    if reader.Warning():
        print("Warn:", reader.Warning())

    attrib = reader.GetAttrib()
    shapes = reader.GetShapes()

    def _filter_shape(shape_name : str):
        if (flags & BoxLoadFlags.LOAD_DOWN) and ("_down_" in shape_name):
            return True
        elif (flags & BoxLoadFlags.LOAD_UP) and ("_up_" in shape_name):
            return True
        elif (flags & BoxLoadFlags.LOAD_SIDES) and (("_front_" in shape_name) or ("_back_" in shape_name) or ("_left_" in shape_name) or ("_right_" in shape_name)):
            return True

        return False
        

    filtered_shapes = list(filter(lambda x: _filter_shape(x.name),shapes))

    side_names = []
    vertices = []
    normals = []
    indices = []

    index_map = dict()

    for shape in filtered_shapes:
        side_names.append(shape.name)        
        for idx in shape.mesh.indices:
            
            if(idx.vertex_index in index_map):
                index = index_map[idx.vertex_index]
            else:
                newidx = len(index_map)
                index_map[idx.vertex_index] = newidx
                index = newidx

                # for all coordinates (x,y,z)
                for j in range(3):
                    vertices.append(attrib.vertices[idx.vertex_index*3+j])
                    normals.append(attrib.normals[idx.normal_index*3+j])

            indices.append(index)            

    box_model = {
        "side_names" : side_names,
        "vertices" : vertices,
        "normals" : normals,
        "indices" : indices,
        "index_map" : index_map
    }

    return box_model

    
