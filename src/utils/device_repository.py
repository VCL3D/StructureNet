import numpy, json

def load_intrinsics_repository(filename):    
    with open(filename, 'r') as json_file:
        if True:
            intrinsics_repository = json.load(json_file)
            intrinsics_dict = dict((intrinsics['Device'], \
                intrinsics['Depth Intrinsics'][0]['1280x720'])\
                    for intrinsics in intrinsics_repository if "realsense" in intrinsics['Type'].lower())

            intrinsics_dict.update(dict((intrinsics['Device'], \
            	intrinsics['Depth Intrinsics'][3]['640x576'])\
                #intrinsics['Depth Intrinsics'][1]['320x288'])\
                    for intrinsics in intrinsics_repository if "azure" in intrinsics['Type'].lower()))
        else:
            pass

    return intrinsics_dict

def get_intrinsics(name, intrinsics_dict, scale_x = 1, scale_y = 1, data_type=numpy.float32):
    #global intrinsics_dict
    if intrinsics_dict is not None:
        intrinsics = numpy.array(intrinsics_dict[name]).reshape(3,3)
        intrinsics_c = intrinsics.copy()
        #intrinsics[1,1] *= -1.0
        intrinsics[0, 0] = intrinsics_c[0, 0] / scale_x
        intrinsics[0, 2] = intrinsics_c[0, 2] / scale_x
        intrinsics[1, 1] = intrinsics_c[1, 1] / scale_y
        intrinsics[1, 2] = intrinsics_c[1, 2] / scale_y
        return intrinsics
    raise ValueError("Intrinsics repository is empty")


def compute_scales(name, width, height, intrinsics_repository):
    entry = [x for x in intrinsics_repository if x["Device"]==name]
    assert len(entry)==1, name + " was not found in device_repository.json"
    entry = entry[0]

    if (str(width) + "x" + str(height)) in entry["Depth Intrinsics"]:
        return 1.0, 1.0
    else:
        for intrinsics in entry["Depth Intrinsics"]:
            intrinsics = list(intrinsics.keys())[0]
            _width, _height = intrinsics.split("x")
            _width, _height = int(_width), int(_height)

            if ((_width % width)==0) and ((_height % height)==0):
                return _width/width, _height/height
