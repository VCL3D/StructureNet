import numpy as np

def get_distortion_coefficients(name, intrinsics_repository):
    entry = [x for x in intrinsics_repository if x["Device"]==name]
    assert len(entry)==1
    entry = entry[0]

    if len([x for x in list(entry.keys()) if "distortion" in x.lower()]):
        radial_distortion_parameters = entry["Depth Radial Distortion Coeffs"]
        tangetial_distortion_parameters = entry["Depth Tangential Distortion Coeffs"]


        return {"radial" : np.array(radial_distortion_parameters),
        "tangential": np.array(tangetial_distortion_parameters)}

    return None