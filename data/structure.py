from torch.utils.data.dataset import Dataset

import os


class StructureDataLoadParameters(object):
    def __init__(self, path, depth, normal, label, pose):
        self.path = path
        self.load_depth = depth
        self.load_normal = normal
        self.load_label = label
        self.load_pose = pose
        

class StructureData(Dataset):
    #360D Dataset#
    def __init__(self, params):
        super(StructureData, self).__init__()
        files = os.listdir(params.path)
        if params.load_depth:
            depths = [f for f in files if f.endswith("_depth.exr")]
        if params.load_normal:
            normals = [f for f in files if f.endswith("_normal.exr")]
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
