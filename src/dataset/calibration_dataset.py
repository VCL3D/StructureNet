import torch
import json
import os
import cv2
import numpy

from torch.utils.data.dataset import Dataset

def correct_labels(labels, number_of_classes):
    if labels is None:
        return None
    
    if number_of_classes == 17:
            #map from 25 to 17 class labels
            #labels[labels == 1] = 1
            #labels[labels == 2] = 2
            labels[labels == 3] = 0
            labels[labels == 4] = 0
            labels[labels == 5] = 3
            labels[labels == 6] = 4


            labels[labels == 7] = 5
            labels[labels == 8] = 6
            labels[labels == 9] = 0
            labels[labels == 10] = 0
            labels[labels == 11] = 7
            labels[labels == 12] = 8

            labels[labels == 13] = 9
            labels[labels == 14] = 0
            labels[labels == 15] = 10
            labels[labels == 16] = 11
            labels[labels == 17] = 0
            labels[labels == 18] = 12

            labels[labels == 19] = 13
            labels[labels == 20] = 0
            labels[labels == 21] = 14
            labels[labels == 22] = 0
            labels[labels == 23] = 15
            labels[labels == 24] = 16
    elif number_of_classes == 21:
            #map from 25 to 21 class labels
            #aka bot as background
            #labels[labels == 1] = 1
            #labels[labels == 2] = 2
            labels[labels == 3] = 0
            labels[labels == 4] = 3
            labels[labels == 5] = 4
            labels[labels == 6] = 5


            labels[labels == 7] = 6
            labels[labels == 8] = 7
            labels[labels == 9] = 0
            labels[labels == 10] = 8
            labels[labels == 11] = 9
            labels[labels == 12] = 10

            labels[labels == 13] = 11
            labels[labels == 14] = 12
            labels[labels == 15] = 13
            labels[labels == 16] = 14
            labels[labels == 17] = 0
            labels[labels == 18] = 15

            labels[labels == 19] = 16
            labels[labels == 20] = 0
            labels[labels == 21] = 17
            labels[labels == 22] = 18
            labels[labels == 23] = 19
            labels[labels == 24] = 20
    return labels

def load_intrinsics_repository(filename):    
    #global intrinsics_dict
    with open(filename, 'r') as json_file:
        intrinsics_repository = json.load(json_file)
        intrinsics_dict = dict((intrinsics['Device'], \
            intrinsics['Depth Intrinsics'][0]['1280x720'])\
                for intrinsics in intrinsics_repository)
    return intrinsics_dict

def get_intrinsics(name, intrinsics_dict, scale=1, data_type=torch.float32):
    #global intrinsics_dict
    if intrinsics_dict is not None:
        intrinsics_data = numpy.array(intrinsics_dict[name])
        intrinsics = torch.tensor(intrinsics_data).reshape(3, 3).type(data_type)    
        intrinsics[0, 0] = intrinsics[0, 0] / scale
        intrinsics[0, 2] = intrinsics[0, 2] / scale
        intrinsics[1, 1] = intrinsics[1, 1] / scale
        intrinsics[1, 2] = intrinsics[1, 2] / scale
        intrinsics_inv = intrinsics.inverse()
        return intrinsics, intrinsics_inv
    raise ValueError("Intrinsics repository is empty")

class DepthmapDatasetCalibrationParams:
    def __init__(
        self,
        path_to_dataset         : str,
        path_to_device_repo     : str,
        scale                   : float,
        name_pos                : int,
        extension               : str,
        decimation_scale        : int,
        nclasses                : int,
        duplicate_devices       = False
        ):
        '''
        scale: value to scale loaded dataset (to adjust for metersmillimeters)
        name_pos: where in filename device name appears (for xxx_yyy_DEVICENAME_zzz.pgm name_pos = 2)
        '''
        self.path_to_dataset = path_to_dataset
        self.path_to_device_repo = path_to_device_repo
        self.scale = scale
        self.name_pos = name_pos
        self.extension = extension
        self.decimation_scale = decimation_scale
        self.duplicate_devices = duplicate_devices
        self.nclasses = nclasses




class DepthmapDatasetCalibration(Dataset):

    def __init__(self, params : DepthmapDatasetCalibrationParams):
        super().__init__()
        self._params = params        
        self._get_filelist()
        self.intrinsics_dict = load_intrinsics_repository(params.path_to_device_repo)
    
    def __len__(self):
        return len(self._filepaths)

    def _get_filelist(self):
        self._filepaths = [os.path.join(self._params.path_to_dataset, x) for x in os.listdir(self._params.path_to_dataset) if os.path.isfile(os.path.join(self._params.path_to_dataset,x)) and x.split(".")[1] == self._params.extension]
        self._labels = [os.path.join(self._params.path_to_dataset, x) for x in os.listdir(self._params.path_to_dataset) if os.path.isfile(os.path.join(self._params.path_to_dataset,x)) and (x.split(".")[1] == 'png') and ("label" in x)]

        self._filepaths.sort()
        self._labels.sort()

    def __getitem__(self,idx):

        fname = self._filepaths[idx]
        img = cv2.imread(fname,cv2.IMREAD_ANYDEPTH).astype(numpy.float32) * self._params.scale

        if self._labels:
            img_labels = torch.from_numpy(cv2.imread(self._labels[idx],cv2.IMREAD_ANYDEPTH).astype(numpy.float32)).unsqueeze(0)
        else:
            img_labels = torch.tensor([])

        device_name = os.path.basename(fname).split("_")[self._params.name_pos]

        intrinsics, intrinsics_inv  = get_intrinsics(device_name[:-1] if self._params.duplicate_devices else device_name, self.intrinsics_dict, self._params.decimation_scale)


        return {
            "depth"             :torch.from_numpy(img).unsqueeze(0),
            "filename"          :fname,
            "device"            :device_name,
            "intrinsics"        :intrinsics,
            "intrinsics_inv"    :intrinsics_inv,
            "labels"            :correct_labels(img_labels, self._params.nclasses),
            "has_labels"        :True if self._labels else False
            }

if __name__ == "__main__":
    params = DepthmapDatasetCalibrationParams(
        "D:\\VCL\\Users\\vlad\\Datasets\\SMPL_playground_data\\new_perfcap_recs\\akiz\\Dump\\Dataset\\root\\Data",
        "D:\\Projects\\vs\\RealSenz\\immerzion\\vs\\immerzion\\x64\\Release\\device_repository.json",
        0.001,
        1,
        "pgm",
        4)
    d = DepthmapDatasetCalibration(params)
    dataset =  torch.utils.data.DataLoader(d,\
            batch_size = 1, shuffle=False,\
            num_workers = 0, pin_memory=False)

    for batch_id, batch in enumerate(d):
        bbb = batch
        bp = True