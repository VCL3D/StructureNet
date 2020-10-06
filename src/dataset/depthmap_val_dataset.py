import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset

def get_modality(x):
    curr_split = x.split(".")
    if curr_split[-1] == "pgm":
        return 0
    else:
        label_split = x.split("_")[-2]
        if label_split == "id":
            return 1
        else:
            return 2





class DepthmapDatasetParams:
    def __init__(self, path_to_dataset : str, scale : float, max_len : int = None, number_of_classes = 17, valset = True):
        '''
        scale: value to scale loaded dataset (to adjust for metersmillimeters)
        max_len: use this to trim dataset to first max_len elements
        '''
        self.path_to_dataset = path_to_dataset        
        self.scale = scale
        self.max_len = max_len
        self.number_of_classes = number_of_classes
        self.valset = valset

class DepthmapDataset(Dataset):

    def __init__(self, params : DepthmapDatasetParams):
        super().__init__()
        self._params = params
        self._get_filelist()
    
    def __len__(self):
        return len(self._filepaths_depth) if (self._params.max_len == None) or (len(self._filepaths_depth) < self._params.max_len) else self._params.max_len

    def _get_filelist(self):
        self._filepaths = list(map(lambda x: os.path.join(self._params.path_to_dataset,x),os.listdir(self._params.path_to_dataset)))
        self._names = list(os.listdir(self._params.path_to_dataset))
        self._filepaths_depth = []
        self._filepaths_labels = []

        # for file in self._filepaths:
        #     mode = get_modality(file)
        #     if mode == 0:
        #         self._filepaths_depth.append(file)
        #     elif mode == 1:
        #         self._filepaths_labels.append(file)
        #     else:
        #         continue

        if not self._params.valset:
            self._filepaths_depth = \
                [os.path.join(self._params.path_to_dataset,x) \
                for x in sorted(self._names, key=lambda x: int(x.split('.')[0].split("_")[2]), reverse=False) if "pgm" in x]
            
            self._filepaths_labels = \
                [os.path.join(self._params.path_to_dataset,x) \
                for x in sorted(self._names, key=lambda x: int(x.split('.')[0].split("_")[2]), reverse=False) if "id" in x]

            self._names = \
                [x \
                for x in sorted(self._names, key=lambda x: int(x.split('.')[0].split("_")[2]), reverse=False) if "id" in x]
        else:
            self._filepaths_depth = \
                [os.path.join(self._params.path_to_dataset,x) \
                for x in sorted(self._names, key=lambda x: x.split('.')[0].split("_")[0], reverse=False) if "pgm" in x]
            
            self._filepaths_labels = \
                [os.path.join(self._params.path_to_dataset,x) \
                for x in sorted(self._names, key=lambda x: x.split('.')[0].split("_")[0], reverse=False) if "id" in x]

            self._names = \
                [x \
                for x in sorted(self._names, key=lambda x: x.split('.')[0].split("_")[0], reverse=False) if "id" in x]

        

    def __getitem__(self,idx):

        frame = {}
        frame["depth"] = {}
        frame["labels"] = {}
        dname = self._filepaths_depth[idx]
        lname = self._filepaths_labels[idx]
        frame["name"] = self._names[idx].split(".")[0]
        img = cv2.imread(dname,cv2.IMREAD_ANYDEPTH).astype(np.float32) * self._params.scale
        img_labels = cv2.imread(lname,cv2.IMREAD_ANYDEPTH).astype(np.float32)
        
        frame["depth"] = torch.from_numpy(img).unsqueeze(0)

        labels = torch.from_numpy(img_labels).unsqueeze(0)

        if self._params.number_of_classes == 17:
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
        elif self._params.number_of_classes == 21:
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




        frame["labels"] = labels

        return frame  


        # #set background as class 17
        # labels[labels == 16] = 99
        # labels[labels == 0] = 16
        # labels[labels == 1] = 0
        # labels[labels == 2] = 1
        # labels[labels == 3] = 2
        # labels[labels == 4] = 3
        # labels[labels == 5] = 4
        # labels[labels == 6] = 5
        # labels[labels == 7] = 6
        # labels[labels == 8] = 7
        # labels[labels == 9] = 8
        # labels[labels == 10] = 9
        # labels[labels == 11] = 10
        # labels[labels == 12] = 11
        # labels[labels == 13] = 12
        # labels[labels == 14] = 13
        # labels[labels == 15] = 14
        # labels[labels == 99] = 15

        # labels += 1