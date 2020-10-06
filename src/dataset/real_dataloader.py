import os
import sys
import torch
import numpy
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import warnings
import json
import cv2
#testing
# sys.path.append('E:\\Projects\\vsc\\deep_depth_denoising\\denoise')
# import importers

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


'''
Dataset importer. We assume that data follows the below structure.
root_path
device_repository.json
	|
	|-----recording_i
	|		|-----Data
	|		|-----Calibration
	|
	|-----recording_i+1
	|		|-----Data
	|		|-----Calibration
	|
'''

class DataLoaderParams:
	def __init__(self
	,root_path 
	,device_list
	,decimation_scale = 2
	,device_repository_path = "."
	,depth_scale = 0.001
	,depth_threshold = 5):
		self.root_path = root_path
		self.device_list = device_list
		self.device_repository_path = device_repository_path
		self.depth_scale = depth_scale
		self.decimation_scale = decimation_scale
		self.depth_threshold = depth_threshold

class DataLoad(Dataset):
	def __init__(self, params):
		super(DataLoad,self).__init__()
		self.params = params
		
		device_repo_path = os.path.join(self.params.device_repository_path,"device_repository.json")
		if not os.path.exists(device_repo_path):
			raise ValueError("{} does not exist, exiting.".format(device_repo_path))
		self.device_repository = load_intrinsics_repository(device_repo_path)

		root_path = self.params.root_path
		

		if not os.path.exists(root_path):
			#TODO maybe log?
			raise ValueError("{} does not exist, exiting.".format(root_path))

		self.data = {}

		#Iterate over each recorded folder
		for recording in os.listdir(root_path):
			abs_recording_path = os.path.join(root_path,recording)
			if not os.path.isdir(abs_recording_path):
				continue
			#Path where data supposed to be stored
			data_path = os.path.join(abs_recording_path,"Data")

			# if not os.path.exists(data_path):
			# 	warnings.warn("Folder {} does not containt \"Data\" folder".format(abs_recording_path))
			# 	continue
			
			#Path to the calibration of that particular recording
			# calibration_path = os.path.join(abs_recording_path,"Calibration")
			# if not os.path.exists(calibration_path):
			# 	warnings.warn("Folder {} does not containt \"Calibration\" folder".format(calibration_path))
			# 	continue
			
			#Data iteration

			for file in os.listdir(data_path):
				full_filename = os.path.join(data_path,file)

				_, ext = os.path.splitext(full_filename)
				if ext != ".png" and ext != ".pgm":
					continue
					
				_id,_name,_type,_ = file.split("_")
				unique_name = recording + "-" + str(_id)
				
				#skip names that we do not want to load
				if _name not in self.params.device_list:
					continue

				if unique_name not in self.data:
					self.data[unique_name] = {}
					#self.data[unique_name]["calibration"] = calibration_path

				if _name not in self.data[unique_name]:
					self.data[unique_name][_name] = {}

				self.data[unique_name][_name][_type] = full_filename
		print("Data loading completed.")


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		#get an entry
		key = list(self.data.keys())[idx]
		datum = self.data[key]

		datum_out = {}
		for device in self.params.device_list:
			_, depth_ext = os.path.splitext(datum[device]["depth"])
			depth_scale = 0.001 if depth_ext == ".png" else 0.0001
			#color_img = importers.image.load_image(datum[device]["color"])
			depth = torch.from_numpy(numpy.array(cv2.imread(datum[device]["depth"], cv2.IMREAD_ANYDEPTH)).astype(numpy.float32)).unsqueeze(0).unsqueeze(0) * depth_scale
			depth_range_mask = (depth < self.params.depth_threshold).float()
			#depth_img = importers.image.load_depth(datum[device]["depth"], scale=depth_scale) * depth_range_mask
			depth_img = depth * depth_range_mask
			intrinsics, intrinsics_inv = get_intrinsics(\
				device, self.device_repository, self.params.decimation_scale)
			# extrinsics, extrinsics_inv = importers.extrinsics.load_extrinsics(\
			# 	os.path.join(datum["calibration"], device + ".extrinsics"))
			
			datum_out.update({
				#"color" : color_img.squeeze(0),
				"depth" : depth_img.squeeze(0), 
				"intrinsics" : intrinsics.float(),
				"intrinsics_original" : torch.zeros((4)),
				"normals" : torch.zeros((3,depth.shape[2],depth.shape[3])).float(),
				"labels" : torch.zeros_like(depth.squeeze(0)).type(torch.uint8),
				"color" : torch.zeros((4,depth.shape[2],depth.shape[3])).type(torch.uint8),
				"camera_resolution" : (-1.0,-1.0),
            	"camera_pose" : torch.zeros((4,4)).float(),
				#"intrinsics_inv" : intrinsics_inv,
				#"extrinsics" : extrinsics,
				#"extrinsics_inv" : extrinsics_inv,
                "type": "real"
				})
		
		return datum_out

	def get_data(self):
		return self.data