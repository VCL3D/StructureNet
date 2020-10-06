import argparse
import os
import sys
import datetime

import torch
import torch.nn as nn

import src.models as models
import src.dataset.box_pose_dataset_factory as dataset_factory
import src.dataset.samplers.pose.pose_sampler as pose_sampler
import src.dataset.samplers.intrinsics_generator as intrinsics_generator
import src.dataset.depthmap_dataset as depthmap_dataset
from src.utils import geometric
from src.io import plywrite, box_model_loader
#from src.io import multidimentional_imsave
from src.utils.image_utils import colorize_label_map, get_color_map_nclasses_17, get_color_map_nclasses_25, get_color_map_nclasses_21

import cv2
import numpy as np
import random
import subprocess
from tqdm import tqdm

from src.utils import projections
from src.dataset.rendering.box_renderer import BoxRenderFlags
from src.utils.geometric import compute_soft_correspondences

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(args):
    usage_text = (
        "StructureNet inference."
        "Usage:  python inference.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # evaluation
    parser.add_argument('--confidence_threshold', type = float, default = 0.0, help ='confidence probability threshold to reject uncofident predictions')
    parser.add_argument('--scale', type = float, default = 0.001, help = 'Factor that converts input to meters')
    # gpu
    parser.add_argument('--batch_size', type = int, default = 24, help = 'Batch size for inference')
    # paths
    parser.add_argument('--input_path', type = str, help = "Path to the input depth maps to test against")
    parser.add_argument('--output_path', type = str, help = "Path to output directory")

    #model
    parser.add_argument('--saved_params_path', default = "default", type=str, help = 'Path to model params file')   
    parser.add_argument('--nclasses', default = 25, type=int, help = 'Number of classes of the model, if not defined inside the checkpoint file')     
    parser.add_argument('--ndf', default=8, type = int,help = 'Ndf of model')
    
    # hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    # debug
    parser.add_argument('--debug', type=int, default=0, help ="debug output. 1 true, 0 false")

    return parser.parse_known_args(args)




def inference(args,device):

    #create model parameters
    model_params = {
        'width': 320,
        'height': 180,
        'ndf': 32,
        'upsample_type': "nearest",
    }

    #random setup
    rnd_seed = 1234
    random.seed(rnd_seed)       # this will generate fixed seeds of subcomponents that create the datasets (factory uses random.random() to initialize seeds)
    torch.random.manual_seed(rnd_seed)

    print("Loading previously saved model from {}".format(args.saved_params_path))
    checkpoint = torch.load(args.saved_params_path)
    
    color_func = { 
        17 : get_color_map_nclasses_17,
        21 : get_color_map_nclasses_21,
        25 : get_color_map_nclasses_25
    }

    model_name = checkpoint['model_name']
    if 'nclasses' in checkpoint:
        nclasses = checkpoint['nclasses']
    else:
        nclasses = args.nclasses

    if 'ndf' in checkpoint:
        model_params['ndf'] = checkpoint['ndf']
    
    model_params['nclasses'] = nclasses
    model = models.get_UNet_model(model_name, model_params)    
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)    
    model.eval()
        
    print('Loaded model name: {}'.format(model_name))

    datasetparams = depthmap_dataset.DepthmapDatasetParams(args.input_path, args.scale)       # scale millimeters to meters
    dsiterator = depthmap_dataset.DepthmapDataset(datasetparams)
    
    dataset =  torch.utils.data.DataLoader(dsiterator,\
            batch_size = args.batch_size, shuffle=False,\
            num_workers = 0, pin_memory=False)

    confidence_threshold = args.confidence_threshold
    frame_index = 0

    pbar = tqdm(total=dataset.__len__())


    for batch_id, batch in enumerate(dataset):
            
        #resize input
        _,_,h,w = batch.shape
        batch_d = nn.functional.interpolate(batch, size=[180, 320], mode='nearest').to(device)

        #inference

        pred = model(batch_d)
        if (len(pred) == 2):
            activs, out = pred
        elif (len(pred) == 3):
            activs, heat_pred, out = pred
        elif (len(pred) == 4):
            activs, heat_pred, out, normals = pred
        else:
            print("unexpected model return value. expected tuple of length 2, 3 or 4.")
            break
        
        batch_size = batch.shape[0]
        for index in range(batch_size):
       
            fpath_label_pred = args.output_path + "\\" + str(frame_index) + '_label_pred.png'
       
            confidence_t, labels_pred_t = out[index].max(0)       
            confidence_t = torch.exp(confidence_t)                          # convert log probability to probability
            labels_pred_t [confidence_t < confidence_threshold] = nclasses  # uncertain classs
            

            labels_pred_t = nn.functional.interpolate(labels_pred_t.unsqueeze(0).unsqueeze(0).float(), size=[h, w], mode='nearest').to(device).squeeze().long()
            labels_pred = labels_pred_t.cpu().data.numpy()            

            labels_pred_n = colorize_label_map(labels_pred, color_func[nclasses]())


            cv2.imwrite(fpath_label_pred,labels_pred_n)
            fpath_normals_gt = args.output_path + "\\" + str(frame_index) + '_normals_gt.png'


            

            frame_index += 1
            pbar.update()
        


if __name__ == "__main__":
    args, uknown = parse_arguments(sys.argv)
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device = torch.device("cuda:{}" .format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else "cpu")

    inference(args, device)