import torch
import torchvision.utils as tu
import cv2
import numpy as np
import os
import sys
import argparse
import src.models as models
from src.utils.geometric import ExtrinsicsCalculator, BoxRenderFlags
from src.utils.save_pointcloud import save_ply



STATIC_IMAGE_SIZE = (180, 320) # (height,width)
STATIC_BOX_FLAG = BoxRenderFlags.LABEL_DOWN_AS_BACKGROUND
STATIC_DEVICE = 'cpu'

def parse_arguments(args):
    usage_text = (
        "Calibration script."
        "Usage:  python calibration.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    parser.add_argument("-d","--depth", type = str, help = "Path to depthmap", required = True)
    parser.add_argument("-m","--model_path", type = str, help = "Path to saved model params", required = True)
    parser.add_argument("-o","--save_path", type = str, help = "Path to save results", required = True)
    parser.add_argument("-b","--box_path", type = str, help = "Path to box", default = r"data/asymmetric_box.obj")
    parser.add_argument("-s","--scale", type = float, help = "Factor that converts depthmap to meters")
    parser.add_argument("-i","--intrinsics", nargs=4, metavar=('fx', 'cx', 'fy', 'cy',),
                        help="camera instrinsic factors", type=float,
                        default=None)
    return parser.parse_known_args(args)

def align(
    model       : torch.nn.Module,
    depthmap    : torch.Tensor,
    intrinsics  : torch.Tensor,
    box_path    : str,
    device      : str,
    save_path   : str,
    box_flag    : BoxRenderFlags = STATIC_BOX_FLAG,
    confidence  : float = 0.75,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    predictions = model(depthmap)[1]
    _, nclasses, height, width = predictions.shape
    
    labels = predictions.argmax(dim = 1, keepdim = True)
    one_hot = torch.nn.functional.one_hot(labels.squeeze(),num_classes = nclasses).permute(2,0,1).unsqueeze(0)
    
    extrinsics_calculator = ExtrinsicsCalculator(box_path, device, box_flag)

    extrinsics, _, pointclouds = extrinsics_calculator.forward(depthmap, one_hot, intrinsics)
    extrinsics = extrinsics.squeeze().numpy().T
    pointclouds = pointclouds[0].permute(1,2,0).reshape(-1,3).numpy()
    save_ply(os.path.join(save_path, "original.ply"),pointclouds , scale = 1)
    pcloud_homo = np.concatenate([pointclouds, np.ones((height * width, 1))], axis = 1)
    transformed_pcloud = pcloud_homo.dot(extrinsics)
    save_ply(os.path.join(save_path, "transformed.ply"),transformed_pcloud[:,:3], scale = 1)
    np.savetxt(os.path.join(save_path, "extrinsics.txt"), extrinsics)
    print(extrinsics)



def loadModel(
    path_to_model   : str,
    device          : str
) -> torch.nn.Module:
    print("Loading previously saved model from {}".format(path_to_model))
    checkpoint = torch.load(path_to_model)
    model_params = {
        'width': 320,
        'height': 180,
        'ndf': 32,
        'upsample_type': "nearest",
    }


    model_name = checkpoint['model_name']
    if 'nclasses' in checkpoint:
        nclasses = checkpoint['nclasses']

    if 'ndf' in checkpoint:
        model_params['ndf'] = checkpoint['ndf']
    
    model_params['nclasses'] = nclasses
    model = models.get_UNet_model(model_name, model_params)    
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)    
    model.eval()
    return model
    
def loadData(
    path_to_depthmap    : str,
    scale               : float
) -> torch.Tensor:
    depth_np = cv2.imread(path_to_depthmap, -1).astype(np.float32)
    depth_np = cv2.resize(depth_np, STATIC_IMAGE_SIZE[::-1], interpolation=cv2.INTER_NEAREST)
    depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)  / scale
    return depth_t

if __name__ == "__main__":
    args, _ = parse_arguments(sys.argv)
    intrinsics = torch.FloatTensor([
                    args.intrinsics[0],
                    0.0,
                    args.intrinsics[1],
                    0.0,
                    args.intrinsics[2],
                    args.intrinsics[3],
                    0.0,
                    0.0,
                    1.0
                ]).view((3,3)).unsqueeze(0)

    model = loadModel(args.model_path,STATIC_DEVICE)
    depthmap = loadData(args.depth, args.scale)

    align(
        model,
        depthmap,
        intrinsics,
        args.box_path,
        STATIC_DEVICE,
        args.save_path
    )

