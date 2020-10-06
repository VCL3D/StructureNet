import math
import numpy
import torch
from torch import nn

from src.utils import projections

def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * torch.sum(y_pred * y_true, axes)
    denominator = torch.sum(torch.pow(y_pred, 2) + torch.pow(y_true, 2), axes)
    
    return 1 - torch.mean(numerator / (denominator + epsilon))

def cosine_loss(n_pred, n_true):
    npred = n_pred.clone()
    ntrue = n_true.clone()

    return torch.sum(1 - torch.sum(n_pred * ntrue, dim=1, keepdim=True)) / (n_true.shape[0] * n_true.shape[1] * n_true.shape[2] * n_true.shape[3])

def generate_gt_heatmap(target, kernel_size, sigma):
    #get binary mask of the gt
    mask = target.clone()
    mask[mask != 0] = 1

    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    #conv layer will be used for Gaussian blurring
    gconv = nn.Conv2d(1, 1, 3, 1, 1)

    #init kernels with Gaussian distribution
    gconv.weight.data = gaussian_kernel
    gconv.bias.data.fill_(0)
    gconv.weight.requires_grad = False

    heatmap = gconv(mask)
    #heatmap = heatmap / torch.sum(heatmap)

    return heatmap


from src.utils.geometric import compute_soft_correspondences, computeNonRigidTransformation
from time import perf_counter 
import src.utils.transformations as transformations

def soft_correspondences_loss(out,batch, confidence, criterion, device, SVD = False):

    soft_cor_pred, soft_cor_gt, visibility_mask = compute_soft_correspondences(
        out,
        batch["depth"].to(device),
        batch["intrinsics"].inverse().to(device),
        batch["labels"].to(device),
        confidence
    )
    extrinsics = torch.eye(4)\
            .expand(soft_cor_pred.shape[0],4,4)\
            .to(soft_cor_pred.device)

    if not SVD:
        loss = criterion(soft_cor_gt*visibility_mask, soft_cor_pred*visibility_mask)
    else:
        loss = 0.0
        try:
            R,t,scale = computeNonRigidTransformation(soft_cor_gt*visibility_mask, soft_cor_pred*visibility_mask)
            R,t,scale = R.float(),t.float(),scale.float()

        except:
            print("Couldnt compute SVD")
            return None
        
        loss = criterion(scale , torch.ones_like(scale).to(device)) + criterion(R, torch.eye(3).expand_as(R).to(device)) + criterion(t, torch.zeros_like(t).to(device))
        # extrinsics[:,:3,:3] = R
        # extrinsics[:,:3, 3] = t.squeeze()


        # transformed = transformations.transform_points_homogeneous(soft_cor_pred.unsqueeze(-1), extrinsics).squeeze()

        # loss = criterion(transformed, soft_cor_gt)

    return loss.float()

def get_color_map_nclasses_17() : 
    colors = [
       #0x12bcea,
       0x000000, # background
        # blue
       0x050c66, # mid bottom front 2f
       0x0b1ae6, # mid bottom back 2b
       0x4754ff, # mid bottom right 2r
       0x0a15b8, # mid bottom left 2l
        # green
       0x3bff5b, # mid top right 3r
       0x00b81e, # mid top left 3l
       0x006611, # mid top front 3f
       0x00e626, # mid top back 3b
        # yellow
       0xffd640, # bottom right 1r
       0xe6b505, # bottom back 1b
       0x665002, # bottom front 1f
       0xb89204, # bottom left 1l
        # red
       0x660900, # top front 4f
       0xff493a, # top right 4r
       0xe61300, # top top back 4b
       0xb30f00, # top left 4l
       
       0x888888     # uncertain (max probability < threshold), class 25
       #0x000000     # uncertain (max probability < threshold), class 25
       #0xff0000     # uncertain (max probability < threshold), class 25
    ]
    return colors
import numpy as np
def colorize_label_map(lmap : np.array, colors : list) -> np.array:
    outlmap = np.zeros((lmap.shape[0], lmap.shape[1], 3), dtype = np.uint8)

    for y in range(lmap.shape[0]):
        for x in range(lmap.shape[1]):
            label =lmap[y,x]
            # open cv default is bgr
            outlmap [y,x,:] = [colors[label] & 0xFF, (colors[label] & 0xFF00) >> 8, (colors[label] & 0xFF0000) >> 16]
    return outlmap


extrinsics_calculator = None
box_renderer = None
def soft_correspondences_loss_unlabeled(
    out,
    batch,
    confidence,
    confidence_number, #number of labeled pixels that make a prediction valid
    criterion,
    render_flags,
    box_path = './data/asymmetric_box.obj' #sorry
):
    from src.utils.geometric import ExtrinsicsCalculator, compute_soft_correspondences_unlabeled
    from src.utils.transformations import transform_points_homogeneous

    device = out.device
    global extrinsics_calculator
    global box_renderer

    if extrinsics_calculator is None:
        extrinsics_calculator = ExtrinsicsCalculator(box_path, device, render_flags)

    if box_renderer is None:
        import src.dataset.rendering.box_renderer as br
        box_renderer_params = br.BoxRendererParams(render_flags = render_flags)
        box_renderer = br.BoxRenderer(box_scale=0.001)

    predicted_sides, visible_sides = compute_soft_correspondences_unlabeled(
        out,
        batch["depth"].to(device),
        batch["intrinsics"].inverse().to(device),
        confidence,
        confidence_number)

    #loss = torch.tensor(0.0).to(device)
    loss = 0.0
    uvs = projections.project_points_to_uvs(predicted_sides.unsqueeze(-1), batch["intrinsics"].to(device))
    out_argmax = (torch.argmax(out, dim = 1).float() * (torch.max(out,dim = 1)[0] > confidence).float()).float()

    backgrounds = torch.min(out,dim = 1)[0]
    try:
        extrinsics, scales = extrinsics_calculator.forward_pointcloud(predicted_sides, visible_sides)
    except:
        print("couldnt compute svd")
        return loss

    extrinsics_c = extrinsics.clone()
    extrinsics_c[:,:3,:3] = extrinsics[:,:3,:3] * scales

    for i in range(predicted_sides.shape[0]):
        points = predicted_sides[i,:,visible_sides[i].squeeze()].unsqueeze(0).unsqueeze(-1)
        transformed_points = transform_points_homogeneous(points,extrinsics[i].unsqueeze(0))
        l2 = criterion(
            transformed_points.squeeze(),
            extrinsics_calculator.box_sides_center[:,visible_sides[i].squeeze()[1:]].squeeze())
        loss = l2 + torch.mean(out[i,0,:])
        
    return loss
