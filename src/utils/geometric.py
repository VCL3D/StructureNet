import torch
from src.utils import projections
from src.io import box_model_loader, plywrite
from src.dataset.rendering.box_renderer import BoxRenderFlags
from enum import Flag, auto
#import .projections
class BoxRenderFlags (Flag):    
    LABEL_UP_AS_BACKGROUND = auto()
    LABEL_DOWN_AS_BACKGROUND = auto()
    LABEL_TOP_AND_BOTTOM_AS_BACKGROUND = LABEL_UP_AS_BACKGROUND | LABEL_DOWN_AS_BACKGROUND

import os.path as osp
'''
Given two sets (set1 and set2 with dimensions b,c,N [b:batch,c:channels,N:spatial])
computes the (non_rigid by default, rigid if scale is ignored) transformation
from set1 to set2.
This function uses SVD as described here(http://nghiaho.com/?page_id=671)
,naming is consistent wherever possible.
'''
def computeNonRigidTransformation(
    source    :torch.tensor,
    target    :torch.tensor
):
    if len(source.shape) == 2:
        source = source.unsqueeze(0)

    if len(target.shape) == 2:
        target = target.unsqueeze(0)

    b1, c1, N1 = source.shape
    b2, c2, N2 = target.shape

    assert b1==b2, "Batch sizes differ" #TODO: maybe change later so that it could support b1=K, b2=1
    assert c1==c2, "Inputs channels differ"
    assert N1==N2, "Number of samples differ"

    b, c, N = b1, c1, N1

    if source.dtype != torch.double:
        source = source.double()

    if target.dtype != torch.double:
        target = target.double()

    device = source.device

    centroid_source = torch.mean(source, dim = 2).unsqueeze(-1)    #bxcx1
    centroid_target = torch.mean(target, dim = 2).unsqueeze(-1)    #bxcx1

    H_source = source - centroid_source
    H_target = target - centroid_target

    variance_source = torch.sum(H_source**2)
    variance_source = torch.einsum('...bij->...b',H_source**2)

    H = torch.einsum('...in,...jn->...ijn',H_source,H_target)
    H = torch.sum(H, dim = -1)


    list_R, list_t, list_scale = [], [], []

    # care https://github.com/pytorch/pytorch/issues/16076#issuecomment-477755364
    for _b in range(b):
        #assert torch.abs(torch.det(H[_b])).item() > 1.0e-15, "Seems that H matrix is singular"
        U,S,V = torch.svd(H[_b])
        R = torch.matmul(V, U.t())

        Z = torch.eye(R.shape[0]).double().to(device)
        Z[-1,-1] *= torch.sign(torch.det(R))

        R = torch.mm(V,torch.mm(Z,U.t()))

        scale = torch.trace(torch.mm(R,H[_b])) / variance_source[_b]

        list_R.append(R.unsqueeze(0))
        list_scale.append(scale.unsqueeze(0).unsqueeze(-1))


    R = torch.cat(list_R, dim = 0)
    scale = torch.cat(list_scale, dim = 0).unsqueeze(-1)
    t = -torch.bmm(R,centroid_source) + centroid_target
    return R, t, scale
    
'''
Function that estimates the median 3D position of all 
segments
INPUTS:
labels          : b,c,H,W where c is # of labels
depth           : b,1,H,W 
intrinsics_inv  : b,3,3
OUTPUT
points          : b,3,c
'''
def computeLabelsMedianPoint(
    labels          :   torch.tensor,
    depth           :   torch.tensor,
    intrinsics_inv  :   torch.tensor
):
    b, _, h, w = depth.shape
    c = labels.shape[1]
    device = depth.device
    grid = projections.create_image_domain_grid(width = w, height= h).to(device)

    pointcloud = projections.deproject_depth_to_points(depth,grid, intrinsics_inv)
    median_points = torch.zeros((b,3,c)).to(device)
    visible_sides = torch.zeros((b,c)).byte().to(device)
    
    for i in range(b):
        for j in range(c):
            z = torch.nonzero(labels[i,j,:,:])
            if z.shape[0] > 200:
                visible_sides[i,j] = 1
                median_points[i,:,j] = torch.median(pointcloud[i,:,z[:,0],z[:,1]].view(3,-1),dim = -1)[0]
    
    return median_points, visible_sides, pointcloud


def compute_center_of_visible_box(id, box):
            vertices = torch.tensor(box['vertices']).reshape(-1,3)
            box_width = (vertices[box['index_map'][0]] - vertices[box['index_map'][1]]).norm()
            box_height = (vertices[box['index_map'][0]] - vertices[box['index_map'][2]]).norm()
            box_depth = (vertices[box['index_map'][22]] - vertices[box['index_map'][21]]).norm()
            if id == 2:
                center =    vertices[box['index_map'][3]] +\
                            vertices[box['index_map'][5]] +\
                            torch.tensor([
                                vertices[box['index_map'][2]][0],
                                vertices[box['index_map'][2]][1] + box_depth,
                                vertices[box['index_map'][2]][2]]) +\
                            torch.tensor([
                                vertices[box['index_map'][6]][0],
                                vertices[box['index_map'][6]][1] + box_depth,
                                vertices[box['index_map'][6]][2]])

                return center / 4
            elif id == 3:
                center =    vertices[box['index_map'][1]] +\
                            vertices[box['index_map'][7]] +\
                            torch.tensor([
                                vertices[box['index_map'][0]][0],
                                vertices[box['index_map'][0]][1] + box_depth,
                                vertices[box['index_map'][0]][2]]) +\
                            torch.tensor([
                                vertices[box['index_map'][17]][0],
                                vertices[box['index_map'][17]][1] + box_depth,
                                vertices[box['index_map'][17]][2]])

                return center / 4
            elif id == 8:
                center =    vertices[box['index_map'][30]] +\
                            vertices[box['index_map'][26]] +\
                            torch.tensor([
                                vertices[box['index_map'][29]][0] - box_depth,
                                vertices[box['index_map'][29]][1],
                                vertices[box['index_map'][29]][2]]) +\
                            torch.tensor([
                                vertices[box['index_map'][27]][0] - box_depth,
                                vertices[box['index_map'][27]][1],
                                vertices[box['index_map'][27]][2]])

                return center / 4
            elif id == 9:
                center =    vertices[box['index_map'][36]] +\
                            vertices[box['index_map'][38]] +\
                            torch.tensor([
                                vertices[box['index_map'][37]][0] - box_depth,
                                vertices[box['index_map'][37]][1],
                                vertices[box['index_map'][37]][2]]) +\
                            torch.tensor([
                                vertices[box['index_map'][31]][0] - box_depth,
                                vertices[box['index_map'][31]][1],
                                vertices[box['index_map'][31]][2]])

                return center / 4
            elif id == 13:
                center =    vertices[box['index_map'][53]] +\
                            vertices[box['index_map'][55]] +\
                            torch.tensor([
                                vertices[box['index_map'][54]][0] + box_depth,
                                vertices[box['index_map'][54]][1],
                                vertices[box['index_map'][54]][2]]) +\
                            torch.tensor([
                                vertices[box['index_map'][48]][0] + box_depth,
                                vertices[box['index_map'][48]][1],
                                vertices[box['index_map'][48]][2]])

                return center / 4
            elif id == 19:
                center =    vertices[box['index_map'][77]] +\
                            vertices[box['index_map'][78]] +\
                            torch.tensor([
                                vertices[box['index_map'][76]][0],
                                vertices[box['index_map'][76]][1] - box_depth,
                                vertices[box['index_map'][76]][2]]) +\
                            torch.tensor([
                                vertices[box['index_map'][75]][0],
                                vertices[box['index_map'][75]][1] - box_depth,
                                vertices[box['index_map'][75]][2]])

                return center / 4
            
            return torch.tensor(box["vertices"])\
                .reshape(-1,4,3).permute(2,1,0).mean(dim = 1)[:,id]

'''
Module that is used to compute the (rough) extrinsics transformation.
Base idea is, given a depth map and its camera intrinsics (therefore pointcloud)
and the corresponding labels, compute the (non rigid in case of sigma) rigid 
transformation to the "BOX" or "global" coordinate system.
'''
class ExtrinsicsCalculator(torch.nn.Module):
    def __init__(self, box_path, device, render_flags):
        def _label_as_background(side_name : str) -> bool:
            if (render_flags == None):
                return False
            elif (render_flags & BoxRenderFlags.LABEL_DOWN_AS_BACKGROUND) and ("_down_" in side_name):
                return True
            elif (render_flags & BoxRenderFlags.LABEL_UP_AS_BACKGROUND) and ("_up_" in side_name):
                return True           

            return False

        
        super(ExtrinsicsCalculator,self).__init__()
        self.device = device
        self.box = box_model_loader.load_box_model(box_path)

        vertices = torch.tensor(self.box["vertices"]).reshape(-1,3)


        valid_ids = []
        for i in range(len(self.box["side_names"])):
            if not _label_as_background(self.box["side_names"][i]):
                valid_ids.append(i)

        # self.box_sides_center2 = torch.tensor(self.box["vertices"])\
        #     .reshape(-1,4,3).permute(2,1,0).mean(dim = 1).to(device)[:,valid_ids]/ 1000.0 ####
        self.box_sides_center = torch.cat([compute_center_of_visible_box(i, self.box).unsqueeze(-1) for i in valid_ids], dim = 1)/ 1000.0
        self.box_sides_center = self.box_sides_center.to(self.device)
        

            

    def forward(
        self,
        depthmap        :   torch.tensor,#b,1,h,w
        labels          :   torch.tensor,#b,c,h,w
        intrinsics      :   torch.tensor #b,3,3
    ):
        b,c,h,w = labels.shape

        extrinsics = torch.eye(4)\
            .expand(b,4,4)\
            .to(depthmap.device)
        scales = []

        sides, visible_sides, pointclouds = computeLabelsMedianPoint(
            labels,
            depthmap,
            intrinsics.inverse()
        )

        for i in range(b):
            R,t,scale = computeNonRigidTransformation(
                sides[i,:,visible_sides[i].bool()][:,1:], #ignore background registration
                self.box_sides_center[:,visible_sides[i,1:].bool()].to(depthmap.device)
            )
            extrinsics[i,:3,:3] = R
            t = torch.zeros_like(t).to(t.device) if True in (t!=t) else t
            extrinsics[i,:3,3] = t.squeeze(-1)
            scales.append(scale)
        
        scales = torch.cat(scales,dim = 0).float()

        return extrinsics, scales, pointclouds


    def forward_pointcloud(
        self,
        sides           :   torch.tensor, #3D positions of every channel
        visible_sides   :   torch.tensor # visible sides of the boxes
    ):
        b = sides.shape[0]

        extrinsics = torch.eye(4)\
            .expand(b,4,4)\
            .to(sides.device)
        scales = []

        for i in range(b):
            # if torch.sum(torch.sum(visible_sides[i])) < 3:
            #     continue
            R,t,scale = computeNonRigidTransformation(
                sides[i,:,visible_sides[i].bool().squeeze()].unsqueeze(0), #ignore background registration
                self.box_sides_center[:,visible_sides[i].bool().squeeze()[1:]].unsqueeze(0)
            )
            extrinsics[i,:3,:3] = R
            extrinsics[i,:3,3] = t.squeeze(-1)
            scales.append(scale)
        
        if scales:
            scales = torch.cat(scales,dim = 0).float()
        else:
            scales = 10.0 * torch.ones((b))

        return extrinsics, scales
    
    def computeLoss(
        self,
        validity_threshold, #label occurences threshold
        labels, #one hot enc
        pclouds
    ):
        def isnan(x):
            return x!=x

        epsilon = 0.0005
        b,c,h,w = labels.shape

        valid = (labels.view(b,c,-1).sum(dim = -1) > validity_threshold)[:,1:] #exclude backghround
        if valid.sum() == 0:
            return None,None,None


        pred_center = torch.einsum("bthw, bchw -> btc", pclouds, labels.float()) \
            / (labels.float().sum(dim = (-1,-2)).unsqueeze(1)  + epsilon)# bx3xc


        residuals = (self.box_sides_center.unsqueeze(0) - pred_center[:,:,1:])
        loss = (((residuals**2).sum(dim = 1) * valid.float()).sum(dim = -1) / valid.sum(dim = -1).float()).sqrt()
        #loss1 = torch.sqrt(torch.einsum("btc, bc -> b", residuals**2, valid.float())) / valid.sum(dim = -1).float()
        #loss = torch.sqrt(torch.einsum("btc, bc -> b", residuals**2, valid.float()))


        for i in range(b):
            print("Visible sides : {} , mean error {} meters.".format(valid[i].sum(), loss[i]))
        return loss, valid.sum(dim = -1), pred_center[:,:,1:].unsqueeze(-1)




'''
Function that is used to compute soft 3D correspondences 
between network prediction and ground truth labels.
'''
def compute_soft_correspondences(
    pred_labels             :   torch.tensor, #NOT log probability
    depth_maps              :   torch.tensor,
    inverse_intrinsics      :   torch.tensor,
    gt_labels               :   torch.tensor,
    confidence              :   float
):
    epsilon = 1.0e-05
    conf_threshold = 0.0
    b,c,h,w = pred_labels.shape
    device = gt_labels.device

    grid = projections.create_image_domain_grid(width = w, height = h).to(device)
    pointclouds = projections.deproject_depth_to_points(depth_maps, grid, inverse_intrinsics)

    soft_correspondences_pred = torch.zeros((b,c,3)).to(device)
    soft_correspondences_gt = torch.zeros((b,c,3)).to(device)
    visibility = torch.zeros((b,c)).float().to(device)

    mask_gt = torch.zeros_like(pred_labels)
    mask_gt.scatter_(1,gt_labels.long(),1) #b,c,h,w {0,1}

    mask_pred = (pred_labels > confidence).float() #b,c,h,w {0,1}

    pred_masked = pred_labels * mask_pred #b,c,h,w [0,1]

    weights = pred_labels * mask_pred #b,c,h,w [0,1]
    soft_correspondences_pred = torch.einsum("bthw, bchw -> bct", pointclouds, pred_masked) / (torch.sum(weights, dim = [-1, -2]) + epsilon).unsqueeze(-1)
    soft_correspondences_gt = torch.einsum("bthw, bchw -> bct", pointclouds, mask_gt) / (torch.sum(mask_gt, dim = [-1, -2]) + epsilon).unsqueeze(-1)
        

    visibility = (torch.sum(mask_gt, dim = [-1 , -2]) !=0 ).float()

    return soft_correspondences_pred.permute(0,2,1), soft_correspondences_gt.permute(0,2,1), visibility.unsqueeze(1)


'''
Function that computes soft correspondences between network predictions
and the ground truth labels from the box.
'''
def compute_soft_correspondences_unlabeled(
    pred_labels             :   torch.tensor, #NOT log probability
    depth_maps              :   torch.tensor,
    inverse_intrinsics      :   torch.tensor,
    confidence              :   float,
    confidence_number       :   int
):
    epsilon = 1.0e-05
    b,c,h,w = pred_labels.shape
    device = pred_labels.device

    grid = projections.create_image_domain_grid(width = w, height = h).to(device)
    pointclouds = projections.deproject_depth_to_points(depth_maps, grid, inverse_intrinsics)

    visibility = torch.zeros((b,c)).to(device).bool()
    soft_correspondences_pred = torch.zeros((b,c,3)).to(device)

    predicted_labels_1d = torch.argmax(pred_labels,dim = 1).float()

    ### find which labels are seen in the predicted tensor given the confidence threshold
    raise Exception("Make it with einsum as labeled")
    for i in range(1,c): #skip background
        mask_pred = (pred_labels[:,i,:,:] > confidence).float()#b,c,h,w
        #visibility[:,i] = (torch.sum(pred_labels[:,i,:,:].view(b,-1), dim = -1) >=confidence_number).float()
        visibility[:,i] = torch.sum((predicted_labels_1d * mask_pred).view(b,-1) == i, dim = -1) >= confidence_number

        weights = (pred_labels[:,i,:,:] * mask_pred).unsqueeze(1)
        pointclouds_masked_pred = pointclouds * weights
        mean_point_pred = torch.sum(pointclouds_masked_pred.view(b,3,-1), dim = -1) / \
            (torch.sum(weights.view(b,1,-1), dim = -1) + epsilon)

        soft_correspondences_pred[:,i,:] = mean_point_pred

    
    return soft_correspondences_pred.permute(0,2,1), visibility.unsqueeze(1)


