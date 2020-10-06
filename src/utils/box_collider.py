from ..io import box_model_loader as box_loader
import torch

class BoxCollider(object):

    def __init__(self, box_scale : float = 0.001, box_file_path : str = './data/asymmetric_box.obj', device : str = 'cuda'):
        self._box = box_loader.load_box_model(box_file_path)
        self._init_bboxes(device)
        self._box_scale = box_scale

    def _init_bboxes(self, device : str):

        box_names = ["mid_bottom","mid_top","bottom","top"]

        self._bboxes = torch.zeros((len(box_names), 3, 2)).to(device)               # bboxes: bboxes[bbox_no,coordinate x,y,z, min/max]
        side_names = self._box["side_names"]
        side_idx_pack = list(zip(range(len(side_names)),side_names))
        for i in range(len(box_names)):            
            
            sides = list(filter(lambda x: x[1].startswith(box_names[i]),side_idx_pack))
            # 4 vertices per side
            #verts = [ torch.Tensor(self._box["vertices"][i*4*3 + k*4*3:i*4*3 + (k+1)*4*3]).reshape((4,3)).unsqueeze(0).to(device) for k in range(len(sides)) ]
            verts = [ torch.Tensor(self._box["vertices"][sides[k][0]*4*3: (sides[k][0]+1)*4*3]).reshape((4,3)).unsqueeze(0).to(device) for k in range(len(sides)) ]
            verts_cat = torch.cat(verts,dim=0)          # side_count x vertex_count x 3
            verts_p = verts_cat.view(-1,3)
            
            self._bboxes[i,:,0] = torch.min(verts_p,dim=0)[0]            # torch min returns tuple (values,indices)
            self._bboxes[i,:,1] = torch.max(verts_p,dim=0)[0]            # torch max returns tuple (values,indices)

            #vert = torch.Tensor(self._box["vertices"][i*4*3:(i+1)*4*3]).reshape((3,4)).to(device)

    def _is_inside_bbox(self, points : torch.Tensor, bbox : torch.Tensor):
        ''' points: batch x 3 x point_count
            bbox: 3 x 2 (bbox_min | bbox_max)

            returns mask batch x point_count -> [0,1]
        '''

        points_trans = points.transpose(1,2)

        mask_max = points_trans <= bbox[:,1]            # broadcast semantics
        mask_min = points_trans >= bbox[:,0]

        mask = torch.sum((mask_min & mask_max).to(torch.int), dim = 2) == 3
        return mask


    def is_inside_box(self, points : torch.Tensor, extrude_factor : float = 1.0) -> torch.Tensor:
        ''' points: batch x 3 x point_count
            returns mask batch x point_count -> [0,1]
        '''

        masks = [self._is_inside_bbox(points,extrude_factor * self._box_scale * self._bboxes[i,:,:]) for i in range(self._bboxes.shape[0])]

        for i in range(len(masks)):
            if(i==0):
                fmask = masks[i]
            else:
                fmask |= masks[i]
        
        return fmask
