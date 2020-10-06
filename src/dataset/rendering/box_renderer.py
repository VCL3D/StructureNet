import numpy as np
import trimesh
import pyrender
from mgen import rotation_around_axis
from ...io import box_model_loader
from .transformations import *
from enum import Flag, auto

class BoxRenderFlags (Flag):    
    LABEL_UP_AS_BACKGROUND = auto()
    LABEL_DOWN_AS_BACKGROUND = auto()
    LABEL_TOP_AND_BOTTOM_AS_BACKGROUND = LABEL_UP_AS_BACKGROUND | LABEL_DOWN_AS_BACKGROUND

class BoxRendererParams:

    def __init__(self, depth_zmin : float = 0.2, depth_zmax : float = 10.0, render_flags : BoxRenderFlags = None):

        self.depth_zmin = depth_zmin
        self.depth_zmax = depth_zmax
        self.render_flags = render_flags

class BoxRenderer(object):

    def __init__(self, box_model_obj_path : str = './data/asymmetric_box.obj', box_scale : float = 1.0, camera_transform : np.array = np.array([ \
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, -1.0, 0.0, 0.0],
                                    [0.0, 0.0,-1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0] 
                                    ]),
                                    box_load_flags : box_model_loader.BoxLoadFlags = box_model_loader.BoxLoadFlags.LOAD_ALL):
        """The units of the model are in millimeters. Use scale to adjust
        camera_transform : numpy 4x4 transformation matrix to apply inorder to align opengl camera space to camera space of the given camera poses
        normally for typical camera models where y is down and origin at top left this camera matrix should be
        camera_transform = np.array([1.0,0.0,0.0,0.0],
                                    [0.0,-1.0,0.0,0.0],
                                    [0.0,0.0,-1.0,0.0],
                                    [0.0,0.0,0.0,1.0])
        """
        
        self._camera_transform = camera_transform
        self._box_model_obj_path = box_model_obj_path
        self._box_load_flags = box_load_flags
        self._background_color = [0.5,0.5,0.5,0.0]
        self._canvas_width = 320
        self._canvas_height = 180
        self._box_scale = box_scale
        self._initialize_offscreen_renderer = True
        self._create()

    def _generate_mesh(self, camera_pose, render_flags : BoxRenderFlags = None):
        
        positions, indices = self._box_geometry
        colors = BoxRenderer._generate_labels_normals_colormap(box_model = self._box_model, camera_pose = camera_pose, 
                camera_transform = self._camera_transform, box_pose = self._box_model_pose, render_flags = render_flags)

        prim = pyrender.Primitive(positions = positions, indices = indices, mode = 4, color_0 = colors)
        mesh = pyrender.Mesh([prim])
        self._box_mesh = mesh

    @staticmethod    
    def _generate_geometry(box_model):
        positions = np.reshape(box_model["vertices"],(-1,3))
        indices = np.reshape(box_model["indices"],(-1,3))
        return positions, indices

    @staticmethod
    def _generate_labels_normals_colormap(box_model, camera_pose : np.array, camera_transform : np.array, box_pose : np.array, render_flags : BoxRenderFlags):
        """
        box_model: box_model object created by box_model_loader
        camera_pose: the camera pose 4x4 numpy array. 
        box_pose: box object pose in global space 4x4 numpy array
        This function will generate colors with r,g,b color coding the normal in camera coordinate system (not global) 
        and alpha containing the label id of the box's side
        """

        def _label_as_background(side_name : str) -> bool:
            if (render_flags == None):
                return False
            elif (render_flags & BoxRenderFlags.LABEL_DOWN_AS_BACKGROUND) and ("_down_" in side_name):
                return True
            elif (render_flags & BoxRenderFlags.LABEL_UP_AS_BACKGROUND) and ("_up_" in side_name):
                return True           

            return False

        normals = np.reshape(box_model["normals"],(-1,3))
        colors = np.zeros((int(len(box_model["vertices"])/3),4))
        label_count = len(box_model["side_names"])

        final_camera_pose = camera_pose @ camera_transform

        inv_final_camera_pose = np.linalg.inv(final_camera_pose)
        next_label_id = 1
        background_label = 0
        for i in range(label_count):
            if not _label_as_background(box_model["side_names"][i]):
                label_id = next_label_id / 255.0
                next_label_id += 1
            else:
                label_id = background_label / 255.0

            for j in range(4):                 # for the 4 vertices of each side
                normal_obj = np.concatenate((normals[i*4+j,:],0.0),axis=None)
                normal_global = np.dot(box_pose,normal_obj)
                eye_norm = np.dot(inv_final_camera_pose,normal_global)
                eye_norm = eye_norm / np.linalg.norm(eye_norm)   
                eye_norm = (eye_norm + 1.0) / 2.0       # normals from (-1.0 1.0) -> (0.0, 1.0)
                eye_norm = eye_norm ** 2.2 # undo gamma correction
                semantic = np.concatenate((eye_norm[:3],label_id),axis=None)
                colors[i*4+j,:] = semantic            

        return colors

    def _generate_scene(self):
        scene = pyrender.Scene()
        scene.ambient_light = [1.0,1.0,1.0]
        self._scene = scene
    
    def _generate_model_pose(self):

        box_model_pose =  np.zeros((4,4))
        box_model_pose[:3,:3] = rotation_around_axis([1,0,0],-np.pi/2)
        box_model_pose[3,3] = 1.0

        box_model_pose = np.dot(box_model_pose,scale_matrix(self._box_scale))
        self._box_model_pose = box_model_pose

    def _add_box_mesh_to_scene(self):     

        if(len(self._scene.mesh_nodes)>0):
            self._scene.remove_node(next(iter(self._scene.mesh_nodes)))

        self._scene.add(self._box_mesh, pose = self._box_model_pose)        
    
    def _create(self):
        self._box_model = box_model_loader.load_box_model(self._box_model_obj_path, flags = self._box_load_flags)
        self._box_geometry = BoxRenderer._generate_geometry(self._box_model)
        self._generate_scene()
        self._generate_model_pose()

    @property
    def canvas_width(self) -> int:
        return self._canvas_width

    @canvas_width.setter
    def canvas_width(self, value : int):
        value = int(value)
        if(self._canvas_width != value):
            self._canvas_width = value
            self._initialize_offscreen_renderer = True

    @property
    def canvas_height(self) -> int:
        return self._canvas_height
    
    @canvas_height.setter
    def canvas_height(self,value : int):
        value = int(value)
        if(self._canvas_height != value):
            self._canvas_height = value
            self._initialize_offscreen_renderer = True
    
    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, value):
        assert(len(value) >= 3 and len(value)<=4)        
        self._background_color = value
    
    def render(self, camera_pose : np.array, camera_intrinsics : np.array , znear : float = 1.0, zfar : float= 100.0, render_flags : BoxRenderFlags = None):
        '''
        camera_pose: numpy 4x4 array of camera pose in global coordinate system
        camera_intrinsics: [fx, fy, cx, cy]: list of 4 floating point values for camera intrinsics (fx,fy,cx,cy in pixels)
        znear: near clipping plane - not relevant to intrinsics - z near defines the clipping of the depth values 
        zfar: far clipping plane - not relevant to intrinsics - z far defines the clipping of the depth values
        '''    

        if(self._initialize_offscreen_renderer):
            self._renderer = pyrender.OffscreenRenderer(self._canvas_width, self._canvas_height)            
            self._initialize_offscreen_renderer = False
        
        if(len(self._scene.camera_nodes)>0):
            self._scene.remove_node(next(iter(self._scene.camera_nodes)))
        
        camera = pyrender.IntrinsicsCamera(fx = camera_intrinsics[0], fy = camera_intrinsics[1], cx = camera_intrinsics[2], cy = camera_intrinsics[3], \
                                            znear = znear, zfar = zfar)

        final_camera_pose = np.dot(camera_pose,self._camera_transform)
        self._scene.bg_color = self._background_color
        self._scene.add(camera, pose = final_camera_pose)
        self._generate_mesh(camera_pose, render_flags)
        self._add_box_mesh_to_scene()
        color, depth = self._renderer.render(self._scene, flags = pyrender.RenderFlags.DISABLE_MULTISAMPLING | pyrender.RenderFlags.RGBA)

        # undo normal color encoding
        normals = (2.0*color[:,:,:3])/255.0 - 1
        labels = color[:,:,3]

        # convert normals to camera coordinate system
        inv_camera_transform = np.linalg.inv(self._camera_transform)
        inv_camera_rot = inv_camera_transform[:3,:3]
        trans_normals = np.dot(inv_camera_rot, normals.reshape((-1,3)).T)
        normals_reshaped = np.reshape(trans_normals.T,normals.shape)
        return color, depth, normals_reshaped, labels
