from .UNet_mask_max import *

import sys

def get_model(model, *kwargs):
    if str.lower(model) == "gcn":
        return GCN(kwargs[0])

def get_UNet_model(name, params):
	if name == 'default' or name == 'baseline':
		return UNet_base(params['width'], params['height'], params['ndf'], params['upsample_type'], params['nclasses'])
	elif name == 'full_mask_max':
		return UNet_mask_max(params['width'], params['height'], params['ndf'], params['upsample_type'], params['nclasses'])
	elif name == 'full_max':
		return UNet_max(params['width'], params['height'], params['ndf'], params['upsample_type'], params['nclasses'])
	elif name == 'heatmap':
		return UNet_heat(params['width'], params['height'], params['ndf'], params['upsample_type'], params['nclasses'])
	elif name == 'full_mask':
		return UNet_mask(params['width'], params['height'], params['ndf'], params['upsample_type'], params['nclasses'])
	elif name == 'with_normals':
		return UNet_normals_base(params['width'], params['height'], params['ndf'], params['upsample_type'], params['nclasses'])
	else:
		print("Could not find the requested model ({})".format(name), file=sys.stderr)