import torch
from torch import nn

import os
import sys

def initialize_weights(model, init = "xavier"):    
    if init == "xavier":
        init_func = nn.init.xavier_normal
    elif init == "kaiming":
        init_func = nn.init.kaiming_normal
    elif init == "gaussian" or init == "normal":
        init_func = nn.init.normal
    else:
        init_func = None
    if init_func is not None:
        #TODO: logging /w print or lib
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) \
                or isinstance(module, nn.ConvTranspose2d):
                    init_func(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
    elif os.path.exists(init):
        #TODO: logging /w print or lib
        weights = torch.load(init)
        model.load_state_dict(weights["state_dict"])        
    else:
        print("Error when initializing model's weights, {} either doesn't exist or is not a valid initialization function.".format(init), \
            file=sys.stderr)

