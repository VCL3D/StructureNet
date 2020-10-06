import torch.optim as optim
import torch
import sys

class OptimizerParameters(object):
    def __init__(self, learning_rate=0.001, momentum=0.9, momentum2=0.999,\
        epsilon=1e-8, weight_decay=0.0005, damp=0):
        super(OptimizerParameters, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.momentum2 = momentum2
        self.epsilon = epsilon
        self.damp = damp
        self.weight_decay = weight_decay

    def get_learning_rate(self):
        return self.learning_rate

    def get_momentum(self):
        return self.momentum

    def get_momentum2(self):
        return self.momentum2

    def get_epsilon(self):
        return self.epsilon

    def get_weight_decay(self):
        return self.weight_decay

    def get_damp(self):
        return self.damp

def get_optimizer(opt_type, model_params, opt_params):
    if opt_type == "adam":
        return optim.Adam(model_params, \
            lr=opt_params.get_learning_rate(), \
            betas=(opt_params.get_momentum(), opt_params.get_momentum2()), \
            eps=opt_params.get_epsilon(),
            weight_decay = opt_params.get_weight_decay() \
        )
    elif opt_type == "sgd":
        return optim.SGD(model_params, \
            lr=opt_params.get_learning_rate(), \
            momentum=opt_params.get_momentum(), \
            weight_decay=opt_params.get_weight_decay(), \
            dampening=opt_params.get_damp() \
        )
    else:
        print("Error when initializing optimizer, {} is not a valid optimizer type.".format(opt_type), \
            file=sys.stderr)
        return None

#not used for now
def get_one_hot_mask(labels, nclasses, batch_size):
    one_hot = torch.zeros(batch_size, nclasses, labels.shape[2], labels.shape[3])
    one_hot.scatter_(1, labels.long(), 1)
    
    return one_hot.long()

# Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = torch.tensor(0.0)
        self.avg = torch.tensor(0.0)
        self.sum = torch.tensor(0.0)
        self.count = torch.tensor(0.0)

    def update(self, val, n=1):
        self.val = val.cpu().detach()
        self.sum += val.cpu().detach() * n
        self.count += n
        self.avg = self.sum / self.count
