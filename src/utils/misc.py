import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def unique_colors():
    #http://phrogz.net/tmp/24colors.html
    colors = [
        [255,0,0],
        [255,255,0],
        [0,234,255],
        [170,0,255],
        [255,127,0],
        [191,255,0],
        [0,149,255],
        [255,0,170],
        [255,212,0],
        [106,255,0],
        [0,64,255],
        [237,185,185],
        [185,215,237],
        [231,233,185],
        [220,185,237],
        [185,237,224],
        [143,35,35],
        [35,98,143],
        [143,106,35],
        [107,35,143],
        [79,143,35],
        [0,0,0],
        [115,115,115],
        [204,204,204]
    ]
    return colors