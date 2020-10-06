import numpy as np
import cv2
from enum import Enum

def random_crop_and_scale_to_fit_target(src_img : np.array, target_width : int, target_height : int, rng : np.random.RandomState = np.random.RandomState()):

    # case 1: no cropping because size fits
    if src_img.shape[0] == target_height and src_img.shape[1] == target_width:
        return src_img

    crop_startY = 0
    crop_startX = 0
    img_height = src_img.shape[0]
    img_width = src_img.shape[1]

    
    s = np.min([float(img_width) / float(target_width), float(img_height) / float(target_height)])

    cw = np.min([int(s * target_width), img_width])
    ch = np.min([int(s * target_height), img_height])

    crop_startX = rng.randint(0,img_width-cw+1)
    crop_startY = rng.randint(0,img_height-ch+1)

    cropped_img = src_img[crop_startY:crop_startY+ch,crop_startX:crop_startX+cw]
    resized_img = cv2.resize(cropped_img, (target_width, target_height),interpolation = cv2.INTER_LINEAR)
    return resized_img

def get_color_map_nclasses_25() : 

    colors = [
       #0x12bcea,
       0x000000, # background
        # blue
       0x050c66, # mid bottom front 2f
       0x0b1ae6, # mid bottom back 2b
       0x000010, # mid bottom down 2d
       0x313466, # mid bottom up 2u
       0x4754ff, # mid bottom right 2r
       0x0a15b8, # mid bottom left 2l
        # green
       0x3bff5b, # mid top right 3r
       0x00b81e, # mid top left 3l
       0x001000, # mid top down 3d
       0x2c6636, # mid top up 3u
       0x006611, # mid top front 3f
       0x00e626, # mid top back 3b
        # yellow
       0xffd640, # bottom right 1r
       0x665a2e, # bottom up 1u
       0xe6b505, # bottom back 1b
       0x665002, # bottom front 1f
       0x101000, # bottom down 1d
       0xb89204, # bottom left 1l
        # red
       0x660900, # top front 4f
       0x100000, # top down 4d
       0xff493a, # top right 4r
       0x66312c, # top up 4u
       0xe61300, # top top back 4b
       0xb30f00, # top left 4l
       
       0x888888     # uncertain (max probability < threshold), class 25
       #0x000000     # uncertain (max probability < threshold), class 25
       #0xff0000     # uncertain (max probability < threshold), class 25
    ]

    return colors

def get_color_map_nclasses_21() : 

    colors = [
       #0x12bcea,
       0x000000, # background
        # blue
       0x050c66, # mid bottom front 2f
       0x0b1ae6, # mid bottom back 2b
       #0x000010, # mid bottom down 2d
       0x313466, # mid bottom up 2u
       0x4754ff, # mid bottom right 2r
       0x0a15b8, # mid bottom left 2l
        # green
       0x3bff5b, # mid top right 3r
       0x00b81e, # mid top left 3l
       #0x001000, # mid top down 3d
       0x2c6636, # mid top up 3u
       0x006611, # mid top front 3f
       0x00e626, # mid top back 3b
        # yellow
       0xffd640, # bottom right 1r
       0x665a2e, # bottom up 1u
       0xe6b505, # bottom back 1b
       0x665002, # bottom front 1f
       #0x101000, # bottom down 1d
       0xb89204, # bottom left 1l
        # red
       0x660900, # top front 4f
       #0x100000, # top down 4d
       0xff493a, # top right 4r
       0x66312c, # top up 4u
       0xe61300, # top top back 4b
       0xb30f00, # top left 4l
       
       0x888888     # uncertain (max probability < threshold), class 25
       #0x000000     # uncertain (max probability < threshold), class 25
       #0xff0000     # uncertain (max probability < threshold), class 25
    ]

    return colors

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

def colorize_label_map(lmap : np.array, colors: list) -> np.array:

    outlmap = np.zeros((lmap.shape[0], lmap.shape[1], 3), dtype = np.uint8)

    for y in range(lmap.shape[0]):
        for x in range(lmap.shape[1]):
            label =lmap[y,x]
            # open cv default is bgr
            outlmap [y,x,:] = [colors[label] & 0xFF, (colors[label] & 0xFF00) >> 8, (colors[label] & 0xFF0000) >> 16]  

    return outlmap

