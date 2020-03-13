import math
import numpy
import torch
from torch import nn

def jaccard(pred, gt, nclasses, class_pixel_count_threshold : int = 0):
    y_true = gt.clone()
    y_pred = pred.clone()

    if len(y_pred.shape) == 3:
        y_pred.unsqueeze_(0)
    
    if len(y_pred.shape) != 4:
        raise Exception("Wrong numer of channels")
    bs,_,_,_ = y_pred.shape

    if len(y_true.shape) == 3:
        y_true.unsqueeze_(0)
    if len(y_true.shape) != 4:
        raise Exception("Wrong numer of channels")

    if pred.shape[1] == 1:
        y_pred = torch.zeros_like(y_pred)\
                        .repeat(1,nclasses,1,1)\
                        .scatter_(1,y_pred.long(),1)
    if gt.shape[1] == 1:
        y_true = torch.zeros_like(y_true)\
                        .repeat(1,nclasses,1,1)\
                        .scatter_(1,y_true.long(),1)
        

    a = torch.sum(y_true, dim = (-1, -2))
    b = torch.sum(y_pred, dim = (-1, -2))

    intersection = y_true.bool() & y_pred.bool()
    union = y_true.bool() | y_pred.bool()
    
    false_negative = (b == 0) & (a != 0)
    false_positive = (a == 0) & (b != 0) #prediction says class is visible but is wrong
    true_negative = ((a == 0) & (b == 0)) | (false_negative & (a < class_pixel_count_threshold))        # if false negative but ground truth has only a few pixels in this class, consider this as true negative

    iou = torch.where(
            false_positive,#condition
            torch.zeros_like(false_positive).float(),#if condition is true
            torch.where(#if condtiontion is false
                true_negative,#inner condition
                torch.ones_like(false_positive).float(),#if condition is true
                torch.sum(intersection.float(), dim = (-1,-2)) / torch.sum(union.float(), dim = (-1,-2))).float())#if condition is false

    mask = torch.ones_like(iou).bool()

    return iou, mask