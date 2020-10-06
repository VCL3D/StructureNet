import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
import sys
import warnings
import errno

from .utils.multidimentional_save import saveTensorEXR
from .utils.device_repository import *
from .utils.threed import *
from .utils.image_utils import *
from .utils.save_pointcloud import *
from .utils.misc import *
from .utils.procrustes import *
from .utils.box_model_loader import *
from .utils.calibration_result import *
from .utils.undistortion import *

def parse_arguments(args):
    usage_text = (
        "Segment inputs."
        "Usage:  python segment_inputs.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # paths
    parser.add_argument("-i","--input_path", type = str, help = "Path to data to be segmented", required = True)
    parser.add_argument("-o","--output_path", type = str, help = "Path to output directory where predictions will be saved.")
    parser.add_argument("-n","--network_path", type=str, help = 'Path to model params file', required = True)   
    parser.add_argument("-s","--scale", default = 0.001, type=float, help = 'Depthmap scale to convert to meters')   
    parser.add_argument("-b","--path_to_box", type = str, help = "Calibration structure box path (.obj).", required = True)
    parser.add_argument('--name_pos', type = int, help = "At which position of the filenames is the device name (for id_dename_frameid.pgm is 1)", required=True)
    parser.add_argument("--number_of_pixels", type = int, help = "Number of pixels so that a segmentation is valid.", default = 200)
    parser.add_argument('--device_repository', type = str, help = "Path to device repository", required = True)
    parser.add_argument("--extension", default = "pgm", type=str, help = 'Filename extension')   
    parser.add_argument("--device", type=str, default='cuda', help='Device on which eveything will run')
    parser.add_argument("--confidence", type=float, default='0.75', help='What confidence is a valid prediction')
    parser.add_argument("-q","--save_qualitative_predictions", type = str2bool, help = 'Whethere save qualitative results from network', default=True)
    parser.add_argument("-p","--save_pointclouds", type = str2bool, help = 'Whethere save pointclouds', default=False)
    parser.add_argument("--save_transformed_pointclouds", type = str2bool, help = 'Whethere save pointclouds', default=False)
    parser.add_argument("--save_qualitative_supervision", type = str2bool, help = 'Whethere save qualitative supervision signal', default=True)
    return parser.parse_known_args(args)

def main(argv):
    args, uknown = parse_arguments(argv)


    if not os.path.exists(args.input_path):
        raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), args.input_path)
    
    if not os.path.exists(args.network_path):
        raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), args.network_path)

    if not os.path.exists(args.device_repository):
        raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), args.device_repository)
    
    try:
        if not os.path.exists(args.output_path):
            raise Exception()
    except:
        warnings.warn("Output directory not found, using input directory")
        args.output_path = args.input_path

    ort_session = ort.InferenceSession(args.network_path)
    device_repository = load_intrinsics_repository(args.device_repository)
    with open(args.device_repository, 'r') as json_file:
        intrinsics_repository = json.load(json_file)

    extrinsics_calculator = ExtrinsicsCalculator(args.path_to_box,BoxRenderFlags.LABEL_DOWN_AS_BACKGROUND)
    calibration_writer = CalibrationResult(method = "StructureNet++")

    _,_,h,w = ort_session.get_inputs()[0].shape
    
    files = [x for x in os.listdir(args.input_path) if x.endswith(".pgm")]
    for counter, file in enumerate(files):
        full_name = os.path.join(args.input_path, file)
        depthmap = cv2.imread(full_name, cv2.IMREAD_ANYDEPTH).astype(np.float32) * args.scale 
        
        
        original_depthmap = depthmap
        original_h, original_w = depthmap.shape
        depthmap = cv2.resize(depthmap, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        
        depthmap = depthmap.reshape(1,1,depthmap.shape[0], depthmap.shape[1])
        filename = file.split(".")[0]
        device_name = filename.split("_")[args.name_pos]


        output = ort_session.run(None, {'i' : np.ascontiguousarray(depthmap)})[1][0]
        nclasses, height, width = output.shape

        
        nans = np.argwhere(np.isnan(output))
        output[np.isnan(output)] = 1.0
        
        infs = np.argwhere(np.isinf(output))
        output[np.isinf(output)] = 0.0

        label_map = output.argmax(axis = 0)
        probabilities = np.exp(output.max(axis = 0, keepdims = False))

        label_map[probabilities < args.confidence] = 0


        if nclasses == 17:
            colorized_labels = get_color_map_nclasses_17()
        elif nclasses == 21:
            colorized_labels = get_color_map_nclasses_21()
        elif nclasses == 25:
            colorized_labels = get_color_map_nclasses_25()


        #depthmap = cv2.resize(depthmap[0,0], dsize=(original_w, original_h), interpolation=cv2.INTER_NEAREST)
        depthmap = original_depthmap
        resized_output = np.transpose(cv2.resize(np.transpose(output, (1,2,0)), dsize=(original_w, original_h), interpolation=cv2.INTER_NEAREST),(2,0,1))

        saveTensorEXR(np.exp(resized_output), os.path.join(args.output_path, filename + "_probabilities.exr"))


        if(args.save_qualitative_predictions):
            label_map = cv2.resize(label_map, dsize=(original_w, original_h), interpolation=cv2.INTER_NEAREST)
            colored = colorize_label_map(label_map, colorized_labels)
            cv2.imwrite(os.path.join(args.output_path, filename + "_colorized.png"), colored)

        scale_x,scale_y = compute_scales(device_name, original_w, original_h, intrinsics_repository)
        intrinsics = get_intrinsics(device_name, device_repository, scale_x, scale_y)
        distortion_coeffs = get_distortion_coefficients(device_name, intrinsics_repository)
        pcloud = deproject_depthmap(depthmap, intrinsics, distortion_coeffs)
        normals = compute_normals(pcloud)


        ids, counts = np.unique(label_map, return_counts = True)
        #so far predicted id 1 is side 0 so we have to subtract
        valid_ids = [id - 1 for id in ids[counts > args.number_of_pixels] if id !=0]#remove background
        predicted_centers = np.stack([pcloud[np.transpose(label_map) == id + 1,:].mean(axis = 0) for id in valid_ids])
        ground_truth_centers = np.stack(extrinsics_calculator.box_sides_center)[valid_ids, :]
        #R,t = umeyama(predicted_centers, ground_truth_centers)
        #predicted_centers_transformed = predicted_centers.dot(R) + t

        R,t,error = umeyama(predicted_centers, ground_truth_centers)
        extrinsics = np.eye(4)
        extrinsics[:3,:3], extrinsics[:3,3] = R, t
        
        extrinsics[:3,:3], extrinsics[:3,3] = np.transpose(R), t # this transpose is because of umeyama function
        R,t = extrinsics[:3,:3], extrinsics[:3,3]
        predicted_centers_transformed = extrinsics.dot(np.concatenate([np.transpose(predicted_centers), np.ones((1,predicted_centers.shape[0]))]))[:3,:]

        if args.save_transformed_pointclouds:
            pcloud_homo = np.concatenate([np.transpose(pcloud.reshape(-1,3)), np.ones((1,original_h * original_w))])
            transformed_pcloud = np.transpose(extrinsics.dot(pcloud_homo))[:,:3]#pcloud.dot(R) + t
            transformed_normals = np.transpose(R.dot(np.transpose(normals.reshape(-1,3))))[:,:3]
            save_ply(os.path.join(args.output_path, device_name + "_transformed.ply"), transformed_pcloud, scale = 1.0, normals = transformed_normals, color = unique_colors()[counter])
            save_ply(os.path.join(args.output_path, device_name + "_predicted_centers_transformed.ply"), np.expand_dims(predicted_centers_transformed, axis = 0), scale = 1.0)
            save_ply(os.path.join(args.output_path, device_name + "_ground_truth_centers.ply"), np.expand_dims(ground_truth_centers, axis = 0), scale = 1.0)


        if args.save_pointclouds:
            save_ply(os.path.join(args.output_path, device_name + ".ply"), pcloud, scale = 1.0, normals = normals, color = unique_colors()[counter])

        stacked = np.concatenate([pcloud, normals], axis = -1).astype(np.float32)

        if args.save_qualitative_supervision:
            normals_to_save = np.transpose(normals, (1,0,2))
            normals_to_save = 255 * (normals_to_save - normals_to_save.min())/( normals_to_save.max() - normals_to_save.min())
            cv2.imwrite(os.path.join(args.output_path, filename + "_supervision_signal.png"), normals_to_save.astype(np.uint8))
        saveTensorEXR(np.transpose(stacked,(2,1,0)), os.path.join(args.output_path, filename + "_supervision_signal.exr"))



        calibration_writer.update(device_name, extrinsics, intrinsics, None)

        print("@ Device: {}, correspondences: {}, error: {}".format(device_name, len(valid_ids), error), file = sys.stdout)
        print("% {}".format((counter) / (2 * len(files))), file = sys.stdout)

    calibration_writer.write(os.path.join(args.output_path, "calibration.json"))
    print("* Initial allignment completed, starting CRF refinement.", file = sys.stdout)


if __name__ == "__main__":
    main(sys.argv)