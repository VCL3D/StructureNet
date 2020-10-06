import argparse
import os
import sys
import errno
import warnings
import subprocess
import cv2

from .utils.multidimentional_load import *
from .utils.box_model_loader import *
from .utils.procrustes import *
from .utils.save_pointcloud import *
from .utils.calibration_result import *
from .utils.misc import *
from .utils.device_repository import *

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
    parser.add_argument("-e","--executable_path", type = str, help = "Path to CRF executable.", required = True)
    parser.add_argument("-b","--path_to_box", type = str, help = "Calibration structure box path (.obj).", required = True)
    parser.add_argument("-n","--number_of_pixels", type = int, help = "Number of pixels so that a segmentation is valid.", default = 200)
    parser.add_argument('--device_repository', type = str, help = "Path to device repository", required = True)
    parser.add_argument('--name_pos', type = int, help = "At which position of the filenames is the device name (for id_dename_frameid.pgm is 1)", required=True)
    parser.add_argument("--supervision", type = str, help = "Path to CRF executable.", required = True, choices=["color_sup", "exr_sup"])
    return parser.parse_known_args(args)

def main(argv):
    args, uknown = parse_arguments(argv)

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), args.input_path)

    if not os.path.exists(args.executable_path):
        raise FileNotFoundError(
        errno.ENOENT, os.strerror(errno.ENOENT), args.executable_path)
    
    try:
        if not os.path.exists(args.output_path):
            raise Exception()
    except:
        warnings.warn("Output directory not found, using input directory")
        args.output_path = args.input_path


    device_names = set([x.split("_")[args.name_pos] for x in os.listdir(args.input_path) if x.endswith(".exr")])
    files = os.listdir(args.input_path)

    extrinsics_calculator = ExtrinsicsCalculator(args.path_to_box,BoxRenderFlags.LABEL_DOWN_AS_BACKGROUND)
    device_repository = load_intrinsics_repository(args.device_repository)
    with open(args.device_repository, 'r') as json_file:
        intrinsics_repository = json.load(json_file)
    calibration_writer = CalibrationResult(method = "StructureNet++")

    for counter, device_name in enumerate(device_names):
        probabilities_path = [x for x in files if (device_name in x) and ("probabilities" in x)][0]
        exr_supervision = [x for x in files if (device_name in x) and ("supervision" in x) and ("exr" in x)][0]
        color_supervision = [x for x in files if (device_name in x) and ("supervision" in x) and ("png" in x)][0]
        if args.supervision == "exr_sup":
            supervision_path = exr_supervision
        else:
            supervision_path = color_supervision

        subprocess.run([args.executable_path, 
                                "--signal", os.path.join(args.input_path,probabilities_path),
                                "--sup_signal", os.path.join(args.input_path,supervision_path),
                                "--out_dir", args.output_path,
                                "--bsc1","10.0",
                                "--bsc2","10.0",
                                "--bsc3","20.0", 
                                "--bsw" , "16.0",
                                "-i", "10",
                                "--" + args.supervision])

        list = loadTensorExr(os.path.join(args.input_path,exr_supervision))
        pcloud, normals = np.concatenate(list[:3], axis = -1), np.concatenate(list[3:], axis = -1)

        w,h,c = pcloud.shape

        post_crf_predictions_path = os.path.join(args.output_path, [x for x in os.listdir(args.output_path) if (device_name in x) and ("post_crf_raw" in x)][0])
        post_crf_labels = np.transpose(cv2.imread(post_crf_predictions_path, cv2.IMREAD_ANYDEPTH))
        
        ids, counts = np.unique(post_crf_labels, return_counts = True)
        #so far predicted id 1 is side 0 so we have to subtract
        valid_ids = [id - 1 for id in ids[counts > args.number_of_pixels] if id !=0]#remove background
        

        predicted_centers = np.stack([pcloud[post_crf_labels == id + 1,:].mean(axis = 0) for id in valid_ids])
        ground_truth_centers = np.stack(extrinsics_calculator.box_sides_center)[valid_ids, :]
        predicted_centers_homo = np.concatenate([predicted_centers.reshape(-1,3), np.ones((predicted_centers.shape[0],1))], axis = 1)
        
        R,t,error = umeyama(predicted_centers, ground_truth_centers)
        extrinsics = np.eye(4)
        extrinsics[:3,:3], extrinsics[:3,3] = np.transpose(R), t # this transpose is because of umeyama function
        R, t = extrinsics[:3,:3], extrinsics[:3,3]


        pcloud_homo = np.concatenate([np.transpose(pcloud.reshape(-1,3)), np.ones((1,h * w))])
        transformed_pcloud = np.transpose(extrinsics.dot(pcloud_homo))[:,:3]#pcloud.dot(R) + t
        transformed_normals = np.transpose(R.dot(np.transpose(normals.reshape(-1,3))))[:,:3]
        

        centers_correspondences = ground_truth_centers - predicted_centers
        centers_correspondences = np.expand_dims(np.transpose(centers_correspondences), axis = -1)
        saving_correspondences = np.expand_dims(np.transpose(np.concatenate([predicted_centers, ground_truth_centers])), axis = -1)
        saving_correspondences_annotations = np.concatenate([centers_correspondences, centers_correspondences], axis = 1)
        

        save_ply(os.path.join(args.output_path, device_name + ".ply"), transformed_pcloud, 1.0,normals=transformed_normals, color = unique_colors()[counter])

        scale_x, scale_y = compute_scales(device_name, w, h, intrinsics_repository)
        intrinsics = get_intrinsics(device_name, device_repository, scale_x, scale_y)
        calibration_writer.update(device_name, extrinsics, intrinsics, None)
        # save_ply(os.path.join(args.output_path, device_name + "_predicted_centers.ply"), np.expand_dims(np.transpose(predicted_centers), axis = 0), 1.0)
        # save_ply(os.path.join(args.output_path, device_name + "_predicted_centers_transformed.ply"), np.expand_dims(np.transpose(predicted_centers_transformed), axis = 0), 1.0)
        # save_ply(os.path.join(args.output_path, device_name + "_ground_truth_centers.ply"), np.expand_dims(np.transpose(ground_truth_centers), axis = 0), 1.0)
        #save_ply(os.path.join(args.output_path, device_name + "_centers_correnspondences.ply"),saving_correspondences, 1.0, normals = saving_correspondences_annotations)

        print("@ Device: {}, correspondences: {}, error: {}".format(device_name, len(valid_ids), error), file = sys.stdout)
        print("% {}".format((len(device_names) + counter ) / (2 * len(device_names))), file = sys.stdout)
    calibration_writer.write(os.path.join(args.output_path, "calibration.json"))
    print("* CRF refinement completed, graph optimization refinement started.", file = sys.stdout)
        


    

    

if __name__ == "__main__":
    main(sys.argv)