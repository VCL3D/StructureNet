import argparse
import sys
import os
import time
sys.path.append(".")

from src.structurenet_calibration import main as initial_calibration
from src.crf_refinement import main as crf_refinement
from src.g2o_refinement import main as g2o_refinement

from argument_parser import *


def main(argv):
    args, uknown = parse_arguments(argv)

    if args.path_prefix_to_root_folder is not None:
        #args.device_repository = os.path.join(args.path_prefix_to_root_folder, args.device_repository)
        args.path_to_box_obj = os.path.join(args.path_prefix_to_root_folder, args.path_to_box_obj)
        args.network_path = os.path.join(args.path_prefix_to_root_folder, args.network_path)
        args.crf_path = os.path.join(args.path_prefix_to_root_folder, args.crf_path)
        args.g2o_path = os.path.join(args.path_prefix_to_root_folder, args.g2o_path)
        args.path_to_box_ss = os.path.join(args.path_prefix_to_root_folder, args.path_to_box_ss)


    if args.output_path is None:
        args.output_path = args.input_path

    

    #################################################################################
    initial_calibration_path = os.path.join(args.output_path, "initial_calibration")
    
    if not os.path.exists(initial_calibration_path):
        os.makedirs(initial_calibration_path)
        try:
            initial_calibration(
                [
                    "--input_path", args.input_path,
                    "--output_path", initial_calibration_path,
                    "--network_path", args.network_path,
                    "--scale", str(args.scale),
                    "--path_to_box", args.path_to_box_obj,
                    "--name_pos", str(args.name_pos),
                    "--number_of_pixels", str(args.number_of_pixels),
                    "--device_repository", args.device_repository,
                    "--extension", args.extension,
                    "--confidence", str(args.confidence),
                    "--save_pointclouds", "True"
                ]   
            )
        except Exception as e:
            print(e, file = sys.stderr)
            os.removedirs(initial_calibration_path)
            exit(1)
    # #################################################################################

    # #################################################################################
    crf_refined_calibration = os.path.join(args.output_path, "crf_refined_calibration")
    
    if not os.path.exists(crf_refined_calibration):
        os.makedirs(crf_refined_calibration)

        try:
            crf_refinement(
                [
                    "--input_path", initial_calibration_path,
                    "--output_path", crf_refined_calibration,
                    "--executable_path", args.crf_path,
                    "--path_to_box", args.path_to_box_obj,
                    "--name_pos", str(args.name_pos),
                    "--number_of_pixels", str(args.number_of_pixels),
                    "--device_repository", args.device_repository,
                    "--supervision", str(args.supervision)
                ]
            )
        except Exception as e:
            print(e, file = sys.stderr)
            os.removedirs(crf_refined_calibration)
            exit(2)
    #################################################################################

    #################################################################################
    g2o_refined_calibration = args.output_path
    g2o_refinement(
        [
            "--poses", os.path.join(crf_refined_calibration,"calibration.json"),
            "--clouds", crf_refined_calibration,
            "--out", g2o_refined_calibration,
            "--executable_path", args.g2o_path,
            "--reference", args.path_to_box_ss,
            "--scale", "1.0",
            "--global_iters", args.global_iters,
            "--local_iters", args.local_iters,
            "--force_normals",
            "--log", "g2o_log.txt"
        ]
    )
    #################################################################################





if __name__ == "__main__":
    main(sys.argv)
##############################
